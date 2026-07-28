//! Static shape and dtype inference.
//!
//! The interpreter computes shapes at run time from the actual tensors, so
//! this is not what makes a model run. What it buys is knowing, at load time,
//! the type of every value: the caller learns what to feed without guessing,
//! a declared shape that contradicts the graph is caught before any dispatch,
//! and Phase 2's memory planning has something to plan on.
//!
//! **Partial by design.** An op this module does not know produces `Unknown`,
//! not an error: an engine that refuses to load a model because its own
//! inference is incomplete would be worse than one that knows less. What is
//! *not* tolerated is a contradiction — an inferred fixed dimension that
//! disagrees with the one declared in the file means one of the two is wrong,
//! and continuing would build on it.

use onnx_vulkan_core::host_ops::{BOOL, FLOAT, INT32, INT64, UINT8};
use onnx_vulkan_core::{AttrValue, GraphIr, InitializerIr, NodeIr};
use std::collections::HashMap;

/// One dimension. ONNX allows symbolic dimensions (`batch_size`), and losing
/// the symbol would turn "the same unknown value" into "two unknown values".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Fixed(i64),
    Symbol(String),
    Unknown,
}

impl Dim {
    pub fn fixed(&self) -> Option<i64> {
        match self {
            Dim::Fixed(n) => Some(*n),
            _ => None,
        }
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Fixed(n) => write!(f, "{n}"),
            Dim::Symbol(s) => write!(f, "{s}"),
            Dim::Unknown => write!(f, "?"),
        }
    }
}

/// Type of a value: ONNX dtype code plus a shape, either of which may be
/// unknown independently — a `Cast` fixes the dtype even where the shape is
/// still open.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorType {
    pub dtype: Option<i32>,
    pub shape: Option<Vec<Dim>>,
}

impl TensorType {
    fn of(dtype: i32, shape: Vec<Dim>) -> Self {
        Self {
            dtype: Some(dtype),
            shape: Some(shape),
        }
    }

    fn dtype_only(dtype: i32) -> Self {
        Self {
            dtype: Some(dtype),
            shape: None,
        }
    }

    /// Shape with every dimension known, if there is one.
    pub fn concrete(&self) -> Option<Vec<i64>> {
        self.shape
            .as_ref()?
            .iter()
            .map(Dim::fixed)
            .collect::<Option<Vec<_>>>()
    }

    pub fn rank(&self) -> Option<usize> {
        self.shape.as_ref().map(Vec::len)
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.dtype {
            Some(d) => write!(f, "dtype {d}")?,
            None => write!(f, "dtype ?")?,
        }
        match &self.shape {
            Some(dims) => {
                let text: Vec<String> = dims.iter().map(Dim::to_string).collect();
                write!(f, " [{}]", text.join(", "))
            }
            None => write!(f, " [?]"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Conflict {
    pub value: String,
    pub declared: TensorType,
    pub inferred: TensorType,
}

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "valore '{}': il file dichiara {}, il grafo produce {}",
            self.value, self.declared, self.inferred
        )
    }
}

/// Type table for every value in a graph.
#[derive(Debug, Clone, Default)]
pub struct Types(HashMap<String, TensorType>);

impl Types {
    pub fn get(&self, name: &str) -> Option<&TensorType> {
        self.0.get(name)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// How many of `names` have every dimension fixed.
    ///
    /// Lower than [`Self::resolved_count`] by construction: a model with a
    /// symbolic batch has no fixed shape anywhere, which says nothing about
    /// how far inference got.
    pub fn concrete_count<'a>(&self, names: impl Iterator<Item = &'a str>) -> usize {
        names
            .filter(|n| self.0.get(*n).is_some_and(|t| t.concrete().is_some()))
            .count()
    }

    /// How many of `names` have a shape with no unknown dimension left —
    /// symbolic ones count as resolved. **This** is how far inference got:
    /// `[batch, 1000]` is a complete answer, `[?, 1000]` is not.
    pub fn resolved_count<'a>(&self, names: impl Iterator<Item = &'a str>) -> usize {
        names
            .filter(|n| {
                self.0.get(*n).is_some_and(|t| {
                    t.shape
                        .as_ref()
                        .is_some_and(|dims| !dims.contains(&Dim::Unknown))
                })
            })
            .count()
    }
}

/// Infers types for every value, starting from what the file declares.
///
/// `declared` holds the types read from `graph.input` / `output` / `value_info`.
/// Returns the table and the contradictions found; the caller decides whether a
/// contradiction is fatal.
pub fn infer(ir: &GraphIr, declared: &HashMap<String, TensorType>) -> (Types, Vec<Conflict>) {
    let mut types: HashMap<String, TensorType> = HashMap::new();
    let mut conflicts = Vec::new();

    for (name, initializer) in &ir.initializers {
        types.insert(name.clone(), initializer_type(initializer));
    }
    // graph inputs are the only source that cannot be inferred
    for (name, declared_type) in declared {
        types.entry(name.clone()).or_insert(declared_type.clone());
    }

    // constants resolvable at load time: needed by Reshape, Slice, Expand… that
    // take their shape from an input rather than from an attribute
    let mut constants = onnx_vulkan_core::constant_outputs(&ir.nodes);
    constants.extend(ir.initializers.clone());

    for node in &ir.nodes {
        let produced = infer_node(node, &types, &constants);
        for (index, name) in node.outputs.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            let inferred = produced.get(index).cloned().unwrap_or_default();
            if let Some(declared_type) = declared.get(name)
                && let Some(conflict) = disagreement(name, declared_type, &inferred)
            {
                conflicts.push(conflict);
            }
            // what the file declares takes precedence: it is the truth of the
            // model, inference is our reconstruction
            let entry = types.entry(name.clone()).or_default();
            let known = declared.get(name).cloned().unwrap_or(inferred);
            if entry.dtype.is_none() {
                entry.dtype = known.dtype;
            }
            if entry.shape.is_none() {
                entry.shape = known.shape;
            }
        }
    }

    (Types(types), conflicts)
}

/// A contradiction is only a contradiction between two *known* facts: an
/// unknown never disagrees with anything.
fn disagreement(name: &str, declared: &TensorType, inferred: &TensorType) -> Option<Conflict> {
    let build = || Conflict {
        value: name.to_string(),
        declared: declared.clone(),
        inferred: inferred.clone(),
    };
    if let (Some(a), Some(b)) = (declared.dtype, inferred.dtype)
        && a != b
    {
        return Some(build());
    }
    let (Some(a), Some(b)) = (&declared.shape, &inferred.shape) else {
        return None;
    };
    if a.len() != b.len() {
        return Some(build());
    }
    let clash = a.iter().zip(b).any(|(x, y)| match (x, y) {
        (Dim::Fixed(m), Dim::Fixed(n)) => m != n,
        _ => false,
    });
    clash.then(build)
}

fn initializer_type(initializer: &InitializerIr) -> TensorType {
    TensorType::of(
        initializer.dtype,
        initializer.shape.iter().map(|d| Dim::Fixed(*d)).collect(),
    )
}

// ---------------------------------------------------------------- per node

struct Ctx<'a> {
    types: &'a HashMap<String, TensorType>,
    constants: &'a HashMap<String, InitializerIr>,
}

impl Ctx<'_> {
    fn input(&self, node: &NodeIr, index: usize) -> TensorType {
        node.inputs
            .get(index)
            .filter(|n| !n.is_empty())
            .and_then(|n| self.types.get(n.as_str()))
            .cloned()
            .unwrap_or_default()
    }

    fn shape_of(&self, node: &NodeIr, index: usize) -> Option<Vec<Dim>> {
        self.input(node, index).shape
    }

    /// Integer values of an input known at load time.
    fn constant_ints(&self, node: &NodeIr, index: usize) -> Option<Vec<i64>> {
        let name = node.inputs.get(index).filter(|n| !n.is_empty())?;
        let tensor = self.constants.get(name.as_str())?;
        onnx_vulkan_core::HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone())
            .to_i64()
            .ok()
    }
}

fn attr_i64(node: &NodeIr, name: &str, default: i64) -> i64 {
    node.attrs
        .get(name)
        .and_then(AttrValue::as_i64)
        .unwrap_or(default)
}

fn attr_ints(node: &NodeIr, name: &str) -> Option<Vec<i64>> {
    node.attrs
        .get(name)
        .and_then(AttrValue::as_ints)
        .map(<[i64]>::to_vec)
}

/// Normalizes a possibly negative axis against a rank.
fn axis(value: i64, rank: usize) -> Option<usize> {
    let rank = rank as i64;
    let a = if value < 0 { value + rank } else { value };
    (0..rank).contains(&a).then_some(a as usize)
}

fn infer_node(
    node: &NodeIr,
    types: &HashMap<String, TensorType>,
    constants: &HashMap<String, InitializerIr>,
) -> Vec<TensorType> {
    let ctx = Ctx { types, constants };
    let first = ctx.input(node, 0);

    match node.op.as_str() {
        // ---- shape and dtype unchanged from the first input
        "Relu"
        | "Sigmoid"
        | "Tanh"
        | "Erf"
        | "Exp"
        | "Log"
        | "Sqrt"
        | "Abs"
        | "Neg"
        | "Floor"
        | "Ceil"
        | "Round"
        | "Reciprocal"
        | "Sin"
        | "Cos"
        | "Softmax"
        | "LogSoftmax"
        | "LayerNormalization"
        | "BatchNormalization"
        | "InstanceNormalization"
        | "Clip"
        | "LeakyRelu"
        | "Elu"
        | "HardSigmoid"
        | "HardSwish"
        | "Gelu"
        | "Identity"
        | "Dropout"
        | "Not" => {
            vec![first]
        }

        // ---- elementwise with ONNX broadcasting
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max" | "Mod" | "And" | "Or" | "Xor"
        | "Equal" | "Less" | "Greater" | "LessOrEqual" | "GreaterOrEqual" => {
            let shape = broadcast(&ctx.shape_of(node, 0), &ctx.shape_of(node, 1));
            let dtype = if matches!(
                node.op.as_str(),
                "Equal"
                    | "Less"
                    | "Greater"
                    | "LessOrEqual"
                    | "GreaterOrEqual"
                    | "And"
                    | "Or"
                    | "Xor"
            ) {
                Some(BOOL)
            } else {
                first.dtype
            };
            vec![TensorType { dtype, shape }]
        }
        "Where" => {
            let shape = broadcast(
                &broadcast(&ctx.shape_of(node, 0), &ctx.shape_of(node, 1)),
                &ctx.shape_of(node, 2),
            );
            vec![TensorType {
                dtype: ctx.input(node, 1).dtype,
                shape,
            }]
        }

        "Cast" => vec![TensorType {
            dtype: node
                .attrs
                .get("to")
                .and_then(AttrValue::as_i64)
                .map(|d| d as i32),
            shape: first.shape,
        }],

        // ---- shape known statically without inspecting the inputs
        "Shape" => {
            let len = first.rank().map(|r| r as i64);
            vec![TensorType::of(
                INT64,
                vec![len.map(Dim::Fixed).unwrap_or(Dim::Unknown)],
            )]
        }
        "Size" => vec![TensorType::of(INT64, vec![])],
        "ConstantOfShape" => {
            let dtype = match node.attrs.get("value") {
                Some(AttrValue::Tensor(t)) => Some(t.dtype),
                _ => Some(FLOAT),
            };
            vec![TensorType {
                dtype,
                shape: ctx
                    .constant_ints(node, 0)
                    .map(|dims| dims.into_iter().map(Dim::Fixed).collect()),
            }]
        }
        "Constant" => vec![match node.attrs.get("value") {
            Some(AttrValue::Tensor(t)) => initializer_type(t),
            _ => TensorType::default(),
        }],
        "Range" => vec![TensorType {
            dtype: first.dtype,
            shape: Some(vec![Dim::Unknown]),
        }],

        // ---- movement
        "Reshape" => vec![TensorType {
            dtype: first.dtype,
            shape: reshape(&first.shape, ctx.constant_ints(node, 1).as_deref()),
        }],
        "Transpose" => vec![TensorType {
            dtype: first.dtype,
            shape: transpose(&first.shape, attr_ints(node, "perm").as_deref()),
        }],
        "Unsqueeze" => vec![TensorType {
            dtype: first.dtype,
            shape: unsqueeze(
                &first.shape,
                attr_ints(node, "axes").or_else(|| ctx.constant_ints(node, 1)),
            ),
        }],
        "Squeeze" => vec![TensorType {
            dtype: first.dtype,
            shape: squeeze(
                &first.shape,
                attr_ints(node, "axes").or_else(|| ctx.constant_ints(node, 1)),
            ),
        }],
        "Flatten" => vec![TensorType {
            dtype: first.dtype,
            shape: flatten(&first.shape, attr_i64(node, "axis", 1)),
        }],
        "Concat" => vec![TensorType {
            dtype: first.dtype,
            shape: concat(node, &ctx),
        }],

        // ---- reductions
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd" | "ReduceL2"
        | "ArgMax" | "ArgMin" => {
            let dtype = if node.op.starts_with("Arg") {
                Some(INT64)
            } else {
                first.dtype
            };
            let axes = attr_ints(node, "axes")
                .or_else(|| {
                    node.attrs
                        .get("axis")
                        .and_then(AttrValue::as_i64)
                        .map(|a| vec![a])
                })
                .or_else(|| ctx.constant_ints(node, 1));
            vec![TensorType {
                dtype,
                shape: reduce(
                    &first.shape,
                    axes.as_deref(),
                    attr_i64(node, "keepdims", 1) != 0,
                ),
            }]
        }

        // ---- linear algebra
        "MatMul" => vec![TensorType {
            dtype: first.dtype,
            shape: matmul(&ctx.shape_of(node, 0), &ctx.shape_of(node, 1)),
        }],
        "MatMulInteger" => vec![TensorType {
            dtype: Some(INT32),
            shape: matmul(&ctx.shape_of(node, 0), &ctx.shape_of(node, 1)),
        }],
        "Gemm" => vec![TensorType {
            dtype: first.dtype,
            shape: gemm(node, &ctx),
        }],

        // ---- convolution and pooling
        "Conv" | "ConvInteger" => {
            let dtype = if node.op == "ConvInteger" {
                Some(INT32)
            } else {
                first.dtype
            };
            vec![TensorType {
                dtype,
                shape: conv(node, &ctx),
            }]
        }
        "ConvTranspose" => vec![TensorType {
            dtype: first.dtype,
            shape: conv_transpose(node, &ctx),
        }],
        "MaxPool" | "AveragePool" => vec![TensorType {
            dtype: first.dtype,
            shape: pool(node, &first.shape),
        }],
        "GlobalAveragePool" | "GlobalMaxPool" => vec![TensorType {
            dtype: first.dtype,
            shape: first.shape.map(|dims| {
                dims.iter()
                    .enumerate()
                    .map(|(i, d)| if i < 2 { d.clone() } else { Dim::Fixed(1) })
                    .collect()
            }),
        }],

        // ---- quantization
        "DynamicQuantizeLinear" => vec![
            TensorType {
                dtype: Some(UINT8),
                shape: first.shape,
            },
            TensorType::of(FLOAT, vec![]),
            TensorType::of(UINT8, vec![]),
        ],
        "QuantizeLinear" => vec![TensorType {
            dtype: ctx.input(node, 2).dtype.or(Some(UINT8)),
            shape: first.shape,
        }],
        "DequantizeLinear" => vec![TensorType {
            dtype: Some(FLOAT),
            shape: first.shape,
        }],

        // ---- selection
        "Gather" => vec![TensorType {
            dtype: first.dtype,
            shape: gather(
                &first.shape,
                &ctx.shape_of(node, 1),
                attr_i64(node, "axis", 0),
            ),
        }],
        "GatherElements" => vec![TensorType {
            dtype: first.dtype,
            shape: ctx.shape_of(node, 1),
        }],
        "ScatterND" | "ScatterElements" => vec![first],
        "TopK" => {
            let shape = ctx.constant_ints(node, 1).and_then(|k| {
                let dims = first.shape.clone()?;
                let a = axis(attr_i64(node, "axis", -1), dims.len())?;
                let mut out = dims;
                out[a] = Dim::Fixed(*k.first()?);
                Some(out)
            });
            vec![
                TensorType {
                    dtype: first.dtype,
                    shape: shape.clone(),
                },
                TensorType {
                    dtype: Some(INT64),
                    shape,
                },
            ]
        }
        "NonZero" => vec![TensorType::dtype_only(INT64)],
        "Split" => split(node, &ctx, &first),

        // ---- shape from a constant input
        "Slice" => vec![TensorType {
            dtype: first.dtype,
            shape: slice(node, &ctx, &first.shape),
        }],
        "Expand" => vec![TensorType {
            dtype: first.dtype,
            shape: broadcast(
                &first.shape,
                &ctx.constant_ints(node, 1)
                    .map(|dims| dims.into_iter().map(Dim::Fixed).collect()),
            ),
        }],
        "Tile" => vec![TensorType {
            dtype: first.dtype,
            shape: tile(&first.shape, ctx.constant_ints(node, 1).as_deref()),
        }],
        "Pad" => vec![TensorType {
            dtype: first.dtype,
            shape: pad(&first.shape, ctx.constant_ints(node, 1).as_deref()),
        }],
        "Resize" | "Upsample" => vec![TensorType {
            dtype: first.dtype,
            shape: resize(node, &ctx, &first.shape),
        }],

        // ---- unknown shape, but dtype is that of the input
        //
        // Listed explicitly, not inferred: propagating the dtype "by default"
        // would give a plausible but wrong value for ops that change it, and a
        // wrong dtype is worse than an unknown dtype.
        "DepthToSpace" | "SpaceToDepth" | "CumSum" | "Trilu" | "Compress" | "ReverseSequence"
        | "GridSample" | "RoiAlign" | "MaxUnpool" | "Einsum" | "Hardmax" => vec![TensorType {
            dtype: first.dtype,
            shape: None,
        }],
        "IsNaN" | "IsInf" => vec![TensorType {
            dtype: Some(BOOL),
            shape: first.shape,
        }],
        "NonMaxSuppression" => vec![TensorType::of(INT64, vec![Dim::Unknown, Dim::Fixed(3)])],

        _ => vec![TensorType::default(); node.outputs.len().max(1)],
    }
}

// ---------------------------------------------------------------- rules

/// ONNX multidirectional broadcasting on symbolic dimensions.
///
/// A dimension against `1` is the other one; two equal dimensions (fixed or the
/// same symbol) stay; anything else is unknown rather than an error — two
/// different symbols may well be equal at run time.
fn broadcast(a: &Option<Vec<Dim>>, b: &Option<Vec<Dim>>) -> Option<Vec<Dim>> {
    let (a, b) = (a.as_ref()?, b.as_ref()?);
    let rank = a.len().max(b.len());
    let pad = |dims: &[Dim], i: usize| -> Dim {
        let offset = rank - dims.len();
        if i < offset {
            Dim::Fixed(1)
        } else {
            dims[i - offset].clone()
        }
    };
    Some(
        (0..rank)
            .map(|i| match (pad(a, i), pad(b, i)) {
                (Dim::Fixed(1), other) | (other, Dim::Fixed(1)) => other,
                (x, y) if x == y => x,
                _ => Dim::Unknown,
            })
            .collect(),
    )
}

/// `Reshape`: `0` keeps the input dimension, `-1` is inferred from the total.
fn reshape(input: &Option<Vec<Dim>>, target: Option<&[i64]>) -> Option<Vec<Dim>> {
    let target = target?;
    let mut out: Vec<Dim> = Vec::with_capacity(target.len());
    for (i, &d) in target.iter().enumerate() {
        out.push(match d {
            0 => input
                .as_ref()
                .and_then(|s| s.get(i))
                .cloned()
                .unwrap_or(Dim::Unknown),
            -1 => Dim::Unknown,
            n if n >= 0 => Dim::Fixed(n),
            _ => Dim::Unknown,
        });
    }
    // the inferred dimension closes only if everything else is known
    if let Some(position) = target.iter().position(|&d| d == -1)
        && let Some(total) = input
            .as_ref()
            .and_then(|s| s.iter().map(Dim::fixed).collect::<Option<Vec<_>>>())
            .map(|dims| dims.iter().product::<i64>())
    {
        let rest: Option<i64> = out
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != position)
            .map(|(_, d)| d.fixed())
            .product();
        if let Some(rest) = rest.filter(|r| *r != 0) {
            out[position] = Dim::Fixed(total / rest);
        }
    }
    Some(out)
}

fn transpose(input: &Option<Vec<Dim>>, perm: Option<&[i64]>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let perm: Vec<usize> = match perm {
        Some(p) => p
            .iter()
            .map(|&a| axis(a, dims.len()))
            .collect::<Option<_>>()?,
        // ONNX default is axis reversal
        None => (0..dims.len()).rev().collect(),
    };
    (perm.len() == dims.len()).then(|| perm.iter().map(|&i| dims[i].clone()).collect())
}

fn unsqueeze(input: &Option<Vec<Dim>>, axes: Option<Vec<i64>>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let axes = axes?;
    let rank = dims.len() + axes.len();
    let mut positions: Vec<usize> = axes.iter().map(|&a| axis(a, rank)).collect::<Option<_>>()?;
    positions.sort_unstable();

    let mut out = dims.clone();
    for position in positions {
        if position > out.len() {
            return None;
        }
        out.insert(position, Dim::Fixed(1));
    }
    Some(out)
}

fn squeeze(input: &Option<Vec<Dim>>, axes: Option<Vec<i64>>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    match axes {
        Some(axes) => {
            let drop: Vec<usize> = axes
                .iter()
                .map(|&a| axis(a, dims.len()))
                .collect::<Option<_>>()?;
            Some(
                dims.iter()
                    .enumerate()
                    .filter(|(i, _)| !drop.contains(i))
                    .map(|(_, d)| d.clone())
                    .collect(),
            )
        }
        // without axes all 1s are removed: possible only if all are known
        None => dims.iter().all(|d| d.fixed().is_some()).then(|| {
            dims.iter()
                .filter(|d| d.fixed() != Some(1))
                .cloned()
                .collect()
        }),
    }
}

fn flatten(input: &Option<Vec<Dim>>, at: i64) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let at = if at < 0 { at + dims.len() as i64 } else { at };
    let at = (0..=dims.len() as i64)
        .contains(&at)
        .then_some(at as usize)?;
    let product = |slice: &[Dim]| -> Dim {
        slice
            .iter()
            .map(Dim::fixed)
            .collect::<Option<Vec<_>>>()
            .map(|v| Dim::Fixed(v.iter().product()))
            .unwrap_or(Dim::Unknown)
    };
    Some(vec![product(&dims[..at]), product(&dims[at..])])
}

fn concat(node: &NodeIr, ctx: &Ctx) -> Option<Vec<Dim>> {
    let shapes: Vec<Vec<Dim>> = (0..node.inputs.len())
        .map(|i| ctx.shape_of(node, i))
        .collect::<Option<_>>()?;
    let first = shapes.first()?.clone();
    let a = axis(attr_i64(node, "axis", 0), first.len())?;
    let mut out = first;
    // the concatenated axis is the sum, and closes only if all addends are
    // known; the other axes stay as in the first input
    let total: Option<i64> = shapes.iter().map(|s| s.get(a).and_then(Dim::fixed)).sum();
    out[a] = total.map(Dim::Fixed).unwrap_or(Dim::Unknown);
    Some(out)
}

fn reduce(input: &Option<Vec<Dim>>, axes: Option<&[i64]>, keepdims: bool) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let reduced: Vec<usize> = match axes {
        Some(axes) => axes
            .iter()
            .map(|&a| axis(a, dims.len()))
            .collect::<Option<_>>()?,
        None => (0..dims.len()).collect(),
    };
    Some(
        dims.iter()
            .enumerate()
            .filter_map(|(i, d)| match (reduced.contains(&i), keepdims) {
                (true, true) => Some(Dim::Fixed(1)),
                (true, false) => None,
                (false, _) => Some(d.clone()),
            })
            .collect(),
    )
}

/// `MatMul` with ONNX's rank-1 promotion and batch broadcasting.
fn matmul(a: &Option<Vec<Dim>>, b: &Option<Vec<Dim>>) -> Option<Vec<Dim>> {
    let (a, b) = (a.as_ref()?, b.as_ref()?);
    if a.is_empty() || b.is_empty() {
        return None;
    }
    // a vector is promoted to a matrix and the added dimension then disappears
    let (a2, drop_row) = if a.len() == 1 {
        (vec![Dim::Fixed(1), a[0].clone()], true)
    } else {
        (a.clone(), false)
    };
    let (b2, drop_col) = if b.len() == 1 {
        (vec![b[0].clone(), Dim::Fixed(1)], true)
    } else {
        (b.clone(), false)
    };

    let batch = broadcast(
        &Some(a2[..a2.len() - 2].to_vec()),
        &Some(b2[..b2.len() - 2].to_vec()),
    )?;
    let mut out = batch;
    if !drop_row {
        out.push(a2[a2.len() - 2].clone());
    }
    if !drop_col {
        out.push(b2[b2.len() - 1].clone());
    }
    Some(out)
}

fn gemm(node: &NodeIr, ctx: &Ctx) -> Option<Vec<Dim>> {
    let a = ctx.shape_of(node, 0)?;
    let b = ctx.shape_of(node, 1)?;
    if a.len() != 2 || b.len() != 2 {
        return None;
    }
    let m = if attr_i64(node, "transA", 0) != 0 {
        &a[1]
    } else {
        &a[0]
    };
    let n = if attr_i64(node, "transB", 0) != 0 {
        &b[0]
    } else {
        &b[1]
    };
    Some(vec![m.clone(), n.clone()])
}

fn gather(data: &Option<Vec<Dim>>, indices: &Option<Vec<Dim>>, at: i64) -> Option<Vec<Dim>> {
    let data = data.as_ref()?;
    let indices = indices.as_ref()?;
    let a = axis(at, data.len())?;
    let mut out = data[..a].to_vec();
    out.extend(indices.iter().cloned());
    out.extend(data[a + 1..].iter().cloned());
    Some(out)
}

/// Spatial output of `Conv`, including `auto_pad`.
fn conv(node: &NodeIr, ctx: &Ctx) -> Option<Vec<Dim>> {
    let x = ctx.shape_of(node, 0)?;
    let w = ctx.shape_of(node, 1)?;
    if x.len() < 3 || w.len() != x.len() {
        return None;
    }
    let spatial = x.len() - 2;
    let kernel: Vec<i64> = match attr_ints(node, "kernel_shape") {
        Some(k) => k,
        None => w[2..].iter().map(Dim::fixed).collect::<Option<_>>()?,
    };
    let mut out = vec![x[0].clone(), w[0].clone()];
    out.extend(spatial_dims(node, &x[2..], &kernel, spatial)?);
    Some(out)
}

/// `ConvTranspose`: the spatial size grows instead of shrinking, and the
/// channel count comes from `W[1] * group` — `W` is `[C_in, C_out/group, ...]`.
fn conv_transpose(node: &NodeIr, ctx: &Ctx) -> Option<Vec<Dim>> {
    let x = ctx.shape_of(node, 0)?;
    let w = ctx.shape_of(node, 1)?;
    if x.len() < 3 || w.len() != x.len() {
        return None;
    }
    let spatial = x.len() - 2;
    let kernel: Vec<i64> = match attr_ints(node, "kernel_shape") {
        Some(k) => k,
        None => w[2..].iter().map(Dim::fixed).collect::<Option<_>>()?,
    };
    let group = attr_i64(node, "group", 1);
    let strides = attr_ints(node, "strides").unwrap_or_else(|| vec![1; spatial]);
    let dilations = attr_ints(node, "dilations").unwrap_or_else(|| vec![1; spatial]);
    let pads = attr_ints(node, "pads").unwrap_or_else(|| vec![0; spatial * 2]);
    let outpad = attr_ints(node, "output_padding").unwrap_or_else(|| vec![0; spatial]);

    let c_out = Dim::fixed(&w[1]).map_or(Dim::Unknown, |c| Dim::Fixed(c * group));
    let mut out = vec![x[0].clone(), c_out];
    for i in 0..spatial {
        let Some(size) = x[2 + i].fixed() else {
            out.push(Dim::Unknown);
            continue;
        };
        let stride = *strides.get(i)?.max(&1);
        let dilation = *dilations.get(i)?.max(&1);
        let pad = pads.get(i).copied().unwrap_or(0) + pads.get(i + spatial).copied().unwrap_or(0);
        let dim = (size - 1) * stride - pad
            + dilation * (kernel.get(i)? - 1)
            + 1
            + outpad.get(i).copied().unwrap_or(0);
        out.push(if dim > 0 {
            Dim::Fixed(dim)
        } else {
            Dim::Unknown
        });
    }
    Some(out)
}

fn pool(node: &NodeIr, input: &Option<Vec<Dim>>) -> Option<Vec<Dim>> {
    let x = input.as_ref()?;
    if x.len() < 3 {
        return None;
    }
    let spatial = x.len() - 2;
    let kernel = attr_ints(node, "kernel_shape")?;
    let mut out = vec![x[0].clone(), x[1].clone()];
    out.extend(spatial_dims(node, &x[2..], &kernel, spatial)?);
    Some(out)
}

/// `div_ceil` on signed integers is still unstable, and the operands here are
/// non-negative by construction.
fn ceil_div(a: i64, b: i64) -> i64 {
    if b <= 0 { 0 } else { (a + b - 1) / b }
}

/// The sliding-window formula, shared by `Conv` and pooling.
///
/// `SAME_UPPER`/`SAME_LOWER` are the reason padding cannot simply be read from
/// the attribute: with those the output only depends on stride.
fn spatial_dims(node: &NodeIr, input: &[Dim], kernel: &[i64], spatial: usize) -> Option<Vec<Dim>> {
    let auto_pad = node
        .attrs
        .get("auto_pad")
        .and_then(AttrValue::as_str)
        .unwrap_or("NOTSET")
        .to_string();
    let strides = attr_ints(node, "strides").unwrap_or_else(|| vec![1; spatial]);
    let dilations = attr_ints(node, "dilations").unwrap_or_else(|| vec![1; spatial]);
    let pads = attr_ints(node, "pads").unwrap_or_else(|| vec![0; spatial * 2]);
    let ceil_mode = attr_i64(node, "ceil_mode", 0) != 0;

    (0..spatial)
        .map(|i| {
            let stride = *strides.get(i)?.max(&1);
            let Some(size) = input.get(i).and_then(Dim::fixed) else {
                return Some(Dim::Unknown);
            };
            if auto_pad.starts_with("SAME") {
                return Some(Dim::Fixed(ceil_div(size, stride)));
            }
            let dilation = *dilations.get(i)?.max(&1);
            let effective = (kernel.get(i)? - 1) * dilation + 1;
            let padding = if auto_pad == "VALID" {
                0
            } else {
                pads.get(i).copied().unwrap_or(0) + pads.get(i + spatial).copied().unwrap_or(0)
            };
            let numerator = size + padding - effective;
            if numerator < 0 {
                return Some(Dim::Unknown);
            }
            Some(Dim::Fixed(if ceil_mode {
                ceil_div(numerator, stride) + 1
            } else {
                numerator / stride + 1
            }))
        })
        .collect()
}

/// `Slice` with load-time bounds. Clamping follows ONNX: the ends saturate to
/// the dimension, and `INT64::MAX` (the usual "to the end") is one of them.
fn slice(node: &NodeIr, ctx: &Ctx, input: &Option<Vec<Dim>>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let starts = ctx.constant_ints(node, 1)?;
    let ends = ctx.constant_ints(node, 2)?;
    let axes = match ctx.constant_ints(node, 3) {
        Some(axes) => axes,
        None => (0..starts.len() as i64).collect(),
    };
    let steps = ctx
        .constant_ints(node, 4)
        .unwrap_or_else(|| vec![1; starts.len()]);

    let mut out = dims.clone();
    for (i, &raw_axis) in axes.iter().enumerate() {
        let a = axis(raw_axis, dims.len())?;
        let Some(size) = dims[a].fixed() else {
            out[a] = Dim::Unknown;
            continue;
        };
        let step = *steps.get(i).unwrap_or(&1);
        if step <= 0 {
            // negative steps reverse, and the count changes sign:
            // outside the little we need here
            out[a] = Dim::Unknown;
            continue;
        }
        let clamp = |v: i64| (if v < 0 { v + size } else { v }).clamp(0, size);
        let (begin, end) = (clamp(*starts.get(i)?), clamp(*ends.get(i)?));
        out[a] = Dim::Fixed(ceil_div((end - begin).max(0), step));
    }
    Some(out)
}

fn tile(input: &Option<Vec<Dim>>, repeats: Option<&[i64]>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let repeats = repeats?;
    (repeats.len() == dims.len()).then(|| {
        dims.iter()
            .zip(repeats)
            .map(|(d, r)| d.fixed().map(|n| Dim::Fixed(n * r)).unwrap_or(Dim::Unknown))
            .collect()
    })
}

/// `Pad`: the input is `[begin…, end…]`, one pair per axis.
fn pad(input: &Option<Vec<Dim>>, pads: Option<&[i64]>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    let pads = pads?;
    (pads.len() == dims.len() * 2).then(|| {
        dims.iter()
            .enumerate()
            .map(|(i, d)| {
                d.fixed()
                    .map(|n| Dim::Fixed(n + pads[i] + pads[i + dims.len()]))
                    .unwrap_or(Dim::Unknown)
            })
            .collect()
    })
}

/// `Resize` takes either `sizes` (absolute) or `scales` (relative); ONNX gives
/// exactly one of them, and `sizes` wins when both are present.
fn resize(node: &NodeIr, ctx: &Ctx, input: &Option<Vec<Dim>>) -> Option<Vec<Dim>> {
    let dims = input.as_ref()?;
    // `sizes` is input 3 in Resize and does not exist in Upsample
    if let Some(sizes) = ctx.constant_ints(node, 3).filter(|s| s.len() == dims.len()) {
        return Some(sizes.into_iter().map(Dim::Fixed).collect());
    }
    let scales_input = if node.op == "Upsample" { 1 } else { 2 };
    let name = node.inputs.get(scales_input).filter(|n| !n.is_empty())?;
    let tensor = ctx.constants.get(name.as_str())?;
    let scales =
        onnx_vulkan_core::HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone())
            .to_f32()
            .ok()?;
    (scales.len() == dims.len()).then(|| {
        dims.iter()
            .zip(&scales)
            .map(|(d, s)| {
                d.fixed()
                    .map(|n| Dim::Fixed((n as f32 * s).floor() as i64))
                    .unwrap_or(Dim::Unknown)
            })
            .collect()
    })
}

fn split(node: &NodeIr, ctx: &Ctx, first: &TensorType) -> Vec<TensorType> {
    let count = node.outputs.len().max(1);
    let sizes = attr_ints(node, "split").or_else(|| ctx.constant_ints(node, 1));
    let shapes: Option<Vec<Vec<Dim>>> = (|| {
        let dims = first.shape.clone()?;
        let a = axis(attr_i64(node, "axis", 0), dims.len())?;
        let sizes = match sizes {
            Some(sizes) => sizes,
            // without `split` the division is into equal parts, computable only
            // if the axis is known
            None => {
                let total = dims[a].fixed()?;
                vec![total / count as i64; count]
            }
        };
        Some(
            sizes
                .iter()
                .map(|&size| {
                    let mut out = dims.clone();
                    out[a] = Dim::Fixed(size);
                    out
                })
                .collect(),
        )
    })();

    match shapes {
        Some(shapes) => shapes
            .into_iter()
            .map(|shape| TensorType {
                dtype: first.dtype,
                shape: Some(shape),
            })
            .collect(),
        None => vec![
            TensorType {
                dtype: first.dtype,
                shape: None
            };
            count
        ],
    }
}
