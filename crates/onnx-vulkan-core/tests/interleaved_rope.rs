//! The interleaved rotary embedding SAM 3 writes out node by node, fused into
//! `InterleavedRope` at load time, must answer **bit for bit** like the
//! nodes it replaces — with the Transpose before it, the Transpose after it
//! and the attention scale folded in or not.
//!
//! The reference is the same graph with the `Neg` output declared as a graph
//! output: an intermediate that escapes cannot be fused, so the rewrite leaves
//! that graph alone and it runs the original kernels on the same device.

use onnx_vulkan_core::host_ops::{FLOAT, INT64};
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 17,
        name: format!("{op}_{}", outputs[0]),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_string()).collect(),
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect(),
    }
}

fn int64(shape: Vec<i64>, values: &[i64]) -> InitializerIr {
    InitializerIr {
        dtype: INT64,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn float(shape: Vec<i64>, values: &[f32]) -> InitializerIr {
    InitializerIr {
        dtype: FLOAT,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// Which neighbours of the rotation the graph carries.
#[derive(Clone, Copy, Debug)]
struct Layout {
    /// `x` is `[B, S, H, D]` and a `Transpose` makes it `[B, H, S, D]`
    bshd: bool,
    /// a `Transpose(·, [0, 1, 3, 2])` follows, as for K
    transpose: bool,
    /// a `Mul` by a scalar constant follows, as SAM 3 scales Q and K
    scale: bool,
}

/// How the exporter wrote the rotation (see `rewrite::traced_rotation` and
/// `rewrite::exported_rotation`).
#[derive(Clone, Copy, Debug)]
enum Form {
    /// TorchScript, as in SAM 3's int8 vision encoder: shapes computed from
    /// `Shape`, the halves taken by a `Split`
    Traced,
    /// `torch.onnx.export(dynamo=True)`, as in SAM 3's fp32 vision encoder:
    /// constant shapes with `allowzero = 1`, two `Slice`s, and K's transpose
    /// written on three dimensions
    Exported,
}

const FORMS: [Form; 2] = [Form::Traced, Form::Exported];

/// SAM 3's rotation as its exporter wrote it (see `rewrite::match_interleaved_rope`).
/// `dims` is `[B, H, S, D]`, which the exported form writes into its shapes.
fn rope_graph(
    layout: Layout,
    form: Form,
    dims: [i64; 4],
    cos: InitializerIr,
    sin: InitializerIr,
) -> GraphIr {
    let [b, h, s, d] = dims;
    let allowzero = || ("allowzero", AttrValue::Int(1));
    let axis = |a: i64| ("axis", AttrValue::Int(a));
    let mut nodes = Vec::new();
    if layout.bshd {
        let perm = ("perm", AttrValue::Ints(vec![0, 2, 1, 3]));
        nodes.push(node("Transpose", &["x"], &["t"], &[perm]));
    }
    let t = if layout.bshd { "t" } else { "x" };
    let last = if layout.transpose || layout.scale {
        "sum"
    } else {
        "y"
    };
    nodes.push(node("Mul", &[t, "cos"], &["direct"], &[]));
    match form {
        Form::Traced => nodes.extend([
            node("Shape", &[t], &["shp"], &[]),
            node("Gather", &["shp", "i0"], &["g0"], &[axis(0)]),
            node("Gather", &["shp", "i1"], &["g1"], &[axis(0)]),
            node("Gather", &["shp", "i2"], &["g2"], &[axis(0)]),
            node("Unsqueeze", &["g0", "zero"], &["d0"], &[]),
            node("Unsqueeze", &["g1", "zero"], &["d1"], &[]),
            node("Unsqueeze", &["g2", "zero"], &["d2"], &[]),
            node(
                "Concat",
                &["d0", "d1", "d2", "minus1", "two"],
                &["pairs_shape"],
                &[axis(0)],
            ),
            node("Reshape", &[t, "pairs_shape"], &["pairs"], &[]),
            node("Split", &["pairs", "halves"], &["x0", "x1"], &[axis(-1)]),
        ]),
        Form::Exported => nodes.extend([
            node("Reshape", &[t, "pairs_dims"], &["pairs"], &[allowzero()]),
            node("Slice", &["pairs", "zero", "one", "minus1"], &["x0"], &[]),
            node("Slice", &["pairs", "one", "two", "minus1"], &["x1"], &[]),
        ]),
    }
    nodes.extend([
        node("Squeeze", &["x0", "minus1"], &["even"], &[]),
        node("Squeeze", &["x1", "minus1"], &["odd"], &[]),
        node("Neg", &["odd"], &["neg"], &[]),
        node("Unsqueeze", &["neg", "minus1"], &["negu"], &[]),
        node("Unsqueeze", &["even", "minus1"], &["evenu"], &[]),
        node("Concat", &["negu", "evenu"], &["swapped"], &[axis(-1)]),
    ]);
    match form {
        Form::Traced => nodes.extend([
            node("Shape", &["swapped"], &["sshp"], &[]),
            node("Slice", &["sshp", "zero", "minus2", "zero"], &["lead"], &[]),
            node("Concat", &["lead", "minus1"], &["rot_shape"], &[axis(0)]),
            node("Reshape", &["swapped", "rot_shape"], &["rot"], &[]),
        ]),
        Form::Exported => nodes.push(node(
            "Reshape",
            &["swapped", "rot_dims"],
            &["rot"],
            &[allowzero()],
        )),
    }
    nodes.extend([
        node("Mul", &["rot", "sin"], &["rotated"], &[]),
        node("Add", &["direct", "rotated"], &[last], &[]),
    ]);
    if layout.transpose {
        let to = if layout.scale { "kt" } else { "y" };
        match form {
            Form::Traced => {
                let perm = ("perm", AttrValue::Ints(vec![0, 1, 3, 2]));
                nodes.push(node("Transpose", &["sum"], &[to], &[perm]));
            }
            Form::Exported => {
                let perm = ("perm", AttrValue::Ints(vec![0, 2, 1]));
                nodes.extend([
                    node("Reshape", &["sum", "k3_dims"], &["k3"], &[]),
                    node("Transpose", &["k3"], &["k3t"], &[perm]),
                    node("Reshape", &["k3t", "k4_dims"], &[to], &[]),
                ]);
            }
        }
    }
    if layout.scale {
        let from = if layout.transpose { "kt" } else { "sum" };
        nodes.push(node("Mul", &[from, "scale"], &["y"], &[]));
    }
    GraphIr {
        nodes,
        initializers: HashMap::from([
            ("i0".into(), int64(vec![], &[0])),
            ("i1".into(), int64(vec![], &[1])),
            ("i2".into(), int64(vec![], &[2])),
            ("zero".into(), int64(vec![1], &[0])),
            ("one".into(), int64(vec![1], &[1])),
            ("minus1".into(), int64(vec![1], &[-1])),
            ("minus2".into(), int64(vec![1], &[-2])),
            ("two".into(), int64(vec![1], &[2])),
            ("halves".into(), int64(vec![2], &[1, 1])),
            ("pairs_dims".into(), int64(vec![5], &[b, h, s, -1, 2])),
            ("rot_dims".into(), int64(vec![4], &[b, h, s, d])),
            ("k3_dims".into(), int64(vec![3], &[-1, s, d])),
            ("k4_dims".into(), int64(vec![4], &[b, h, d, s])),
            // SAM 3's 64^-1/4, which it applies to both Q and K
            ("scale".into(), float(vec![], &[0.353_553_38])),
            ("cos".into(), cos),
            ("sin".into(), sin),
        ]),
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        ..Default::default()
    }
}

struct Run {
    bits: Vec<u32>,
    shape: Vec<i64>,
    ops: Vec<String>,
    fused_kernel: bool,
}

fn run(context: &VkContext, graph: GraphIr, shape: &[i64], x: &[f32]) -> Run {
    let executor = Executor::new(context, graph).expect("executor");
    let ops = executor
        .graph()
        .nodes
        .iter()
        .map(|n| n.op.clone())
        .collect();
    let feed = HostTensor::from_f32(shape.to_vec(), x);
    let outputs = executor.run(vec![("x", Tensor::Host(feed))]).expect("run");
    let y = outputs.host("y").expect("output");
    let shape = y.shape.clone();
    let bits = y.to_f32().unwrap().iter().map(|v| v.to_bits()).collect();
    outputs.finish();
    let fused_kernel = executor
        .cache()
        .pipeline_keys()
        .iter()
        .any(|key| key.operation() == "InterleavedRope");
    Run {
        bits,
        shape,
        ops,
        fused_kernel,
    }
}

/// Deterministic 32-bit patterns (a 64-bit LCG's high half).
fn patterns(seed: u64, n: usize) -> impl Iterator<Item = u32> {
    let mut state = seed;
    (0..n).map(move |_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    })
}

/// Half arbitrary bit patterns (both signs, subnormals, ±inf, NaNs), half
/// activations of a realistic magnitude.
fn activations(seed: u64, n: usize) -> Vec<f32> {
    patterns(seed, n)
        .enumerate()
        .map(|(i, bits)| {
            if i % 2 == 0 {
                f32::from_bits(bits)
            } else {
                (bits as f32 / u32::MAX as f32 * 2.0 - 1.0) * 6.0
            }
        })
        .collect()
}

/// Fused against unfused, bit for bit; `kernel` says whether the fused shader
/// must have run or the step-by-step replay.
fn check(
    layout: Layout,
    form: Form,
    dims: [i64; 4],
    cos: InitializerIr,
    sin: InitializerIr,
    kernel: bool,
) {
    let [b, h, s, d] = dims;
    let shape = if layout.bshd {
        [b, s, h, d]
    } else {
        [b, h, s, d]
    };
    let x = activations(
        dims.iter().product::<i64>() as u64,
        (b * h * s * d) as usize,
    );
    let context = VkContext::new().expect("Vulkan context");

    let mut reference = rope_graph(layout, form, dims, cos.clone(), sin.clone());
    reference.outputs.push("neg".into());
    let expected = run(&context, reference, &shape, &x);
    assert!(expected.ops.iter().any(|op| op == "Neg"));
    assert!(!expected.fused_kernel);

    let got = run(
        &context,
        rope_graph(layout, form, dims, cos, sin),
        &shape,
        &x,
    );
    assert_eq!(got.ops, ["InterleavedRope"], "{layout:?} {form:?}");
    assert_eq!(got.fused_kernel, kernel, "{layout:?} {form:?}");
    assert_eq!(got.shape, expected.shape, "{layout:?} {form:?}");
    let differing: Vec<(usize, u32, u32)> = got
        .bits
        .iter()
        .zip(&expected.bits)
        .enumerate()
        .filter(|(_, (g, e))| g != e)
        .map(|(i, (g, e))| (i, *g, *e))
        .collect();
    assert!(
        differing.is_empty(),
        "{layout:?} {form:?}: {} of {} elements differ, first (index, got, expected): {:?}",
        differing.len(),
        got.bits.len(),
        &differing[..differing.len().min(5)]
    );
}

/// Tables of `cos`/`sin`-like values of the given shape.
fn table(seed: u64, shape: Vec<i64>) -> InitializerIr {
    let n = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = patterns(seed, n)
        .map(|bits| (bits as f32 / u32::MAX as f32 * 2.0 - 1.0).sin())
        .collect();
    float(shape, &values)
}

/// Tables of arbitrary bit patterns: the rounding of every product and sum
/// is exercised, not only those of numbers below one.
fn wild_table(seed: u64, shape: Vec<i64>) -> InitializerIr {
    let n = shape.iter().product::<i64>() as usize;
    let values: Vec<f32> = patterns(seed, n).map(f32::from_bits).collect();
    float(shape, &values)
}

const ALL_LAYOUTS: [Layout; 4] = [
    // SAM 3's Q
    Layout {
        bshd: true,
        transpose: false,
        scale: true,
    },
    // SAM 3's K
    Layout {
        bshd: true,
        transpose: true,
        scale: true,
    },
    Layout {
        bshd: false,
        transpose: false,
        scale: false,
    },
    Layout {
        bshd: false,
        transpose: true,
        scale: false,
    },
];

#[test]
fn fused_rope_matches_the_original_nodes() {
    // SAM 3's window-attention shape, 590k elements per layout
    let dims = [1, 16, 576, 64];
    for layout in ALL_LAYOUTS {
        for form in FORMS {
            check(
                layout,
                form,
                dims,
                table(1, vec![576, 64]),
                table(2, vec![576, 64]),
                true,
            );
        }
    }
}

#[test]
fn fused_rope_matches_on_arbitrary_tables() {
    // odd sizes, two batches, tables of rank 4 as [1, 1, S, D]; the second
    // splits D over three tiles of the transposed store, the last partial
    for dims in [[2, 3, 37, 10], [1, 2, 45, 66]] {
        let [_, _, s, d] = dims;
        for layout in ALL_LAYOUTS {
            for form in FORMS {
                let cos = wild_table(3, vec![1, 1, s, d]);
                let sin = wild_table(4, vec![1, 1, s, d]);
                check(layout, form, dims, cos, sin, true);
            }
        }
    }
}

/// Per-head tables broadcast as `[H, S, D]`, not `[S, D]`: the shader does not
/// cover them, so the original steps are replayed.
#[test]
fn per_head_tables_replay_the_original_steps() {
    let dims = [1, 4, 32, 16];
    for layout in ALL_LAYOUTS {
        for form in FORMS {
            let cos = table(5, vec![4, 32, 16]);
            let sin = table(6, vec![4, 32, 16]);
            check(layout, form, dims, cos, sin, false);
        }
    }
}

/// A few host values: the original `Mul` runs on host, so the shader must not.
#[test]
fn small_host_input_replays_the_original_steps() {
    let dims = [1, 2, 3, 4];
    for layout in ALL_LAYOUTS {
        for form in FORMS {
            check(
                layout,
                form,
                dims,
                table(7, vec![3, 4]),
                table(8, vec![3, 4]),
                false,
            );
        }
    }
}
