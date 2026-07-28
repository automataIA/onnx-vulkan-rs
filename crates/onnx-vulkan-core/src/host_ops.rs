//! **Host-side** execution of shape/control ops in fused subgraphs.
//!
//! Small tensors (shape, indices, masks — typically int64/int32/bool)
//! can be computed on CPU without introducing GPU synchronization.
//!
//! `HostTensor` is a dtype-generic CPU tensor (raw bytes). Functions here
//! implement ONNX semantics for the ops subset handled by the interpreter.
//! runs host-side.

use crate::{ElementType, Error, Result, broadcast, elem_size, element_count, storage_len};

macro_rules! bail {
    ($($argument:tt)*) => {
        return Err(Error::InvalidTensor(format!($($argument)*)))
    };
}

macro_rules! ensure {
    ($condition:expr, $($argument:tt)*) => {
        if !$condition {
            bail!($($argument)*);
        }
    };
}

pub const FLOAT: i32 = ElementType::Float32 as i32;
pub const UINT8: i32 = ElementType::Uint8 as i32;
pub const INT8: i32 = ElementType::Int8 as i32;
pub const INT32: i32 = ElementType::Int32 as i32;
pub const INT64: i32 = ElementType::Int64 as i32;
pub const BOOL: i32 = ElementType::Bool as i32;

/// Dtype-generic host tensor (raw bytes in row-major layout).
#[derive(Clone, Debug, PartialEq)]
pub struct HostTensor {
    pub dtype: i32,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}

impl HostTensor {
    pub fn elem_count(&self) -> usize {
        element_count(&self.shape).unwrap_or(0)
    }

    pub fn new(dtype: i32, shape: Vec<i64>, data: Vec<u8>) -> Self {
        HostTensor { dtype, shape, data }
    }

    /// Verifies shape, dtype, and storage size are consistent.
    pub fn validate(&self) -> Result<()> {
        let count = element_count(&self.shape)?;
        let expected = storage_len(self.dtype, count).ok_or_else(|| {
            Error::InvalidTensor(format!("dtype {} senza storage fisso", self.dtype))
        })?;
        ensure!(
            self.data.len() == expected,
            "storage di {} byte, attesi {expected} per dtype {} e shape {:?}",
            self.data.len(),
            self.dtype,
            self.shape
        );
        Ok(())
    }

    pub fn from_i64(shape: Vec<i64>, v: &[i64]) -> Self {
        let mut data = Vec::with_capacity(v.len() * 8);
        for x in v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        HostTensor::new(INT64, shape, data)
    }

    pub fn from_f32(shape: Vec<i64>, v: &[f32]) -> Self {
        let mut data = Vec::with_capacity(v.len() * 4);
        for x in v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        HostTensor::new(FLOAT, shape, data)
    }

    /// Values as i64 (promotes int32/bool; f32 truncated). For shape-math.
    pub fn to_i64(&self) -> Result<Vec<i64>> {
        self.validate()?;
        let n = self.elem_count();
        Ok(match self.dtype {
            INT64 => self
                .data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            INT32 => self
                .data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
            BOOL => self.data.iter().take(n).map(|&b| (b != 0) as i64).collect(),
            UINT8 => self.data.iter().take(n).map(|&b| b as i64).collect(),
            INT8 => self.data.iter().take(n).map(|&b| b as i8 as i64).collect(),
            FLOAT => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
            d => bail!("host to_i64: dtype {d} not supported"),
        })
    }

    /// Values as f32 (promotes int).
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        self.validate()?;
        Ok(match self.dtype {
            FLOAT => self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            INT64 | INT32 | BOOL | UINT8 | INT8 => {
                self.to_i64()?.into_iter().map(|x| x as f32).collect()
            }
            d => bail!("host to_f32: dtype {d} not supported"),
        })
    }
}

/// `Cast`: converts dtype (attribute `to`). Common subset.
pub fn cast(x: &HostTensor, to: i32) -> Result<HostTensor> {
    x.validate()?;
    if x.dtype == to {
        return Ok(x.clone());
    }
    let shape = x.shape.clone();
    Ok(match to {
        FLOAT => HostTensor::from_f32(shape, &x.to_f32()?),
        INT64 => HostTensor::from_i64(shape, &x.to_i64()?),
        INT32 => {
            let v = x.to_i64()?;
            let mut data = Vec::with_capacity(v.len() * 4);
            for e in v {
                data.extend_from_slice(&(e as i32).to_le_bytes());
            }
            HostTensor::new(INT32, shape, data)
        }
        BOOL => {
            let v = x.to_i64()?;
            HostTensor::new(BOOL, shape, v.iter().map(|&e| (e != 0) as u8).collect())
        }
        d => bail!("host Cast: target dtype {d} not supported"),
    })
}

/// Host-side binary arithmetic op with ONNX broadcasting.
/// Sub/Div are used by the interpreter's upcoming ops (Sub, Div).
#[derive(Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Min,
    Max,
}

/// Single-axis reduction, host-side.
#[derive(Clone, Copy, PartialEq)]
pub enum RedOp {
    Sum,
    Mean,
    Max,
    Min,
}

/// Single-axis reduction for **non-f32** tensors not handled by GPU kernels.
///
/// Necessary because support checks inspect node types rather than dtypes:
/// `ReduceSum` on an int64 mask is claimed like all the others, and must have
/// a working path. Math stays in i64 and the result returns to the input
/// dtype; the mean truncates, as a cast from float would.
pub fn reduce(x: &HostTensor, axis: usize, op: RedOp, keepdims: bool) -> Result<HostTensor> {
    let rank = x.shape.len();
    if axis >= rank {
        return Err(Error::InvalidTensor(format!(
            "reduce: axis {axis} out of rank {rank}"
        )));
    }
    let values = x.to_i64()?;
    let c = x.shape[axis].max(0) as usize;
    let inner: usize = x.shape[axis + 1..].iter().product::<i64>().max(1) as usize;
    let rows = values.len().checked_div(c.max(1)).unwrap_or(0);

    let mut out = Vec::with_capacity(rows * 8);
    for r in 0..rows {
        let base = (r / inner) * c * inner + r % inner;
        let it = (0..c).map(|k| values[base + k * inner]);
        let acc = match op {
            RedOp::Max => it.fold(i64::MIN, i64::max),
            RedOp::Min => it.fold(i64::MAX, i64::min),
            RedOp::Sum => it.sum(),
            RedOp::Mean => it.sum::<i64>() / c.max(1) as i64,
        };
        out.extend_from_slice(&acc.to_le_bytes());
    }

    let mut shape = x.shape.clone();
    if keepdims {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }
    cast(&HostTensor::new(INT64, shape, out), x.dtype)
}

/// `CumSum` along one axis, with `exclusive` and `reverse`.
///
/// Host-side whatever the dtype: a prefix sum is sequential along the axis, and
/// the only occurrence in the tested models is an integer mask of a few
/// hundred elements. A GPU scan is worth writing when one shows up in a hot
/// path, not before.
pub fn cumsum(x: &HostTensor, axis: i64, exclusive: bool, reverse: bool) -> Result<HostTensor> {
    x.validate()?;
    let rank = x.shape.len() as i64;
    let a = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&a),
        "CumSum: axis {axis} out of range (rank {rank})"
    );
    let a = a as usize;
    let c = x.shape[a].max(0) as usize;
    let inner: usize = x.shape[a + 1..].iter().product::<i64>().max(1) as usize;

    let float = x.dtype == FLOAT;
    let values: Vec<f64> = if float {
        x.to_f32()?.into_iter().map(f64::from).collect()
    } else {
        x.to_i64()?.into_iter().map(|v| v as f64).collect()
    };
    let mut out = vec![0f64; values.len()];
    let rows = values.len().checked_div(c.max(1)).unwrap_or(0);
    for r in 0..rows {
        let base = (r / inner) * c * inner + r % inner;
        let mut acc = 0f64;
        for step in 0..c {
            // `reverse` walks the axis from the end; `exclusive` writes the sum
            // *before* including the current element
            let k = if reverse { c - 1 - step } else { step };
            let index = base + k * inner;
            if exclusive {
                out[index] = acc;
                acc += values[index];
            } else {
                acc += values[index];
                out[index] = acc;
            }
        }
    }

    let shape = x.shape.clone();
    if float {
        Ok(HostTensor::from_f32(
            shape,
            &out.into_iter().map(|v| v as f32).collect::<Vec<_>>(),
        ))
    } else {
        cast(
            &HostTensor::from_i64(
                shape,
                &out.into_iter().map(|v| v as i64).collect::<Vec<_>>(),
            ),
            x.dtype,
        )
    }
}

/// Elementwise binary op with broadcasting. int64 if both are integers, otherwise
/// f32 (sufficient semantics for shape-math and the few host-side f32 cases).
pub fn binary(a: &HostTensor, b: &HostTensor, op: BinOp) -> Result<HostTensor> {
    let bc = broadcast(&a.shape, &b.shape)?;
    let n: usize = bc.out_shape.iter().product::<i64>().max(0) as usize;
    // out_strides are cumulative products (always ≥1): they decompose the linear
    // output index; a_strides/b_strides are 0 on broadcast dimensions.
    let offset = |strides: &[u32], i: usize| -> usize {
        let mut rem = i;
        let mut off = 0usize;
        for (os, st) in bc.out_strides.iter().zip(strides.iter()) {
            let os = (*os as usize).max(1);
            off += (rem / os) * (*st as usize);
            rem %= os;
        }
        off
    };
    let int_domain = is_int(a.dtype) && is_int(b.dtype);
    if int_domain {
        let (av, bv) = (a.to_i64()?, b.to_i64()?);
        let mut out = vec![0i64; n];
        for (i, o) in out.iter_mut().enumerate() {
            let (x, y) = (av[offset(&bc.a_strides, i)], bv[offset(&bc.b_strides, i)]);
            *o = match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => {
                    ensure!(y != 0, "host Div: divisione intera per zero");
                    x / y
                }
                BinOp::Pow => {
                    ensure!(
                        (0..=u32::MAX as i64).contains(&y),
                        "host Pow: exponent {y} out of range"
                    );
                    x.pow(y as u32)
                }
                BinOp::Min => x.min(y),
                BinOp::Max => x.max(y),
            };
        }
        Ok(HostTensor::from_i64(bc.out_shape, &out))
    } else {
        let (av, bv) = (a.to_f32()?, b.to_f32()?);
        let mut out = vec![0f32; n];
        for (i, o) in out.iter_mut().enumerate() {
            let (x, y) = (av[offset(&bc.a_strides, i)], bv[offset(&bc.b_strides, i)]);
            *o = match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::Pow => x.powf(y),
                BinOp::Min => x.min(y),
                BinOp::Max => x.max(y),
            };
        }
        Ok(HostTensor::from_f32(bc.out_shape, &out))
    }
}

fn is_int(dtype: i32) -> bool {
    matches!(dtype, INT64 | INT32 | BOOL)
}

/// Linear offset into an operand given the output index and its broadcast strides.
fn bc_offset(out_strides: &[u32], strides: &[u32], i: usize) -> usize {
    let mut rem = i;
    let mut off = 0usize;
    for (os, st) in out_strides.iter().zip(strides.iter()) {
        let os = (*os as usize).max(1);
        off += (rem / os) * (*st as usize);
        rem %= os;
    }
    off
}

/// Elementwise comparison/logic → bool, with ONNX broadcasting (Equal/Less/And).
#[derive(Clone, Copy)]
pub enum CmpOp {
    Equal,
    Less,
    Greater,
    And,
}

pub fn compare(a: &HostTensor, b: &HostTensor, op: CmpOp) -> Result<HostTensor> {
    let bc = broadcast(&a.shape, &b.shape)?;
    let n: usize = bc.out_shape.iter().product::<i64>().max(0) as usize;
    let mut out = vec![0u8; n];
    let use_int = is_int(a.dtype) && is_int(b.dtype);
    if use_int {
        let (av, bv) = (a.to_i64()?, b.to_i64()?);
        for (i, o) in out.iter_mut().enumerate() {
            let x = av[bc_offset(&bc.out_strides, &bc.a_strides, i)];
            let y = bv[bc_offset(&bc.out_strides, &bc.b_strides, i)];
            *o = match op {
                CmpOp::Equal => x == y,
                CmpOp::Less => x < y,
                CmpOp::Greater => x > y,
                CmpOp::And => x != 0 && y != 0,
            } as u8;
        }
    } else {
        let (av, bv) = (a.to_f32()?, b.to_f32()?);
        for (i, o) in out.iter_mut().enumerate() {
            let x = av[bc_offset(&bc.out_strides, &bc.a_strides, i)];
            let y = bv[bc_offset(&bc.out_strides, &bc.b_strides, i)];
            *o = match op {
                CmpOp::Equal => x == y,
                CmpOp::Less => x < y,
                CmpOp::Greater => x > y,
                CmpOp::And => x != 0.0 && y != 0.0,
            } as u8;
        }
    }
    Ok(HostTensor::new(BOOL, bc.out_shape, out))
}

/// `Mod` with ONNX broadcasting.
///
/// Two different remainders, selected by the `fmod` attribute, and they are not
/// the same function on mixed signs:
///
/// - `fmod = 0` — the Python/NumPy remainder, whose sign follows the
///   **divisor**: `-7 mod 5 = 3`. ONNX restricts it to integer types.
/// - `fmod = 1` — the C `fmod`, whose sign follows the **dividend**:
///   `-7 fmod 5 = -2`. This is the only form allowed on floats.
///
/// On host because every `Mod` seen so far computes shape math on scalars: the
/// cost that matters is the block split, not the arithmetic.
pub fn modulo(a: &HostTensor, b: &HostTensor, fmod: bool) -> Result<HostTensor> {
    let bc = broadcast(&a.shape, &b.shape)?;
    let n: usize = bc.out_shape.iter().product::<i64>().max(0) as usize;
    if is_int(a.dtype) && is_int(b.dtype) {
        let (av, bv) = (a.to_i64()?, b.to_i64()?);
        let mut out = vec![0i64; n];
        for (i, o) in out.iter_mut().enumerate() {
            let x = av[bc_offset(&bc.out_strides, &bc.a_strides, i)];
            let y = bv[bc_offset(&bc.out_strides, &bc.b_strides, i)];
            ensure!(y != 0, "Mod: divisione per zero");
            let r = x % y; // Rust `%` is the C remainder: sign of the dividend
            *o = if !fmod && r != 0 && (r < 0) != (y < 0) {
                r + y
            } else {
                r
            };
        }
        return cast(&HostTensor::from_i64(bc.out_shape, &out), a.dtype);
    }
    ensure!(fmod, "Mod: fmod = 0 is not defined on floats (ONNX spec)");
    let (av, bv) = (a.to_f32()?, b.to_f32()?);
    let mut out = vec![0f32; n];
    for (i, o) in out.iter_mut().enumerate() {
        let x = av[bc_offset(&bc.out_strides, &bc.a_strides, i)];
        let y = bv[bc_offset(&bc.out_strides, &bc.b_strides, i)];
        *o = x % y;
    }
    Ok(HostTensor::from_f32(bc.out_shape, &out))
}

/// `Floor` (no-op on integers).
pub fn floor(x: &HostTensor) -> Result<HostTensor> {
    x.validate()?;
    if is_int(x.dtype) {
        return Ok(x.clone());
    }
    let v: Vec<f32> = x.to_f32()?.iter().map(|f| f.floor()).collect();
    Ok(HostTensor::from_f32(x.shape.clone(), &v))
}

/// Logical `Not` → bool.
pub fn not(x: &HostTensor) -> Result<HostTensor> {
    let v = x.to_i64()?;
    Ok(HostTensor::new(
        BOOL,
        x.shape.clone(),
        v.iter().map(|&b| (b == 0) as u8).collect(),
    ))
}

/// `ConstantOfShape`: tensor of `shape` filled with `value` (bytes, length
/// elem_size of the dtype); default f32 0.
pub fn const_of_shape(shape: Vec<i64>, dtype: i32, value: &[u8]) -> Result<HostTensor> {
    let es = elem_size(dtype);
    ensure!(es > 0, "ConstantOfShape: dtype {dtype} with no size");
    let n = element_count(&shape)?;
    let mut data = vec![0u8; n * es];
    if value.len() == es {
        for chunk in data.chunks_mut(es) {
            chunk.copy_from_slice(value);
        }
    }
    Ok(HostTensor::new(dtype, shape, data))
}

/// `Expand`: broadcasts `x` toward `target` (ONNX rules; target may add
/// dimensions).
pub fn expand(x: &HostTensor, target: &[i64]) -> Result<HostTensor> {
    x.validate()?;
    let bc = broadcast(&x.shape, target)?;
    let out_shape = bc.out_shape;
    let es = elem_size(x.dtype);
    ensure!(es > 0, "Expand: dtype {} with no size", x.dtype);
    let rank = out_shape.len();
    let xs = bcast_strides(&x.shape, &out_shape);
    let n = element_count(&out_shape)?;
    let mut out = vec![0u8; n * es];
    for o in 0..n {
        let mut acc = n;
        let mut xi = 0i64;
        for d in 0..rank {
            let od = out_shape[d].max(1) as usize;
            acc /= od;
            let c = (o / acc.max(1)) % od;
            xi += c as i64 * xs[d];
        }
        let s = xi as usize;
        out[o * es..(o + 1) * es].copy_from_slice(&x.data[s * es..(s + 1) * es]);
    }
    Ok(HostTensor::new(x.dtype, out_shape, out))
}

/// `Tile`: replicates `x` `repeats[d]` times along each dim.
pub fn tile(x: &HostTensor, repeats: &[i64]) -> Result<HostTensor> {
    x.validate()?;
    let rank = x.shape.len();
    ensure!(
        repeats.len() == rank,
        "Tile: repeats {} != rank {rank}",
        repeats.len()
    );
    ensure!(
        repeats.iter().all(|&repeat| repeat >= 0),
        "Tile: repeats negativi {repeats:?}"
    );
    let es = elem_size(x.dtype);
    let in_str = row_major_strides(&x.shape);
    let out_shape: Vec<i64> = (0..rank).map(|d| x.shape[d] * repeats[d]).collect();
    let n = element_count(&out_shape)?;
    let mut out = vec![0u8; n * es];
    for o in 0..n {
        let mut acc = n;
        let mut xi = 0i64;
        for d in 0..rank {
            let od = out_shape[d].max(1) as usize;
            acc /= od;
            let c = (o / acc.max(1)) % od;
            xi += (c as i64 % x.shape[d].max(1)) * in_str[d];
        }
        let s = xi as usize;
        out[o * es..(o + 1) * es].copy_from_slice(&x.data[s * es..(s + 1) * es]);
    }
    Ok(HostTensor::new(x.dtype, out_shape, out))
}

/// Row-major strides of a shape.
pub fn row_major_strides(shape: &[i64]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    let mut acc = 1i64;
    for d in (0..shape.len()).rev() {
        s[d] = acc;
        acc *= shape[d].max(0);
    }
    s
}

/// `Pad` mode=constant, host-side dtype-generic. `begins`/`ends` per dim,
/// `cval` = bytes of the constant value (length elem_size; empty ⇒ zeros).
pub fn pad(data: &HostTensor, begins: &[i64], ends: &[i64], cval: &[u8]) -> Result<HostTensor> {
    data.validate()?;
    let es = elem_size(data.dtype);
    ensure!(es > 0, "Pad: dtype {} with unknown size", data.dtype);
    let rank = data.shape.len();
    ensure!(
        begins.len() == rank && ends.len() == rank,
        "Pad: pads incompatibili con rank {rank}"
    );
    let in_str = row_major_strides(&data.shape);
    let out_shape: Vec<i64> = (0..rank)
        .map(|d| data.shape[d] + begins[d] + ends[d])
        .collect();
    let n = element_count(&out_shape)?;
    let mut out = vec![0u8; n * es];
    let cval_bytes = if cval.len() == es { Some(cval) } else { None };
    for o in 0..n {
        let mut acc = n;
        let mut si = 0i64;
        let mut oob = false;
        for d in 0..rank {
            let od = out_shape[d].max(1) as usize;
            acc /= od;
            let c = (o / acc.max(1)) % od;
            let ic = c as i64 - begins[d];
            if ic < 0 || ic >= data.shape[d] {
                oob = true;
            }
            si += ic * in_str[d];
        }
        if oob {
            if let Some(cv) = cval_bytes {
                out[o * es..(o + 1) * es].copy_from_slice(cv);
            }
        } else {
            let s = si as usize;
            out[o * es..(o + 1) * es].copy_from_slice(&data.data[s * es..(s + 1) * es]);
        }
    }
    Ok(HostTensor::new(data.dtype, out_shape, out))
}

/// Broadcast strides (in elements) of `in_shape` relative to `out_shape`
/// (right-aligned): 0 on broadcast or absent dimensions.
pub fn bcast_strides(in_shape: &[i64], out_shape: &[i64]) -> Vec<i64> {
    let r = out_shape.len();
    let ir = in_shape.len();
    let in_str = row_major_strides(in_shape);
    let mut out = vec![0i64; r];
    for off in 0..r {
        if off < ir {
            let id = ir - 1 - off;
            if in_shape[id] == out_shape[r - 1 - off] && in_shape[id] != 1 {
                out[r - 1 - off] = in_str[id];
            }
        }
    }
    out
}

/// `Where` host-side (cond ? x : y) with 3-way ONNX broadcasting. `out_shape`
/// is already the broadcast of the three shapes; x and y share the dtype (= output).
pub fn where_op(
    cond: &HostTensor,
    x: &HostTensor,
    y: &HostTensor,
    out_shape: &[i64],
) -> Result<HostTensor> {
    cond.validate()?;
    x.validate()?;
    y.validate()?;
    let xy_shape = broadcast(&x.shape, &y.shape)?.out_shape;
    let expected_shape = broadcast(&cond.shape, &xy_shape)?.out_shape;
    ensure!(
        expected_shape == out_shape,
        "Where: output {out_shape:?}, expected {expected_shape:?}"
    );
    let es = elem_size(x.dtype);
    ensure!(es > 0, "Where: dtype {} with unknown size", x.dtype);
    ensure!(x.dtype == y.dtype, "Where: X e Y dtype diversi");
    let rank = out_shape.len();
    let cs = bcast_strides(&cond.shape, out_shape);
    let xs = bcast_strides(&x.shape, out_shape);
    let ys = bcast_strides(&y.shape, out_shape);
    let cond_v = cond.to_i64()?;
    let n = element_count(out_shape)?;
    let mut out = vec![0u8; n * es];
    for o in 0..n {
        let mut acc = n;
        let (mut ci, mut xi, mut yi) = (0i64, 0i64, 0i64);
        for d in 0..rank {
            let od = out_shape[d].max(1) as usize;
            acc /= od;
            let c = (o / acc.max(1)) % od;
            ci += c as i64 * cs[d];
            xi += c as i64 * xs[d];
            yi += c as i64 * ys[d];
        }
        let (src, si) = if cond_v[ci as usize] != 0 {
            (x, xi as usize)
        } else {
            (y, yi as usize)
        };
        out[o * es..(o + 1) * es].copy_from_slice(&src.data[si * es..(si + 1) * es]);
    }
    Ok(HostTensor::new(x.dtype, out_shape.to_vec(), out))
}

/// Resolves `Slice` (opset ≥10) into (out_shape, start per dim, step per dim) with
/// ONNX clamping rules. Untouched dims stay full (start 0, step 1).
pub fn slice_params(
    shape: &[i64],
    starts: &[i64],
    ends: &[i64],
    axes: Option<&[i64]>,
    steps: Option<&[i64]>,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>)> {
    element_count(shape)?;
    let rank = shape.len();
    let mut st = vec![0i64; rank];
    let mut sp = vec![1i64; rank];
    let mut out = shape.to_vec();
    ensure!(
        starts.len() == ends.len(),
        "Slice: starts/ends of different length"
    );
    if let Some(axes) = axes {
        ensure!(
            axes.len() == starts.len(),
            "Slice: axes of different length than starts"
        );
    }
    if let Some(steps) = steps {
        ensure!(
            steps.len() == starts.len(),
            "Slice: steps of different length than starts"
        );
    }
    for i in 0..starts.len() {
        let ax = axes.map(|a| a[i]).unwrap_or(i as i64);
        let ax = if ax < 0 { ax + rank as i64 } else { ax };
        ensure!(
            (0..rank as i64).contains(&ax),
            "Slice: axis {ax} out of range"
        );
        let ax = ax as usize;
        let dim = shape[ax];
        let step = steps.map(|s| s[i]).unwrap_or(1);
        ensure!(step != 0, "Slice: step 0 is invalid");
        let mut s0 = starts[i];
        let mut e0 = ends[i];
        if s0 < 0 {
            s0 += dim;
        }
        if e0 < 0 {
            e0 += dim;
        }
        let (s0, e0) = if step > 0 {
            (s0.clamp(0, dim), e0.clamp(0, dim))
        } else {
            (s0.clamp(0, dim - 1), e0.clamp(-1, dim - 1))
        };
        let count = if step > 0 {
            if e0 > s0 {
                (e0 - s0 + step - 1) / step
            } else {
                0
            }
        } else if s0 > e0 {
            (s0 - e0 + (-step) - 1) / (-step)
        } else {
            0
        };
        st[ax] = s0;
        sp[ax] = step;
        out[ax] = count.max(0);
    }
    Ok((out, st, sp))
}

/// `Slice` host-side dtype-generic given the resolved parameters.
pub fn slice(data: &HostTensor, out_shape: &[i64], st: &[i64], sp: &[i64]) -> Result<HostTensor> {
    data.validate()?;
    let es = elem_size(data.dtype);
    ensure!(es > 0, "Slice: dtype {} with unknown size", data.dtype);
    let rank = data.shape.len();
    ensure!(
        out_shape.len() == rank && st.len() == rank && sp.len() == rank,
        "Slice: parametri incompatibili con rank {rank}"
    );
    let in_strides = row_major_strides(&data.shape);
    let n = element_count(out_shape)?;
    let mut out = vec![0u8; n * es];
    for o in 0..n {
        let mut acc = n;
        let mut src = 0i64;
        for d in 0..rank {
            let od = out_shape[d].max(1) as usize;
            acc /= od;
            let c = (o / acc.max(1)) % od;
            src += (st[d] + c as i64 * sp[d]) * in_strides[d];
        }
        let s = src as usize;
        out[o * es..(o + 1) * es].copy_from_slice(&data.data[s * es..(s + 1) * es]);
    }
    Ok(HostTensor::new(data.dtype, out_shape.to_vec(), out))
}

/// `Gather` host-side dtype-generic along `axis`: output =
/// `data.shape[:axis] + indices.shape + data.shape[axis+1:]`. Negative indices
/// normalized (+axis_dim).
pub fn gather(data: &HostTensor, indices: &HostTensor, axis: i64) -> Result<HostTensor> {
    data.validate()?;
    indices.validate()?;
    let es = elem_size(data.dtype);
    ensure!(es > 0, "Gather: dtype {} with unknown size", data.dtype);
    let rank = data.shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&ax),
        "Gather: axis {axis} out of range (rank {rank})"
    );
    let ax = ax as usize;
    let axis_dim = data.shape[ax];
    let inner: usize = data.shape[ax + 1..].iter().product::<i64>().max(1) as usize;
    let outer: usize = data.shape[..ax].iter().product::<i64>().max(1) as usize;
    let idx = indices.to_i64()?;
    let idx_count = idx.len();

    let mut out_shape = Vec::with_capacity(rank as usize - 1 + indices.shape.len());
    out_shape.extend_from_slice(&data.shape[..ax]);
    out_shape.extend_from_slice(&indices.shape);
    out_shape.extend_from_slice(&data.shape[ax + 1..]);
    let n = outer * idx_count * inner;
    let mut out = vec![0u8; n * es];
    for o in 0..n {
        let inner_i = o % inner;
        let idx_i = (o / inner) % idx_count;
        let outer_i = o / (inner * idx_count);
        let mut g = idx[idx_i];
        if g < 0 {
            g += axis_dim;
        }
        ensure!(
            (0..axis_dim).contains(&g),
            "Gather: index {g} out of range [0,{axis_dim})"
        );
        let src = outer_i * (axis_dim as usize * inner) + g as usize * inner + inner_i;
        out[o * es..(o + 1) * es].copy_from_slice(&data.data[src * es..(src + 1) * es]);
    }
    Ok(HostTensor::new(data.dtype, out_shape, out))
}

/// `GatherElements` along `axis`: `out[i][j][k] = data[i][idx[i][j][k]][k]` for
/// `axis = 1`. Indices and output have the same shape; only the coordinate
/// on the axis changes. Negative indices normalized (+axis_dim).
pub fn gather_elements(data: &HostTensor, indices: &HostTensor, axis: i64) -> Result<HostTensor> {
    data.validate()?;
    indices.validate()?;
    let es = elem_size(data.dtype);
    ensure!(es > 0, "GatherElements: dtype {} with no size", data.dtype);
    let rank = data.shape.len() as i64;
    ensure!(
        indices.shape.len() as i64 == rank,
        "GatherElements: rank indici {} ≠ rank dati {rank}",
        indices.shape.len()
    );
    let ax = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&ax),
        "GatherElements: axis {axis} out of range (rank {rank})"
    );
    let ax = ax as usize;
    let axis_dim = data.shape[ax];
    let data_strides = row_major_strides(&data.shape);
    let out_strides = row_major_strides(&indices.shape);
    let idx = indices.to_i64()?;

    let mut out = vec![0u8; idx.len() * es];
    for (o, &raw) in idx.iter().enumerate() {
        let mut g = raw;
        if g < 0 {
            g += axis_dim;
        }
        ensure!(
            (0..axis_dim).contains(&g),
            "GatherElements: index {g} out of range [0,{axis_dim})"
        );
        // same output coordinate on every axis, except `ax` where it equals `g`
        let mut src = 0i64;
        let mut rest = o as i64;
        for (d, stride) in out_strides.iter().enumerate() {
            let coord = if *stride == 0 { 0 } else { rest / stride };
            rest -= coord * stride;
            src += if d == ax { g } else { coord } * data_strides[d];
        }
        let src = src as usize;
        out[o * es..(o + 1) * es].copy_from_slice(&data.data[src * es..(src + 1) * es]);
    }
    Ok(HostTensor::new(data.dtype, indices.shape.clone(), out))
}

/// `ScatterND` with `reduction = none`: copies `data` and overwrites the
/// slices addressed by `indices`, whose last dimension gives how many initial
/// coordinates of `data` are fixed. The remaining slices are taken from
/// `updates` in the same order.
pub fn scatter_nd(
    data: &HostTensor,
    indices: &HostTensor,
    updates: &HostTensor,
) -> Result<HostTensor> {
    data.validate()?;
    indices.validate()?;
    updates.validate()?;
    ensure!(
        data.dtype == updates.dtype,
        "ScatterND: dtype updates {} ≠ data {}",
        updates.dtype,
        data.dtype
    );
    let es = elem_size(data.dtype);
    ensure!(es > 0, "ScatterND: dtype {} with no size", data.dtype);
    ensure!(!indices.shape.is_empty(), "ScatterND: indices di rank 0");
    let k = indices.shape[indices.shape.len() - 1] as usize;
    ensure!(
        k <= data.shape.len(),
        "ScatterND: indices con {k} coordinate su data di rank {}",
        data.shape.len()
    );
    // each row of `indices` addresses a slice of this length
    let slice_len: usize = data.shape[k..].iter().product::<i64>().max(1) as usize;
    let strides = row_major_strides(&data.shape);
    let idx = indices.to_i64()?;
    let rows = idx.len() / k.max(1);
    ensure!(
        updates.elem_count() == rows * slice_len,
        "ScatterND: updates con {} elementi, attesi {}",
        updates.elem_count(),
        rows * slice_len
    );

    let mut out = data.data.clone();
    for r in 0..rows {
        let mut base = 0i64;
        for d in 0..k {
            let mut c = idx[r * k + d];
            if c < 0 {
                c += data.shape[d];
            }
            ensure!(
                (0..data.shape[d]).contains(&c),
                "ScatterND: index {c} out of range [0,{}) on axis {d}",
                data.shape[d]
            );
            base += c * strides[d];
        }
        let dst = base as usize * es;
        let src = r * slice_len * es;
        out[dst..dst + slice_len * es].copy_from_slice(&updates.data[src..src + slice_len * es]);
    }
    Ok(HostTensor::new(data.dtype, data.shape.clone(), out))
}

/// `TopK` along `axis`: returns (values, int64 indices). Always sorts — with
/// `sorted = 0` ONNX leaves the order free, so sorting is conformant.
/// Ties follow the lowest index, as in the reference implementation.
pub fn top_k(
    x: &HostTensor,
    k: usize,
    axis: i64,
    largest: bool,
) -> Result<(HostTensor, HostTensor)> {
    x.validate()?;
    let rank = x.shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&ax),
        "TopK: axis {axis} out of range (rank {rank})"
    );
    let ax = ax as usize;
    let axis_dim = x.shape[ax] as usize;
    ensure!(
        k <= axis_dim,
        "TopK: k = {k} larger than the axis ({axis_dim})"
    );
    let inner: usize = x.shape[ax + 1..].iter().product::<i64>().max(1) as usize;
    let values = x.to_f32()?;

    let mut out_shape = x.shape.clone();
    out_shape[ax] = k as i64;
    let rows = values.len() / axis_dim;
    let mut out_v = Vec::with_capacity(rows * k);
    let mut out_i = Vec::with_capacity(rows * k);
    for r in 0..rows {
        let base = (r / inner) * axis_dim * inner + r % inner;
        let mut order: Vec<usize> = (0..axis_dim).collect();
        order.sort_by(|&a, &b| {
            let (va, vb) = (values[base + a * inner], values[base + b * inner]);
            let cmp = if largest {
                vb.total_cmp(&va)
            } else {
                va.total_cmp(&vb)
            };
            cmp.then(a.cmp(&b))
        });
        for &j in &order[..k] {
            out_v.push(values[base + j * inner]);
            out_i.push(j as i64);
        }
    }
    // rows were produced in `r` order, which iterates outer×inner:
    // recomposing the output means putting each row back in place
    let mut v = vec![0f32; rows * k];
    let mut ind = vec![0i64; rows * k];
    for r in 0..rows {
        let base = (r / inner) * k * inner + r % inner;
        for j in 0..k {
            v[base + j * inner] = out_v[r * k + j];
            ind[base + j * inner] = out_i[r * k + j];
        }
    }
    let vt = cast(&HostTensor::from_f32(out_shape.clone(), &v), x.dtype)?;
    Ok((vt, HostTensor::from_i64(out_shape, &ind)))
}

/// `Concat` host-side dtype-generic along `axis` (per-element bytes). All
/// inputs share dtype and shape except along the concatenated axis.
pub fn concat(inputs: &[HostTensor], axis: i64) -> Result<HostTensor> {
    ensure!(!inputs.is_empty(), "Concat: no input");
    for input in inputs {
        input.validate()?;
    }
    let dtype = inputs[0].dtype;
    let es = elem_size(dtype);
    ensure!(es > 0, "Concat: dtype {dtype} with unknown size");
    let rank = inputs[0].shape.len() as i64;
    let ax = if axis < 0 { axis + rank } else { axis };
    ensure!(
        (0..rank).contains(&ax),
        "Concat: axis {axis} out of range (rank {rank})"
    );
    let ax = ax as usize;

    let base = &inputs[0].shape;
    for tensor in inputs {
        ensure!(tensor.dtype == dtype, "Concat: dtype misti");
        ensure!(tensor.shape.len() == base.len(), "Concat: rank diversi");
        ensure!(
            tensor
                .shape
                .iter()
                .zip(base)
                .enumerate()
                .all(|(dimension, (actual, expected))| { dimension == ax || actual == expected }),
            "Concat: shape incompatibili {:?} e {base:?}",
            tensor.shape
        );
    }
    let inner: usize = base[ax + 1..].iter().product::<i64>().max(0) as usize;
    let outer: usize = base[..ax].iter().product::<i64>().max(0) as usize;
    let out_axis: i64 = inputs.iter().map(|t| t.shape[ax]).sum();

    let mut out_shape = base.clone();
    out_shape[ax] = out_axis;
    let out_block = out_axis as usize * inner; // per "outer" group
    let mut data = vec![0u8; outer * out_block * es];

    let mut off = 0usize; // current position along the axis in the output
    for t in inputs {
        let a = t.shape[ax].max(0) as usize;
        let in_block = a * inner;
        for o in 0..outer {
            let src = &t.data[o * in_block * es..(o * in_block + in_block) * es];
            let dst_start = (o * out_block + off * inner) * es;
            data[dst_start..dst_start + in_block * es].copy_from_slice(src);
        }
        off += a;
    }
    Ok(HostTensor::new(dtype, out_shape, data))
}

/// `DequantizeLinear` **per tensor**: `(x - zero_point) · scale`.
///
/// Per-axis quantization (a `scale` with more than one element) is rejected:
/// this exists to constant-fold weight dequantization at load time, and every
/// QDQ model measured so far quantizes its constants per tensor. Rejecting is
/// what keeps the fold from silently producing a wrong tensor.
pub fn dequantize_linear(
    x: &HostTensor,
    scale: &HostTensor,
    zero_point: Option<&HostTensor>,
) -> Result<HostTensor> {
    x.validate()?;
    scale.validate()?;
    let scale = match scale.to_f32()?[..] {
        [only] => only,
        _ => bail!("DequantizeLinear host: per-axis scale not supported"),
    };
    let zero = match zero_point {
        None => 0,
        Some(z) => {
            z.validate()?;
            ensure!(
                z.dtype == x.dtype,
                "DequantizeLinear: zero_point dtype {} != input {}",
                z.dtype,
                x.dtype
            );
            match z.to_i64()?[..] {
                [only] => only,
                _ => bail!("DequantizeLinear host: per-axis zero_point not supported"),
            }
        }
    };
    // The subtraction stays in integers: an int32 bias can exceed 2^24, where
    // converting to f32 first and subtracting after would round differently.
    let values: Vec<f32> = x
        .to_i64()?
        .into_iter()
        .map(|v| (v - zero) as f32 * scale)
        .collect();
    Ok(HostTensor::from_f32(x.shape.clone(), &values))
}

/// Total bytes expected for a `HostTensor` given dtype+shape (invariant check).
pub fn expected_bytes(dtype: i32, shape: &[i64]) -> usize {
    element_count(shape)
        .ok()
        .and_then(|count| storage_len(dtype, count))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_broadcasts_integer_shape_math() {
        let matrix = HostTensor::from_i64(vec![2, 2], &[1, 2, 3, 4]);
        let row = HostTensor::from_i64(vec![2], &[10, 20]);

        let result = binary(&matrix, &row, BinOp::Add).expect("broadcast valido");

        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.to_i64().expect("tensor int64"), vec![11, 22, 13, 24]);
    }

    #[test]
    fn gather_normalizes_negative_indices() {
        let values = HostTensor::from_i64(vec![3], &[10, 20, 30]);
        let indices = HostTensor::from_i64(vec![2], &[-1, 0]);

        let result = gather(&values, &indices, 0).expect("indici validi");

        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.to_i64().expect("tensor int64"), vec![30, 10]);
    }

    #[test]
    fn slice_supports_negative_steps() {
        let values = HostTensor::from_i64(vec![5], &[0, 1, 2, 3, 4]);
        let (shape, starts, steps) =
            slice_params(&values.shape, &[4], &[-6], None, Some(&[-1])).expect("slice valida");

        let result = slice(&values, &shape, &starts, &steps).expect("slice eseguibile");

        assert_eq!(result.to_i64().expect("tensor int64"), vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn integer_division_by_zero_is_an_error() {
        let numerator = HostTensor::from_i64(vec![], &[1]);
        let denominator = HostTensor::from_i64(vec![], &[0]);

        assert!(matches!(
            binary(&numerator, &denominator, BinOp::Div),
            Err(Error::InvalidTensor(_))
        ));
    }

    #[test]
    fn rejects_incoherent_tensor_storage() {
        let tensor = HostTensor::new(INT64, vec![2], vec![0; 8]);

        assert!(matches!(tensor.to_i64(), Err(Error::InvalidTensor(_))));
    }
}
