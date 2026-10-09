//! Elementwise WGSL sources: binary/unary templates with ONNX broadcasting,
//! cast and ternary select.

pub const MAX_RANK: usize = 8;

pub const BINARY: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push {
    n: u32, rank: u32, pad0: u32, pad1: u32,
    os0: vec4<u32>, os1: vec4<u32>,
    as0: vec4<u32>, as1: vec4<u32>,
    bs0: vec4<u32>, bs1: vec4<u32>,
}
var<immediate> pc: Push;

fn dim(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
    if (d < 4u) { return v0[d]; }
    return v1[d - 4u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    var rem = i;
    var off_a = 0u;
    var off_b = 0u;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let os = dim(pc.os0, pc.os1, d);
        let c = rem / os;
        rem = rem % os;
        off_a = off_a + c * dim(pc.as0, pc.as1, d);
        off_b = off_b + c * dim(pc.bs0, pc.bs1, d);
    }
    out[i] = OP;
}
"#;

pub const UNARY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { n: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    let v = x[i];
    out[i] = OP;
}
"#;

/// ONNX `Pow` on negative base: WGSL `pow()` is defined only for base ≥ 0
/// (lowered to `exp2(y·log2(x))`), while ONNX permits `(-2)^2 = 4`. With
/// integer exponent module is computed and sign re-applied; with non-integer
/// exponent and negative base result remains NaN per IEEE math.
pub const POW_EXPR: &str = concat!(
    "select(pow(abs(a[off_a]), b[off_b]), -pow(abs(a[off_a]), b[off_b]), ",
    "a[off_a] < 0.0 && abs(b[off_b] - round(b[off_b])) < 1e-6 ",
    "&& (i32(round(b[off_b])) & 1) != 0)"
);

/// Auxiliary functions prepended to [`UNARY`] when used by expression.
/// WGSL lacks `erf`: Abramowitz-Stegun 7.1.26 rational approximation
/// (max error ~1.5e-7, well below test tolerance).
pub const UNARY_HELPERS_ERF: &str = r#"
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-a * a));
}
"#;

/// The exact GELU as `Div → Erf → Add → Mul → Mul`, in one pass (prepend
/// [`UNARY_HELPERS_ERF`]). Each step rounds as its own kernel does: the same
/// `erf_approx`, and the constants the graph divides, adds and multiplies by
/// as push constants instead of broadcast operands. The one multiply feeding an
/// add across the old kernel boundaries is the sign in `erf_approx` times its
/// magnitude; that product is exact, so a driver fusing it with `+ offset`
/// into an FMA rounds the same as the two separate kernels did. `factor_first`
/// selects the graph's order of the two products: `(v · e) · factor`, or
/// `v · (e · factor)`.
pub const DECOMPOSED_GELU: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { n: u32, divisor: f32, offset: f32, factor: f32, factor_first: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    out[i] = decomposed_gelu(x[i]);
}
"#;

/// The function [`DECOMPOSED_GELU`] applies per element (prepend
/// [`UNARY_HELPERS_ERF`]); reads `divisor`, `offset`, `factor` and
/// `factor_first` from the shader's push constants, so a kernel that quantizes
/// the GELU computes the very same value.
pub const DECOMPOSED_GELU_FN: &str = r#"
fn decomposed_gelu(v: f32) -> f32 {
    let e = erf_approx(v / pc.divisor) + pc.offset;
    if (pc.factor_first != 0u) {
        return v * (e * pc.factor);
    }
    return (v * e) * pc.factor;
}
"#;

pub const DECOMPOSED_GELU_BINDINGS: u32 = 2;
pub const DECOMPOSED_GELU_PUSH_BYTES: u32 = 20;

/// Interleaved rotary embedding, `x · cos + rot(x) · sin` with an optional
/// input transpose, output transpose and scale (`glsl/interleaved_rope.comp`).
/// Compiled from GLSL because it needs `precise`: see the source.
const INTERLEAVED_ROPE_SPV: &[u8] = include_bytes!("spv/interleaved_rope.spv");

/// The SPIR-V words of [`INTERLEAVED_ROPE_SPV`], from its little-endian bytes.
pub fn interleaved_rope_spirv() -> Vec<u32> {
    INTERLEAVED_ROPE_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub const INTERLEAVED_ROPE_BINDINGS: u32 = 4;
pub const INTERLEAVED_ROPE_PUSH_BYTES: u32 = 24;
/// `x` is `[B, S, H, D]`: the Transpose to `[B, H, S, D]` is folded in.
pub const INTERLEAVED_ROPE_BSHD: u32 = 1;
/// The output is `[B, H, D, S]`: the Transpose of K for `Q · Kᵀ` is folded in.
pub const INTERLEAVED_ROPE_TRANSPOSE: u32 = 2;
/// The result is multiplied by the push-constant scale.
pub const INTERLEAVED_ROPE_SCALE: u32 = 4;

/// `max(x · scale + bias, 0)` with scale and bias one value or one per channel,
/// the `Mul → Add → Relu` that `rewrite::fuse_mul_add_relu` replaces
/// (`glsl/mul_add_relu.comp`). Compiled from GLSL because it needs `precise`.
const MUL_ADD_RELU_SPV: &[u8] = include_bytes!("spv/mul_add_relu.spv");

/// The SPIR-V words of [`MUL_ADD_RELU_SPV`], from its little-endian bytes.
pub fn mul_add_relu_spirv() -> Vec<u32> {
    MUL_ADD_RELU_SPV
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub const MUL_ADD_RELU_BINDINGS: u32 = 4;
pub const MUL_ADD_RELU_PUSH_BYTES: u32 = 20;

/// Side of the shared-memory tile the transposed output goes through.
pub const INTERLEAVED_ROPE_TILE: usize = 32;

/// Workgroups for `bh` slices of `[S, D]`: one thread per output element, or,
/// for the transposed output, one workgroup per `TILE × TILE` tile of `(s, d)`
/// and per slice. `None` when a count does not fit a `u32`.
pub fn interleaved_rope_groups(bh: usize, s: usize, d: usize, transpose: bool) -> Option<[u32; 3]> {
    let groups = if transpose {
        let tile = INTERLEAVED_ROPE_TILE;
        [s.div_ceil(tile), d.div_ceil(tile), bh]
    } else {
        [(bh * s * d).div_ceil(256), 1, 1]
    };
    let [x, y, z] = groups.map(|count| u32::try_from(count).ok());
    Some([x?, y?, z?])
}

/// `Clip`: bounds are uniform across tensor, traveling via push
/// constants instead of binding (missing `min`/`max` becomes ±inf).
pub const CLIP: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { n: u32, lo: f32, hi: f32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    out[i] = clamp(x[i], pc.lo, pc.hi);
}
"#;

pub const CLIP_BINDINGS: u32 = 2;
pub const CLIP_PUSH_BYTES: u32 = 12;

pub const CAST_I32_F32: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { n: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    out[i] = f32(x[i]);
}
"#;

/// Device↔device cast between i32 and f32 on 32-bit words: `mode == 0` → i32→f32,
/// otherwise f32→i32.
pub const CAST_DEV: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
struct Push { n: u32, mode: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    if (pc.mode == 0u) {
        dst[i] = bitcast<u32>(f32(bitcast<i32>(src[i])));
    } else {
        dst[i] = bitcast<u32>(i32(bitcast<f32>(src[i])));
    }
}
"#;

/// `Where`: ternary select with broadcasting; per-dim strides (out, cond, x, y)
/// arrive in an i32 `params` buffer preceded by the rank.
pub const WHERE: &str = r#"
@group(0) @binding(0) var<storage, read> cond: array<u32>; // bool packed a byte
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> params: array<i32>;
struct Push { n: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.n) { return; }
    let rank = u32(params[0]);
    var acc = pc.n;
    var ci = 0i;
    var xi = 0i;
    var yi = 0i;
    for (var d = 0u; d < rank; d = d + 1u) {
        let odim = u32(params[1u + d]);
        acc = acc / odim;
        let c = i32((o / acc) % odim);
        ci = ci + c * params[1u + rank + d];
        xi = xi + c * params[1u + 2u * rank + d];
        yi = yi + c * params[1u + 3u * rank + d];
    }
    let cu = u32(ci);
    let cbyte = (cond[cu >> 2u] >> ((cu & 3u) * 8u)) & 0xffu;
    out[o] = select(y[u32(yi)], x[u32(xi)], cbyte != 0u);
}
"#;

/// f32 `IsNaN` into a bool tensor packed one byte per element, the layout
/// `WHERE` reads its condition in: one thread per 32-bit word of four.
/// The test is on the bits, not `v != v`, which a compiler may fold away.
pub const IS_NAN: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
struct Push { n: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w * 4u >= pc.n) { return; }
    var word = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let i = w * 4u + k;
        if (i < pc.n && (x[i] & 0x7fffffffu) > 0x7f800000u) {
            word = word | (1u << (k * 8u));
        }
    }
    out[w] = word;
}
"#;

/// `PackPositiveBits`: `numpy.packbits(x > 0, axis = -1)` over a last axis
/// whose length is a multiple of 8, so byte `i` packs `x[8i .. 8i + 8]` of the
/// flattened tensor, the first element in the most significant bit. One
/// invocation writes one u32 word (four bytes, little-endian); the output
/// buffer is rounded up to whole words. Grid-stride, so any size fits the
/// workgroup limit.
///
/// `x > 0` is decided on the bits, not with a float comparison: the GPU
/// flushes subnormals to zero, so `1e-40 > 0.0` is false there and true for
/// ONNX `Greater` on the CPU. Positive means sign clear, not ±0 and not NaN:
/// `0 < bits ≤ 0x7f800000`, i.e. `bits - 1 < 0x7f800000` in u32.
pub const PACK_POSITIVE_BITS: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> bits: array<u32>;
struct Push { bytes: u32 }
var<immediate> pc: Push;
fn nibble(v: vec4<u32>) -> u32 {
    let p = (v - vec4<u32>(1u)) < vec4<u32>(0x7f800000u);
    return select(0u, 8u, p.x) | select(0u, 4u, p.y) | select(0u, 2u, p.z) | select(0u, 1u, p.w);
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    let words = (pc.bytes + 3u) / 4u;
    for (var w = gid.x; w < words; w = w + groups.x * 256u) {
        var word = 0u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            let byte = w * 4u + k;
            if (byte < pc.bytes) {
                let value = (nibble(x[byte * 2u]) << 4u) | nibble(x[byte * 2u + 1u]);
                word = word | (value << (k * 8u));
            }
        }
        bits[w] = word;
    }
}
"#;

/// `AddTransposePad`: `Pad(Transpose(Reshape(Add(a, b))))` in one pass over
/// the padded output, which is written contiguously. Up to four axes, outer
/// first, after the host has dropped unit axes and merged neighbours: per axis
/// the padded `extent`, the `begin` padding, the `interior` length and the
/// element strides of `a` and `b` along it (0 where they broadcast). Outside
/// the interior the output is `value`; inside it is the same single f32
/// `a + b` as the `BINARY` template, so the bits do not change. Grid-stride,
/// so any size fits the workgroup limit.
pub const ADD_TRANSPOSE_PAD: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

struct Push {
    n: u32, value: f32, pad0: u32, pad1: u32,
    extent: vec4<u32>, begin: vec4<u32>, interior: vec4<u32>,
    sa: vec4<u32>, sb: vec4<u32>,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
    for (var i = gid.x; i < pc.n; i = i + groups.x * 256u) {
        var rem = i;
        var off_a = 0u;
        var off_b = 0u;
        var inside = true;
        for (var d = 4u; d > 0u; d = d - 1u) {
            let e = d - 1u;
            let c = rem % pc.extent[e];
            rem = rem / pc.extent[e];
            // a coordinate before `begin` wraps around past `interior`
            let k = c - pc.begin[e];
            inside = inside && k < pc.interior[e];
            off_a = off_a + k * pc.sa[e];
            off_b = off_b + k * pc.sb[e];
        }
        if (inside) {
            out[i] = a[off_a] + b[off_b];
        } else {
            out[i] = pc.value;
        }
    }
}
"#;

pub const ADD_TRANSPOSE_PAD_BINDINGS: u32 = 3;
pub const ADD_TRANSPOSE_PAD_PUSH_BYTES: u32 = 96;
pub const CAST_DEV_BINDINGS: u32 = 2;
pub const CAST_DEV_PUSH_BYTES: u32 = 8;
pub const WHERE_BINDINGS: u32 = 5;
pub const WHERE_PUSH_BYTES: u32 = 4;
pub const IS_NAN_BINDINGS: u32 = 2;
pub const IS_NAN_PUSH_BYTES: u32 = 4;
pub const PACK_POSITIVE_BITS_BINDINGS: u32 = 2;
pub const PACK_POSITIVE_BITS_PUSH_BYTES: u32 = 4;

#[cfg(test)]
mod tests {
    use super::*;
    use vk_compute::compile_wgsl;

    #[test]
    fn shared_elementwise_sources_compile() {
        compile_wgsl(&BINARY.replace("OP", "a[off_a] + b[off_b]")).expect("shader binary valido");
        compile_wgsl(&UNARY.replace("OP", "max(v, 0.0)")).expect("shader unary valido");
        compile_wgsl(CAST_I32_F32).expect("shader cast valido");
        compile_wgsl(CAST_DEV).expect("shader cast device valido");
        compile_wgsl(WHERE).expect("shader Where valido");
        compile_wgsl(IS_NAN).expect("shader IsNaN valido");
        compile_wgsl(PACK_POSITIVE_BITS).expect("shader PackPositiveBits valido");
    }
}
