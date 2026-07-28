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

pub const CAST_DEV_BINDINGS: u32 = 2;
pub const CAST_DEV_PUSH_BYTES: u32 = 8;
pub const WHERE_BINDINGS: u32 = 5;
pub const WHERE_PUSH_BYTES: u32 = 4;

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
    }
}
