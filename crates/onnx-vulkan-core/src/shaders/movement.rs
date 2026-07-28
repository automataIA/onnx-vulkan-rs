//! Data movement WGSL sources: permutation, concatenation, gather,
//! padding and slicing.

/// Generic permutation of 32-bit elements, up to rank 8.
pub const TRANSPOSE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Push {
    n: u32, rank: u32, pad0: u32, pad1: u32,
    osh0: vec4<u32>, osh1: vec4<u32>,
    ist0: vec4<u32>, ist1: vec4<u32>,
}
var<immediate> pc: Push;

fn dim(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
    if (d < 4u) { return v0[d]; }
    return v1[d - 4u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.n) { return; }
    var src_idx = 0u;
    var acc = pc.n;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let sh = dim(pc.osh0, pc.osh1, d);
        acc = acc / sh;
        let c = (o / acc) % sh;
        src_idx = src_idx + c * dim(pc.ist0, pc.ist1, d);
    }
    dst[o] = src[src_idx];
}
"#;

/// `Concat`: copies an input into concatenation axis slice, offset `off`.
pub const CONCAT: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
struct Push { n: u32, inner: u32, a: u32, out_axis: u32, off: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= pc.n) { return; }
    let inner_i = e % pc.inner;
    let tmp = e / pc.inner;
    let axis_i = tmp % pc.a;
    let outer_i = tmp / pc.a;
    let o = outer_i * (pc.out_axis * pc.inner) + (pc.off + axis_i) * pc.inner + inner_i;
    dst[o] = src[e];
}
"#;

/// f32 `Gather` along an axis, indices i32 already normalized.
pub const GATHER: &str = r#"
@group(0) @binding(0) var<storage, read> data: array<f32>;
@group(0) @binding(1) var<storage, read> idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Push { n: u32, inner: u32, idx_count: u32, axis_dim: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.n) { return; }
    let inner_i = o % pc.inner;
    let idx_i = (o / pc.inner) % pc.idx_count;
    let outer_i = o / (pc.inner * pc.idx_count);
    let g = u32(idx[idx_i]);
    let src = outer_i * (pc.axis_dim * pc.inner) + g * pc.inner + inner_i;
    out[o] = data[src];
}
"#;

/// `Pad` in `constant` mode; output shape, initial pads, shape and strides
/// of input arrive in an i32 `params` buffer.
pub const PAD: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<i32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct Push { n: u32, rank: u32, cval: f32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.n) { return; }
    var acc = pc.n;
    var si = 0i;
    var oob = false;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let odim = u32(params[1u + d]);
        acc = acc / odim;
        let c = i32((o / acc) % odim);
        let ic = c - params[1u + pc.rank + d];
        let idim = params[1u + 2u * pc.rank + d];
        if (ic < 0 || ic >= idim) { oob = true; }
        si = si + ic * params[1u + 3u * pc.rank + d];
    }
    if (oob) { dst[o] = pc.cval; } else { dst[o] = src[u32(si)]; }
}
"#;

/// `Slice` with per-dim start/step/stride in an i32 `params` buffer.
pub const SLICE: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<i32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
struct Push { n: u32, rank: u32 }
var<immediate> pc: Push;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.n) { return; }
    var acc = pc.n;
    var si = 0i;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let odim = u32(params[d]);
        acc = acc / odim;
        let c = i32((o / acc) % odim);
        let start = params[2u * pc.rank + d];
        let step = params[3u * pc.rank + d];
        let istride = params[pc.rank + d];
        si = si + (start + c * step) * istride;
    }
    dst[o] = src[u32(si)];
}
"#;

pub const TRANSPOSE_BINDINGS: u32 = 2;
pub const TRANSPOSE_PUSH_BYTES: u32 = 80;
pub const CONCAT_BINDINGS: u32 = 2;
pub const CONCAT_PUSH_BYTES: u32 = 20;
pub const GATHER_BINDINGS: u32 = 3;
pub const GATHER_PUSH_BYTES: u32 = 16;
pub const PAD_BINDINGS: u32 = 3;
pub const PAD_PUSH_BYTES: u32 = 12;
pub const SLICE_BINDINGS: u32 = 3;
pub const SLICE_PUSH_BYTES: u32 = 8;

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for source in [
            super::TRANSPOSE,
            super::CONCAT,
            super::GATHER,
            super::PAD,
            super::SLICE,
        ] {
            vk_compute::compile_wgsl(source).expect("shader di movimento valido");
        }
    }
}
