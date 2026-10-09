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

/// `TRANSPOSE` for 1-byte elements (u8/i8 packed four per word). One
/// invocation assembles one output word, so no two write the same one. Same
/// push layout and bindings as `TRANSPOSE`; dispatch `ceil(n / 4)` threads.
pub const TRANSPOSE_BYTES: &str = r#"
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

fn source(o: u32) -> u32 {
    var src_idx = 0u;
    var acc = pc.n;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let sh = dim(pc.osh0, pc.osh1, d);
        acc = acc / sh;
        let c = (o / acc) % sh;
        src_idx = src_idx + c * dim(pc.ist0, pc.ist1, d);
    }
    return src_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let base = gid.x * 4u;
    if (base >= pc.n) { return; }
    var word = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        if (base + j < pc.n) {
            let s = source(base + j);
            let byte = (src[s >> 2u] >> ((s & 3u) * 8u)) & 0xffu;
            word = word | (byte << (j * 8u));
        }
    }
    dst[gid.x] = word;
}
"#;

/// Permutation that keeps the innermost axis in place, as a copy of
/// contiguous rows of 16-byte words: SAM 3's window partition `[B, a, u, b,
/// v, C] → [B, a, b, u, v, C]` moves rows of `v·C` elements. The outer
/// coordinates are decoded once per word instead of once per element. Push:
/// `n` words, `row` words per row, the outer `rank` (≤ 8), the outer output
/// dims and the input strides of the outer dims, in words.
pub const TRANSPOSE_ROWS: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;

struct Push {
    n: u32, rank: u32, row: u32, pad0: u32,
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
    let r = o / pc.row;
    var src_idx = o - r * pc.row;
    var acc = pc.n / pc.row;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let sh = dim(pc.osh0, pc.osh1, d);
        acc = acc / sh;
        let c = (r / acc) % sh;
        src_idx = src_idx + c * dim(pc.ist0, pc.ist1, d);
    }
    dst[o] = src[src_idx];
}
"#;

/// Permutation of 32-bit elements that moves the innermost axis, through a
/// 32×32 tile in workgroup memory so both the reads and the writes are
/// contiguous: `TRANSPOSE` reads one strided word per invocation. The tile
/// spans axis `a`, innermost in the input, and axis `b`, innermost in the
/// output; every other axis is a batch axis, one per `z` workgroup. A unit is
/// `unit` (1 to 4) adjacent words that stay together, as the pairs SAM 3's
/// pixel shuffle moves. Push: the extents of `a` and `b` in units, the input
/// stride of `b` and the output stride of `a` in words, the unit, the batch
/// rank (≤ 4), then the batch dims and their input and output strides in
/// words. Dispatch `[ceil(da / 32), ceil(db / 32), batch]` workgroups.
pub const TRANSPOSE_TILED: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Push {
    da: u32, db: u32, in_b: u32, out_a: u32,
    unit: u32, rank: u32, pad0: u32, pad1: u32,
    bdim: vec4<u32>, bin: vec4<u32>, bout: vec4<u32>,
}
var<immediate> pc: Push;

// 32 rows of 32 units of up to 4 words, each row one word longer so the
// column reads of single words fall in distinct banks
var<workgroup> tile: array<u32, 4128>;

@compute @workgroup_size(32, 8)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var z = wg.z;
    var in_off = 0u;
    var out_off = 0u;
    for (var d = 0u; d < pc.rank; d = d + 1u) {
        let c = z % pc.bdim[d];
        z = z / pc.bdim[d];
        in_off = in_off + c * pc.bin[d];
        out_off = out_off + c * pc.bout[d];
    }
    let u = pc.unit;
    let row = 32u * u + 1u;
    let a0 = wg.x * 32u;
    let b0 = wg.y * 32u;
    let count_a = min(32u, pc.da - a0);
    let count_b = min(32u, pc.db - b0);
    // tile row ib: the units a0.. at b0 + ib, contiguous in the input
    let words_a = count_a * u;
    for (var ib = lid.y; ib < count_b; ib = ib + 8u) {
        let base = in_off + (b0 + ib) * pc.in_b + a0 * u;
        for (var w = lid.x; w < words_a; w = w + 32u) {
            tile[ib * row + w] = src[base + w];
        }
    }
    workgroupBarrier();
    // output row ia: the units b0.. at a0 + ia, contiguous in the output
    let words_b = count_b * u;
    for (var ia = lid.y; ia < count_a; ia = ia + 8u) {
        let base = out_off + (a0 + ia) * pc.out_a + b0 * u;
        for (var v = lid.x; v < words_b; v = v + 32u) {
            let ib = v / u;
            dst[base + v] = tile[ib * row + ia * u + (v - ib * u)];
        }
    }
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
            super::TRANSPOSE_ROWS,
            super::TRANSPOSE_TILED,
            super::CONCAT,
            super::GATHER,
            super::PAD,
            super::SLICE,
        ] {
            vk_compute::compile_wgsl(source).expect("shader di movimento valido");
        }
    }
}
