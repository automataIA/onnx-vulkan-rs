//! Shared shaders and dispatch layouts for `ConvInteger`.
//!
//! The same ladder as [`crate::shaders::conv`], in `i32`: a direct kernel for
//! the grouped cases that are not one GEMM, an implicit tiled GEMM for
//! `group == 1`, a 64×64 blocked variant of it, and a split-K variant of that.
//! Routing reuses [`conv::prefer_blocked`] and [`conv::split_k`] unchanged —
//! measured on the 23 nodes of resnet50-int8 that reach this kernel, the f32
//! calibration picks within **1.9%** of the best configuration per geometry
//! (9.48× against a 9.66× oracle, `examples/conv_integer_gemm`), so the integer
//! path adds no constant of its own.
//!
//! Two things differ from the f32 case, both in this path's favour:
//!
//! - **Split-K is exact.** `i32` addition is associative, and the accumulator
//!   cannot overflow at these depths — `K · 255 · 255 ≤ 4608 · 65025 ≈ 3.0e8`
//!   against `i32::MAX ≈ 2.1e9` — so slicing `K` reproduces the unsplit result
//!   bit for bit. The f32 reduction only reassociates within tolerance.
//! - **There is no bias.** `ConvInteger` returns the raw int32 accumulator and
//!   the requantize epilogue owns everything after it, so the reduction has
//!   nothing it must be careful not to add once per slice.

use crate::shaders::conv;

pub const BINDINGS: u32 = 5;
/// 20 u32 fields in the push constant struct. The last is `split`, read only by
/// [`blocked_splitk`]; the single-pass kernels ignore it, so one layout serves
/// every variant.
pub const PUSH_BYTES: u32 = 80;
/// Output tiles, shared with the f32 kernels so the routing predicates apply
/// to the same quantities they were calibrated on.
pub const TILE_SIZE: u32 = conv::TILE_SIZE;
pub const BLOCKED_TILE_SIZE: u32 = conv::BLOCKED_TILE_SIZE;
pub const SPLIT_REDUCE_BINDINGS: u32 = 2;

/// Declarations shared by every variant.
///
/// ONNX allows both `uint8` and `int8` for X and W independently:
/// `x_signed`/`w_signed` select sign extension of the bytes and of the zero
/// points, which are per-tensor scalars read from a buffer (no readback).
const PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;   // u8 packed
@group(0) @binding(1) var<storage, read> w: array<u32>;   // u8 packed
@group(0) @binding(2) var<storage, read> azp: array<u32>;
@group(0) @binding(3) var<storage, read> wzp: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<i32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32,
    x_signed: u32, w_signed: u32, split: u32,
}
var<immediate> pc: Push;

/// Byte as signed integer if `signed`, else unsigned.
fn as_signed(raw: u32, signed: u32) -> i32 {
    let v = i32(raw & 0xffu);
    if (signed != 0u && v > 127) { return v - 256; }
    return v;
}
fn xat(idx: u32) -> i32 { return as_signed(x[idx >> 2u] >> ((idx & 3u) * 8u), pc.x_signed); }
fn wat(idx: u32) -> i32 { return as_signed(w[idx >> 2u] >> ((idx & 3u) * 8u), pc.w_signed); }
"#;

/// Direct int8→i32 quantized 1D/2D Conv (1D normalized to 2D with W=1): one
/// thread per output element, with group/stride/pad/dilation.
///
/// This is the only variant grouped convolutions can use. A depthwise
/// convolution is not one GEMM — each output channel sees only its own slice of
/// the input, so rows cannot share a staged column — and its `K` is
/// `KH · KW = 9` on every such node in the suite, which leaves nothing to tile
/// and nothing to split.
pub fn direct() -> String {
    format!(
        "{PRELUDE}{}",
        r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let t2 = t1 / pc.h_out;
    let m = t2 % pc.c_out;
    let bn = t2 / pc.c_out;
    let gso = pc.c_out / pc.group;
    let g = m / gso;
    let a_zp = as_signed(azp[0], pc.x_signed);
    let w_zp = as_signed(wzp[0], pc.w_signed);
    var acc = 0i;
    for (var cg = 0u; cg < pc.gsi; cg = cg + 1u) {
        let ic = g * pc.gsi + cg;
        for (var r = 0u; r < pc.kh; r = r + 1u) {
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            if (ih < 0 || ih >= i32(pc.h_in)) { continue; }
            for (var s = 0u; s < pc.kw; s = s + 1u) {
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
                if (iw < 0 || iw >= i32(pc.w_in)) { continue; }
                let xidx = ((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw);
                let widx = ((m * pc.gsi + cg) * pc.kh + r) * pc.kw + s;
                acc = acc + (xat(xidx) - a_zp) * (wat(widx) - w_zp);
            }
        }
    }
    out[o] = acc;
}
"#
    )
}

/// Implicit GEMM for `group == 1`, the integer twin of [`conv::implicit_gemm`]:
/// `out[C_out, P] = W[C_out, C_in·KH·KW] × X_im2col[C_in·KH·KW, P]`. The im2col
/// matrix is never materialized — its columns are rebuilt from their `K` index.
///
/// The zero points are folded at staging time, so the inner loop is a plain
/// integer dot product. That also makes the padding fall out for free: out of
/// bounds means the quantized zero, which *is* `a_zp`, and `a_zp - a_zp = 0` is
/// exactly the value the f32 kernel writes there.
pub fn implicit_gemm() -> String {
    format!(
        "{PRELUDE}{}",
        r#"
const TILE = 16u;
var<workgroup> w_tile: array<i32, 256>;
var<workgroup> x_tile: array<i32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ty = lid.y;
    let tx = lid.x;
    let m = wid.y * TILE + ty;           // output channel = row
    let p = wid.x * TILE + tx;           // output pixel = column
    let bn = wid.z;                      // batch image
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let a_zp = as_signed(azp[0], pc.x_signed);
    let w_zp = as_signed(wzp[0], pc.w_signed);

    var acc = 0i;
    let ntiles = (kdepth + TILE - 1u) / TILE;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let kw_idx = t * TILE + tx;
        var wv = 0i;
        if (m < pc.c_out && kw_idx < kdepth) { wv = wat(m * kdepth + kw_idx) - w_zp; }
        w_tile[ty * TILE + tx] = wv;
        let kx_idx = t * TILE + ty;
        var value = 0i;
        if (p < pixels && kx_idx < kdepth) {
            let ksize = pc.kh * pc.kw;
            let ic = kx_idx / ksize;
            let rem = kx_idx % ksize;
            let r = rem / pc.kw;
            let s = rem % pc.kw;
            let oh = p / pc.w_out;
            let ow = p % pc.w_out;
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
            // out of bounds = the quantized zero, i.e. a_zp, i.e. 0 once folded
            if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                value = xat(((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)) - a_zp;
            }
        }
        x_tile[ty * TILE + tx] = value;
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + w_tile[ty * TILE + i] * x_tile[i * TILE + tx];
        }
        workgroupBarrier();
    }
    if (m >= pc.c_out || p >= pixels) { return; }
    out[(bn * pc.c_out + m) * pixels + p] = acc;
}
"#
    )
}

/// The same implicit GEMM on a 64×64 output tile with a 4×4 micro-tile held in
/// registers, for the geometries [`conv::prefer_blocked`] accepts.
///
/// [`implicit_gemm`] keeps one output per thread, so its inner loop spends two
/// shared reads per multiply-add; here 8 reads feed 16. Accumulation order is
/// identical — both walk `K` in steps of 16 — and the arithmetic is integer, so
/// the two kernels agree exactly and routing between them cannot move an output.
pub fn blocked() -> String {
    blocked_body(false)
}

/// [`blocked`] with its `K` loop sliced across `wid.z`, writing one partial
/// image per slice for [`SPLIT_REDUCE`] to sum.
///
/// `wid.z` carries both the batch image and the slice — `bn = z / split`,
/// `slice = z % split` — since the grid has only three dimensions and the batch
/// already owned this one.
pub fn blocked_splitk() -> String {
    blocked_body(true)
}

fn blocked_body(split: bool) -> String {
    // the three lines the split-K variant changes: which slice of K this
    // workgroup walks, and where its partial goes
    let (batch, bounds, store) = if split {
        (
            "let bn = wid.z / pc.split;\n    let slice = wid.z % pc.split;",
            "let tper = (ntiles + pc.split - 1u) / pc.split;\n\
             \x20   let tstart = slice * tper;\n\
             \x20   var tend = tstart + tper;\n\
             \x20   if (tend > ntiles) { tend = ntiles; }",
            "out[slice * pc.total + (bn * pc.c_out + m) * pixels + p] = accv[j];",
        )
    } else {
        (
            "let bn = wid.z;",
            "let tstart = 0u;\n    let tend = ntiles;",
            "out[(bn * pc.c_out + m) * pixels + p] = accv[j];",
        )
    };
    format!(
        "{PRELUDE}{}",
        r#"
const TILE = 64u;
const KSTEP = 16u;
var<workgroup> w_tile: array<i32, 1024>;
var<workgroup> x_tile: array<i32, 1024>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.y * TILE;          // first output channel of the block
    let col0 = wid.x * TILE;          // first output pixel of the block
    {batch}
    let pixels = pc.h_out * pc.w_out;
    let kdepth = pc.c_in * pc.kh * pc.kw;
    let ksize = pc.kh * pc.kw;
    let a_zp = as_signed(azp[0], pc.x_signed);
    let w_zp = as_signed(wzp[0], pc.w_signed);

    var acc0 = vec4<i32>(0);
    var acc1 = vec4<i32>(0);
    var acc2 = vec4<i32>(0);
    var acc3 = vec4<i32>(0);

    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    {bounds}
    for (var t = tstart; t < tend; t = t + 1u) {
        let k0 = t * KSTEP;
        // --- stage W: 64 rows × 16 of K, 4 values per thread
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0i;
            if (gr < pc.c_out && gk < kdepth) { v = wat(gr * kdepth + gk) - w_zp; }
            w_tile[l] = v;
        }
        // --- stage im2col: 16 of K × 64 pixels, rebuilt from the index
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gk = k0 + l / TILE;
            let gc = col0 + l % TILE;
            var v = 0i;
            if (gk < kdepth && gc < pixels) {
                let ic = gk / ksize;
                let rem = gk % ksize;
                let r = rem / pc.kw;
                let sx = rem % pc.kw;
                let oh = gc / pc.w_out;
                let ow = gc % pc.w_out;
                let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(sx) * i32(pc.dw);
                // out of bounds = the quantized zero, i.e. 0 once folded
                if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                    v = xat(((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)) - a_zp;
                }
            }
            x_tile[l] = v;
        }
        workgroupBarrier();
        // --- 4 scalars of W + 4 of im2col per 16 multiply-adds
        let arow = lid.y * 4u;
        let bcol = lid.x * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let bo = kk * TILE + bcol;
            let bvec = vec4<i32>(x_tile[bo], x_tile[bo + 1u], x_tile[bo + 2u], x_tile[bo + 3u]);
            acc0 = acc0 + vec4<i32>(w_tile[(arow + 0u) * KSTEP + kk]) * bvec;
            acc1 = acc1 + vec4<i32>(w_tile[(arow + 1u) * KSTEP + kk]) * bvec;
            acc2 = acc2 + vec4<i32>(w_tile[(arow + 2u) * KSTEP + kk]) * bvec;
            acc3 = acc3 + vec4<i32>(w_tile[(arow + 3u) * KSTEP + kk]) * bvec;
        }
        workgroupBarrier();
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = row0 + lid.y * 4u + i;
        if (m >= pc.c_out) { continue; }
        var accv = acc0;
        if (i == 1u) { accv = acc1; }
        if (i == 2u) { accv = acc2; }
        if (i == 3u) { accv = acc3; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let p = col0 + lid.x * 4u + j;
            if (p >= pixels) { continue; }
            {store}
        }
    }
}
"#
        .replace("{batch}", batch)
        .replace("{bounds}", bounds)
        .replace("{store}", store)
    )
}

/// Sums the `split` partial images. Exact, and with no bias to apply — see the
/// module header.
pub const SPLIT_REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<i32>;
@group(0) @binding(1) var<storage, read_write> out: array<i32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32,
    x_signed: u32, w_signed: u32, split: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    var acc = 0i;
    for (var s = 0u; s < pc.split; s = s + 1u) {
        acc = acc + partials[s * pc.total + o];
    }
    out[o] = acc;
}
"#;

pub const IM2COL_BINDINGS: u32 = 3;
/// 14 u32 fields, all geometry plus the sign flip.
pub const IM2COL_PUSH_BYTES: u32 = 56;

/// The materialization every other kernel in this module exists to avoid.
///
/// The implicit GEMM rebuilds each column of the im2col matrix from its index
/// and never writes one. Cooperative matrices cannot: `coopMatLoad` takes a
/// base offset and a row stride, so the operand has to exist in memory. This
/// pass writes it — `[P, K]`, one row per output pixel, which is exactly the
/// `[N, K]` layout [`crate::shaders::matmul_integer::PACK_B`] produces, so the
/// multiply that follows is the shared `MatMulInteger` dispatch with
/// `M = C_out`, `N = P`, and `out` landing in NCHW without a transpose.
///
/// Worth it only on the cooperative path: measured on resnet50-int8's own
/// geometries (`examples/conv_integer_coop`), im2col plus the tensor cores is
/// **1.47×** the implicit ladder, while im2col plus the portable WGSL matmul is
/// **0.79×** — materializing loses on its own and wins only for what it unlocks.
///
/// Out-of-bounds taps get the activation zero point, which is what
/// `ConvInteger` pads with, so the padded product `(zp − zp)(w − w_zp)` vanishes
/// without the multiply knowing any geometry. `flip` is `0x80808080` when X is
/// `int8`: the cooperative combination is `u8 × u8` and this pass is already
/// touching every byte, so the sign flip the matmul cannot do costs nothing
/// here.
pub const IM2COL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;    // u8 packed [C_in, H, W]
@group(0) @binding(1) var<storage, read> azp: array<u32>;
@group(0) @binding(2) var<storage, read_write> col: array<u32>;  // u8 packed [P, K]

struct Push {
    words: u32, kdepth: u32, khkw: u32,
    h_in: u32, w_in: u32, w_out: u32, kw: u32,
    sh: u32, sw: u32, phb: u32, pwb: u32, dh: u32, dw: u32,
    flip: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.words) { return; }
    let k4 = pc.kdepth / 4u;
    let p = i / k4;
    let k0 = (i % k4) * 4u;
    let oh = p / pc.w_out;
    let ow = p % pc.w_out;
    let a_zp = azp[0] & 0xffu;
    var word = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let k = k0 + j;
        let c = k / pc.khkw;
        let r = (k % pc.khkw) / pc.kw;
        let s = k % pc.kw;
        let ih = i32(oh * pc.sh + r * pc.dh) - i32(pc.phb);
        let iw = i32(ow * pc.sw + s * pc.dw) - i32(pc.pwb);
        var b = a_zp;
        if (ih >= 0 && iw >= 0 && ih < i32(pc.h_in) && iw < i32(pc.w_in)) {
            let idx = (c * pc.h_in + u32(ih)) * pc.w_in + u32(iw);
            b = (x[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu;
        }
        word = word | (b << (j * 8u));
    }
    col[i] = word ^ pc.flip;
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for source in [
            super::direct(),
            super::implicit_gemm(),
            super::blocked(),
            super::blocked_splitk(),
        ] {
            vk_compute::compile_wgsl(&source).expect("shader ConvInteger valido");
        }
        vk_compute::compile_wgsl(super::SPLIT_REDUCE).expect("shader ConvInteger reduce valido");
        vk_compute::compile_wgsl(super::IM2COL).expect("shader ConvInteger im2col valido");
    }
}
