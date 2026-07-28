//! Shared shaders and dispatch layouts for floating point `Conv`.
//!
//! Two variants with identical bindings and push constants, chosen based on
//! `group`:
//!
//! - `CONV_F32_GEMM` for `group == 1`: implicit tiled GEMM, how GPUs
//!   compute convolutions (each value loaded into shared memory
//!   serves 16 threads instead of one).
//! - `CONV_F32_DIRECT` for grouped/depthwise convolutions, which are not
//!   a single GEMM: each output channel views only its own
//!   group of input channels, so row blocks cannot share loads.
//!   of the same im2col column.

pub const BINDINGS: u32 = 4;
/// 19 u32 fields in the push constant struct. The last is `split`, read only
/// by [`blocked_splitk`]; the single-pass kernels ignore it, so one layout
/// serves every variant.
pub const PUSH_BYTES: u32 = 76;
pub const TILE_SIZE: u32 = 16;
/// Output tile of [`blocked`], which is otherwise a drop-in for
/// [`implicit_gemm`] — same bindings, same push constants, only the grid
/// divisor changes.
pub const BLOCKED_TILE_SIZE: u32 = 64;

/// How many workgroups [`blocked`] would launch for this output.
fn workgroups(pixels: usize, c_out: usize) -> usize {
    pixels.div_ceil(BLOCKED_TILE_SIZE as usize) * c_out.div_ceil(BLOCKED_TILE_SIZE as usize)
}

/// Fewest workgroups at which the 64×64 tile still pays for itself.
///
/// Measured on a 4070 (46 SMs) over the 88 distinct `Conv` geometries of
/// resnet50-qdq, yolov4 and yolov8n (`examples/conv_blocked`): every geometry
/// at or above 24 workgroups was faster blocked (1.20×–4.74×), every geometry
/// below it was slower (down to 0.50×), with a single 1.15× exception at 22.
/// The quantity that separates them is neither `P` nor `K` on its own but how
/// much of the machine the grid can fill — below roughly half the SM count the
/// bigger tile only costs.
///
/// The constant is therefore tied to this GPU's SM count. Vulkan core exposes
/// no portable way to query it (only vendor extensions such as
/// `VK_AMD_shader_core_properties` do), so it stays a measured constant until
/// `VkContext` can derive it from the device.
const WG_FLOOR: usize = 24;

/// Whether [`blocked`] is worth its 64×64 tile on this output geometry.
///
/// Blanket routing is worth 1.04× on resnet50 and 1.41× on yolov8n; gated on
/// this predicate the same kernel is worth 1.33× and 1.57×, and 2.58× on
/// yolov4 — in each case within 0.1% of picking the best kernel per shape by
/// hand. See `examples/conv_blocked`.
pub fn prefer_blocked(pixels: usize, c_out: usize) -> bool {
    workgroups(pixels, c_out) >= WG_FLOOR
}

/// Declarations shared by both variants.
const PRELUDE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32, has_bias: u32,
    split: u32,
}
var<immediate> pc: Push;
"#;

/// Direct 1D/2D Conv (1D normalized to 2D with W=1): one thread per output element
/// output, with group/stride/pad/dilation.
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
    var acc = 0.0;
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
                acc = acc + x[xidx] * w[widx];
            }
        }
    }
    if (pc.has_bias != 0u) { acc = acc + bias[m]; }
    out[o] = acc;
}
"#
    )
}

/// Implicit GEMM for `group == 1`:
/// `out[C_out, P] = W[C_out, C_in·KH·KW] × X_im2col[C_in·KH·KW, P]`, with
/// `P = H_out·W_out`. The im2col matrix is never materialized: its
/// columns are indexed on the fly into `x`, so the read bandwidth remains
/// that of the original tensor.
pub fn implicit_gemm() -> String {
    format!(
        "{PRELUDE}{}",
        r#"
const TILE = 16u;
var<workgroup> w_tile: array<f32, 256>;
var<workgroup> x_tile: array<f32, 256>;

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

    var acc = 0.0;
    let ntiles = (kdepth + TILE - 1u) / TILE;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let kw_idx = t * TILE + tx;
        if (m < pc.c_out && kw_idx < kdepth) {
            w_tile[ty * TILE + tx] = w[m * kdepth + kw_idx];
        } else {
            w_tile[ty * TILE + tx] = 0.0;
        }
        let kx_idx = t * TILE + ty;
        var value = 0.0;
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
            // out of bounds = zero: this is the conv's implicit padding
            if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                value = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
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
    if (pc.has_bias != 0u) { acc = acc + bias[m]; }
    out[(bn * pc.c_out + m) * pixels + p] = acc;
}
"#
    )
}

/// The same implicit GEMM on a 64×64 output tile with a 4×4 micro-tile held in
/// registers, for the geometries [`prefer_blocked`] accepts.
///
/// [`implicit_gemm`] keeps one output per thread, so its inner loop spends two
/// shared reads per FMA; here 8 reads feed 16 FMAs. That is the transformation
/// `MatMul` and `Gemm` already took, and on `Conv` it is worth anything from
/// 0.50× to 4.74× depending purely on how many workgroups the grid launches —
/// hence the predicate. `W` is read straight as the GEMM's `A`; the `B` side is
/// the im2col matrix, whose staging tile rebuilds each column from its `K`
/// index exactly as the 16×16 kernel does, so nothing is materialized here
/// either.
///
/// Accumulation order is identical to [`implicit_gemm`] — both walk `K` in
/// steps of 16 — so the two kernels agree bit for bit on all 88 geometries
/// measured, and routing between them cannot move a model's output.
pub fn blocked() -> String {
    blocked_body(false)
}

/// [`blocked`] with its `K` loop sliced across `wid.z`, writing one partial
/// image per slice for [`SPLIT_REDUCE`] to sum.
///
/// `wid.z` carries both the batch image and the slice — `bn = z / split`,
/// `slice = z % split` — since the grid has only three dimensions and the batch
/// already owned this one. The bias is deliberately not applied here: adding it
/// per slice would multiply it by `split`. It belongs to the reduction, the only
/// pass that sees a whole sum.
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
            "var v = accv[j];\n\
             \x20           if (pc.has_bias != 0u) { v = v + bias[m]; }\n\
             \x20           out[(bn * pc.c_out + m) * pixels + p] = v;",
        )
    };
    format!(
        "{PRELUDE}{}",
        r#"
const TILE = 64u;
const KSTEP = 16u;
var<workgroup> w_tile: array<f32, 1024>;
var<workgroup> x_tile: array<f32, 1024>;

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

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (kdepth + KSTEP - 1u) / KSTEP;
    {bounds}
    for (var t = tstart; t < tend; t = t + 1u) {
        let k0 = t * KSTEP;
        // --- stage W: 64 rows × 16 of K, 4 values per thread
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gr = row0 + l / KSTEP;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gr < pc.c_out && gk < kdepth) { v = w[gr * kdepth + gk]; }
            w_tile[l] = v;
        }
        // --- stage im2col: 16 of K × 64 pixels, rebuilt from the index
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let gk = k0 + l / TILE;
            let gc = col0 + l % TILE;
            var v = 0.0;
            if (gk < kdepth && gc < pixels) {
                let ic = gk / ksize;
                let rem = gk % ksize;
                let r = rem / pc.kw;
                let sx = rem % pc.kw;
                let oh = gc / pc.w_out;
                let ow = gc % pc.w_out;
                let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(sx) * i32(pc.dw);
                // out of bounds = zero: the conv's implicit padding
                if (ih >= 0 && ih < i32(pc.h_in) && iw >= 0 && iw < i32(pc.w_in)) {
                    v = x[((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw)];
                }
            }
            x_tile[l] = v;
        }
        workgroupBarrier();
        // --- 4 scalars of W + 4 of im2col per 16 FMAs
        let arow = lid.y * 4u;
        let bcol = lid.x * 4u;
        for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
            let bo = kk * TILE + bcol;
            let bvec = vec4<f32>(x_tile[bo], x_tile[bo + 1u], x_tile[bo + 2u], x_tile[bo + 3u]);
            acc0 = fma(vec4<f32>(w_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0);
            acc1 = fma(vec4<f32>(w_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1);
            acc2 = fma(vec4<f32>(w_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2);
            acc3 = fma(vec4<f32>(w_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3);
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

// ------------------------------------------------------------- split-K

pub const SPLIT_REDUCE_BINDINGS: u32 = 3;
/// Workgroups the split is sized to launch, on the 64×64 tile.
const SPLIT_TARGET_WGS: usize = 128;
/// Ceiling on the split, bounding the partials buffer at `32 ×` the output.
const SPLIT_MAX: usize = 32;
/// Smallest `K` worth splitting at all, and the smallest slice a split may
/// leave a workgroup — both one `KSTEP` block per staging round, four deep.
const SPLIT_MIN_K: usize = 256;
const SPLIT_MIN_K_PER_SLICE: usize = 64;

/// How many ways to split `K = C_in·KH·KW`, or `None` to dispatch one pass.
///
/// [`prefer_blocked`] answers a different question than this one, and both are
/// needed. `docs/resnet50-gap.md` measured that the 64×64 tile is worth ~1.9× on
/// resnet50's small-output convolutions **once the machine is full** and ~1.0×
/// at batch 1, and concluded no tile change reaches them. That was right about
/// the tile and wrong about the conclusion: the wide tile buys arithmetic
/// intensity by spending grid, and split-K is what buys the grid back. Neither
/// works alone — measured on those geometries, split-K on the 16×16 kernel is
/// **1.18×**, the 64×64 tile alone is ~1.0×, and together they are **3.8×**.
///
/// So a `Some` here means *both*: the 64×64 tile and this many slices, whatever
/// [`prefer_blocked`] would have said. `K` is the discriminator, not the
/// workgroup count — the geometries where splitting never pays
/// (`3→64 7×7`, `64→256 1×1`, `64→64 1×1`, `128→512 1×1`) are exactly those
/// with `K ≤ 147`, which have nothing to split.
///
/// Sized to land near `SPLIT_TARGET_WGS` workgroups, as
/// `matmul_fp32::gemv_split` does, and calibrated the same way: on this GPU's
/// 46 SMs, with `examples/conv_splitk`. Measured 1.3× to 4.7× per geometry and
/// **2.45× over all 53 `Conv` nodes** of resnet50-qdq.
pub fn split_k(pixels: usize, c_out: usize, kdepth: usize) -> Option<u32> {
    if kdepth < SPLIT_MIN_K {
        return None;
    }
    let base = workgroups(pixels, c_out);
    let split = SPLIT_TARGET_WGS
        .div_ceil(base.max(1))
        .min(SPLIT_MAX)
        .min(kdepth / SPLIT_MIN_K_PER_SLICE);
    (split > 1).then_some(split as u32)
}

/// Sums the `split` partial images and applies the bias.
pub const SPLIT_REDUCE: &str = r#"
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gsi: u32, has_bias: u32,
    split: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    var acc = 0.0;
    for (var s = 0u; s < pc.split; s = s + 1u) {
        acc = acc + partials[s * pc.total + o];
    }
    if (pc.has_bias != 0u) {
        // o indexes [N, C_out, P]: the channel is what selects the bias
        acc = acc + bias[(o / (pc.h_out * pc.w_out)) % pc.c_out];
    }
    out[o] = acc;
}
"#;

#[cfg(test)]
mod tests {
    use super::prefer_blocked;

    #[test]
    fn sources_compile() {
        for source in [super::direct(), super::implicit_gemm(), super::blocked()] {
            vk_compute::compile_wgsl(&source).expect("shader Conv f32 valido");
        }
    }

    /// The measured split from `examples/conv_blocked`: the geometries below
    /// the floor were 0.50×–0.83× on a 4070, the ones at or above it
    /// 1.20×–4.74×. These are real nodes of resnet50-qdq, yolov4 and yolov8n.
    #[test]
    fn the_predicate_matches_what_was_measured() {
        // rejected: too few workgroups to fill the machine
        assert!(!prefer_blocked(49, 512)); //  8 wg — resnet50 512→512 3x3 @7²
        assert!(!prefer_blocked(196, 256)); // 16 wg — resnet50 256→256 3x3 @14²
        assert!(!prefer_blocked(400, 64)); //  7 wg — yolov8n 64→64 3x3 @20²
        assert!(!prefer_blocked(169, 255)); // 12 wg — yolov4 1024→255 1x1 @13²
        assert!(!prefer_blocked(16, 1)); //  1 wg — yolov8n 16→1 1x1 @4²

        // accepted, starting exactly at the floor
        assert!(prefer_blocked(169, 512)); //  24 wg — yolov4 512→512 3x3 @13²
        assert!(prefer_blocked(784, 128)); //  26 wg — resnet50 128→128 3x3 @28²
        assert!(prefer_blocked(49, 2048)); //  32 wg — resnet50 512→2048 1x1 @7²
        assert!(prefer_blocked(43264, 64)); // 676 wg — yolov4 64→64 1x1 @208²
        assert!(prefer_blocked(173056, 32)); // 2704 wg — yolov4 3→32 3x3 @416²
    }

    /// Small `P` alone does not reject: enough output channels make the grid
    /// wide even on a 7×7 feature map, and those geometries did gain (1.57×).
    #[test]
    fn channels_can_carry_a_tiny_feature_map() {
        assert!(!prefer_blocked(49, 512));
        assert!(prefer_blocked(49, 2048));
    }
}
