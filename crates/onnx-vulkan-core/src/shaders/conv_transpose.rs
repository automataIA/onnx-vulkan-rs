//! Shared shader and dispatch layout for floating point `ConvTranspose`.
//!
//! One thread per **output** element, gathering. The scatter form — walk the
//! input, spray each value over its kernel footprint — is the direct reading of
//! the operator, but it makes several threads write the same output element and
//! would need atomics on f32. Inverting the index relation removes that:
//!
//! ```text
//!   forward:  oh = ih*stride - pad + r*dilation
//!   inverted: ih = (oh + pad - r*dilation) / stride,  exact division only
//! ```
//!
//! so a thread owning `oh` walks the kernel rows `r`, keeps the ones where
//! `oh + pad - r*dilation` is a non-negative multiple of the stride, and reads
//! the single input row that contributed. Every output element is written once.
//!
//! The weight layout is the other thing that differs from `Conv`: here `W` is
//! `[C_in, C_out/group, kH, kW]` — indexed by the **input** channel first. That
//! is why this cannot reuse `conv::direct` with a flipped geometry.
//!
//! ## The phase-GEMM route
//!
//! That kernel reads each input element `C_out` times from global memory and
//! keeps nothing in shared memory, which measured 0.333 TFLOP/s on an RTX 4070
//! — 1.1% of the card. When the footprints never overlap ([`phase_gemm_applies`])
//! the operator is instead `kH·kW` independent GEMMs
//!
//! ```text
//!   phase[r][s][m][p] = Σ_ic W[ic, m, r, s] · X[ic, p]      p over input pixels
//! ```
//!
//! of M = C_out, K = C_in, N = H_in·W_in, each followed by an [`INTERLEAVE`]
//! that scatters its result to `out[m, ih*sh + r, iw*sw + s]`. Routed through
//! the blocked `gemm::GEMM`, sam3 ViT-H's three nodes go from 200.1 ms to
//! 12.0 ms. The two paths were measured against each other first, and agree
//! **bit for bit** on both real shapes (`examples/convtranspose_gemm.rs`).
//!
//! ## The polyphase route (1D)
//!
//! When the footprints overlap (`kernel > stride`) or pads crop the output, an
//! output sums several kernel taps and the phases are no longer independent.
//! Grouping the outputs by phase `φ = (oh + pad) mod stride` still gives one
//! GEMM per phase, with the taps of that phase folded into K
//! ([`POLYPHASE_1D`]). Piper's four HiFi-GAN upsamplers (k16/s8/p4, k4/s2/p1)
//! go from 67.6 ms to 2.4 ms in isolation, and agree with [`direct`] **bit for
//! bit**, because each output keeps a single accumulator walked in the same
//! order (`examples/convtranspose_polyphase.rs`).
//!
//! Computing every tap with one GEMM over `M = C_out·kH·kW` and summing them
//! afterwards (col2im) was measured as well: as fast, but it rounds each tap's
//! `C_in` sum separately and needs a transient buffer twice the output.

use super::gemm::{K_STEP, TILE_SIZE};

pub const BINDINGS: u32 = 4;
/// 18 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 72;

/// Direct 1D/2D ConvTranspose (1D normalized to 2D with W=1).
pub fn direct() -> String {
    r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, c_in: u32, c_out: u32, group: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, gso: u32, has_bias: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= pc.total) { return; }
    let ow = o % pc.w_out;
    let t1 = o / pc.w_out;
    let oh = t1 % pc.h_out;
    let t2 = t1 / pc.h_out;
    let m = t2 % pc.c_out;                // output channel
    let bn = t2 / pc.c_out;
    let g = m / pc.gso;                   // group this channel belongs to
    let mg = m % pc.gso;                  // channel within the group
    let gsi = pc.c_in / pc.group;         // input channels per group

    var acc = 0.0;
    for (var r = 0u; r < pc.kh; r = r + 1u) {
        // ih*sh = oh + phb - r*dh, with exact division: if the remainder is
        // non-zero, no input lands on this output for this kernel row.
        let nh = i32(oh) + i32(pc.phb) - i32(r) * i32(pc.dh);
        if (nh < 0 || nh % i32(pc.sh) != 0) { continue; }
        let ih = nh / i32(pc.sh);
        if (ih >= i32(pc.h_in)) { continue; }
        for (var s = 0u; s < pc.kw; s = s + 1u) {
            let nw = i32(ow) + i32(pc.pwb) - i32(s) * i32(pc.dw);
            if (nw < 0 || nw % i32(pc.sw) != 0) { continue; }
            let iw = nw / i32(pc.sw);
            if (iw >= i32(pc.w_in)) { continue; }
            for (var cg = 0u; cg < gsi; cg = cg + 1u) {
                let ic = g * gsi + cg;
                let xidx = ((bn * pc.c_in + ic) * pc.h_in + u32(ih)) * pc.w_in + u32(iw);
                // W is [C_in, C_out/group, kH, kW]: input channel first
                let widx = ((ic * pc.gso + mg) * pc.kh + r) * pc.kw + s;
                acc = acc + x[xidx] * w[widx];
            }
        }
    }
    if (pc.has_bias != 0u) { acc = acc + bias[m]; }
    out[o] = acc;
}
"#
    .to_string()
}

/// Workgroup size of [`direct`]: one invocation per output element.
pub const DIRECT_WORKGROUP: u32 = 256;

const U32_MAX: usize = u32::MAX as usize;
const I32_MAX: usize = i32::MAX as usize;

/// Every dimension [`direct`] reads, of a node that passed the operator's
/// validation (kernel, strides, dilations and group at least 1). Pairs are
/// `(H, W)`, `W = 1` for 1D.
#[derive(Clone, Copy, Debug)]
pub struct DirectDims {
    pub batch: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub group: usize,
    pub input: (usize, usize),
    pub output: (usize, usize),
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    /// Leading pads.
    pub pad: (usize, usize),
    pub dilation: (usize, usize),
}

impl DirectDims {
    /// Output elements, one invocation each.
    pub fn total(&self) -> Option<usize> {
        self.batch
            .checked_mul(self.c_out)?
            .checked_mul(self.output.0)?
            .checked_mul(self.output.1)
    }
}

/// Whether one axis of [`direct`]'s `i32` arithmetic keeps its sign: the
/// largest `oh + pad`, the largest `r·dilation` subtracted from it, and the
/// stride and input length it is divided by and compared with.
fn direct_axis_fits(
    output: usize,
    pad: usize,
    k: usize,
    dil: usize,
    stride: usize,
    input: usize,
) -> bool {
    let fit = || -> Option<bool> {
        Some(
            output.checked_add(pad)?.checked_sub(1)? <= I32_MAX
                && k.checked_sub(1)?.checked_mul(dil)? <= I32_MAX
                && stride <= I32_MAX
                && input <= I32_MAX,
        )
    };
    fit() == Some(true)
}

/// Whether [`direct`] addresses `d` exactly: every push field, the `X`, `W`
/// and output lengths fit `u32` (so do the flat indices below them), and the
/// signed row and column arithmetic does not wrap.
pub fn direct_indices_fit(d: &DirectDims) -> bool {
    let fit = || -> Option<bool> {
        let gso = d.c_out.checked_div(d.group)?;
        let x_len = d
            .batch
            .checked_mul(d.c_in)?
            .checked_mul(d.input.0)?
            .checked_mul(d.input.1)?;
        let w_len = d
            .c_in
            .checked_mul(gso)?
            .checked_mul(d.kernel.0)?
            .checked_mul(d.kernel.1)?;
        let pairs = [d.input, d.output, d.kernel, d.stride, d.pad, d.dilation];
        Some(
            [d.total()?, x_len, w_len, d.c_in, d.c_out, d.group]
                .iter()
                .all(|&v| v <= U32_MAX)
                && pairs.iter().all(|&(h, w)| h <= U32_MAX && w <= U32_MAX)
                && direct_axis_fits(
                    d.output.0,
                    d.pad.0,
                    d.kernel.0,
                    d.dilation.0,
                    d.stride.0,
                    d.input.0,
                )
                && direct_axis_fits(
                    d.output.1,
                    d.pad.1,
                    d.kernel.1,
                    d.dilation.1,
                    d.stride.1,
                    d.input.1,
                ),
        )
    };
    fit() == Some(true)
}

/// Whether the device launches [`direct`]'s one-dimensional grid for `d`.
pub fn direct_fits_device(d: &DirectDims, max_workgroup_count: [u32; 3]) -> bool {
    d.total()
        .is_some_and(|t| t.div_ceil(DIRECT_WORKGROUP as usize) <= max_workgroup_count[0] as usize)
}

/// Geometry of one `ConvTranspose` node, enough to decide the phase-GEMM route.
#[derive(Clone, Copy, Debug)]
pub struct PhaseGeom {
    pub batch: i64,
    pub group: i64,
    /// (kH, kW), (strideH, strideW), (dilationH, dilationW).
    pub kernel: (i64, i64),
    pub stride: (i64, i64),
    pub dilation: (i64, i64),
    pub zero_pads: bool,
    pub zero_output_padding: bool,
    /// The node is 1D (normalized to `W = 1`).
    pub one_d: bool,
    /// `C_in · C_out · H_out · W_out`, the multiply-accumulates the node costs.
    pub macs: i64,
}

/// Below this much work the gather kernel wins: the phase route costs
/// `2·kH·kW` dispatches instead of one, and a 64×64 GEMM tile computed for a
/// handful of output channels is mostly padding. The gather kernel runs at
/// ~0.33 TFLOP/s measured, so 8M MACs is about 50 µs of it — the point where
/// the extra dispatches stop mattering.
pub const PHASE_GEMM_MIN_MACS: i64 = 1 << 23;

/// Whether the node decomposes into `kH·kW` independent GEMMs.
///
/// The decomposition needs every output pixel to be fed by exactly one kernel
/// offset, which holds when the footprints never overlap — stride at least the
/// kernel size, no padding, no dilation. `output_padding` would append output
/// rows no phase produces, and `group > 1` and `batch > 1` would each need the
/// GEMM to walk a non-contiguous slice of a buffer it addresses from zero.
/// Everything else falls back to [`direct`].
pub fn phase_gemm_applies(g: &PhaseGeom) -> bool {
    g.batch == 1
        && g.group == 1
        && g.zero_pads
        && g.zero_output_padding
        && g.dilation == (1, 1)
        && g.stride.0 >= g.kernel.0
        && g.stride.1 >= g.kernel.1
        && g.macs >= PHASE_GEMM_MIN_MACS
}

/// Which kernel sequence a `ConvTranspose` node runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// `kH·kW` GEMMs + interleave ([`phase_gemm_applies`], constant `W`).
    Phase,
    /// One GEMM per output phase, [`POLYPHASE_1D`].
    Polyphase,
    /// [`direct`].
    Gather,
}

/// The route of a node, `w_constant` telling whether `W` is an initializer.
///
/// The phase route keeps priority where it applies: it was measured on sam3.
/// The polyphase route covers the 1D overlapping and padded geometries it
/// rejects, with the same batch, group and work limits; 2D nodes, dilation and
/// `output_padding` stay on the gather kernel, which is the only route tested
/// on them.
pub fn route(g: &PhaseGeom, w_constant: bool) -> Route {
    if phase_gemm_applies(g) && w_constant {
        Route::Phase
    } else if g.one_d
        && g.batch == 1
        && g.group == 1
        && g.zero_output_padding
        && g.dilation == (1, 1)
        && g.macs >= PHASE_GEMM_MIN_MACS
    {
        Route::Polyphase
    } else {
        Route::Gather
    }
}

pub const POLYPHASE_BINDINGS: u32 = 4;
/// 9 u32 fields in the push constant struct.
pub const POLYPHASE_PUSH_BYTES: u32 = 36;

/// 1D `ConvTranspose` as one implicit GEMM per output phase, `wid.z = φ`.
///
/// Output `oh` with `oh + pad = q·stride + φ` is fed by the taps
/// `r = φ + j·stride` at input `q − j`. Per phase this is a GEMM of
/// M = C_out, N = q, K = (j, C_in) with `j` major:
///
/// ```text
///   out[m, q·s + φ − pad] = Σ_j Σ_ic W[ic, m, φ + j·s] · X[ic, q − j]  + bias[m]
/// ```
///
/// Neither operand is materialized: A is read from `W` as stored and B from
/// `X` shifted by the tap, zero outside the input, and the epilogue writes
/// straight to the output, dropping the `q` that land in the pads. Each output
/// is one accumulator walked in the order of [`direct`] — taps ascending, then
/// input channels — so the two may agree bit for bit. The tile loop is
/// `gemm::GEMM`'s (64×64, 4×4 per thread); see there.
pub const POLYPHASE_1D: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
struct Push {
    c_in: u32, c_out: u32, l_in: u32, l_out: u32,
    k: u32, s: u32, pad: u32, nq: u32, has_bias: u32,
}
var<immediate> pc: Push;

const TILE = 64u;
const KSTEP = 16u;
var<workgroup> as_tile: array<f32, 1024>;
var<workgroup> bs_tile: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let phase = wid.z;
    let row0 = wid.y * TILE;          // first output channel of the tile
    let col0 = wid.x * TILE;          // first q of the tile
    let tid = lid.y * 16u + lid.x;
    // r = phase + j*s < k; none when stride > kernel leaves this phase empty
    let taps = select(0u, (pc.k - phase + pc.s - 1u) / pc.s, phase < pc.k);
    let kdim = taps * pc.c_in;
    // Some column of the tile reads outside the input for some tap (q < j or
    // q - j >= l_in). Uniform per workgroup: interior tiles keep the plain loop.
    let border = col0 + 1u < taps || col0 + TILE > pc.l_in;
    let q4 = vec4<u32>(col0 + lid.x * 4u) + vec4<u32>(0u, 1u, 2u, 3u);

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (kdim + KSTEP - 1u) / KSTEP;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let k0 = t * KSTEP;

        // --- stage A: W[ic, m, phase + j*s] for (m, (j, ic))
        for (var s = 0u; s < 4u; s = s + 1u) {
            let l = tid + s * 256u;
            let r = l / KSTEP;
            let gm = row0 + r;
            let gk = k0 + l % KSTEP;
            var v = 0.0;
            if (gm < pc.c_out && gk < kdim) {
                let j = gk / pc.c_in;
                let ic = gk % pc.c_in;
                v = w[(ic * pc.c_out + gm) * pc.k + phase + j * pc.s];
            }
            as_tile[l] = v;
        }

        // --- stage B: X[ic, q - j], zero outside the input
        let bk = tid / 16u;
        let bc = (tid % 16u) * 4u;
        let gk = k0 + bk;
        var bv = vec4<f32>(0.0);
        if (gk < kdim) {
            let j = gk / pc.c_in;
            let ic = gk % pc.c_in;
            for (var e = 0u; e < 4u; e = e + 1u) {
                let q = col0 + bc + e;
                if (q < pc.nq && q >= j && q - j < pc.l_in) {
                    bv[e] = x[ic * pc.l_in + q - j];
                }
            }
        }
        bs_tile[tid] = bv;
        workgroupBarrier();

        let arow = lid.y * 4u;
        if (border) {
            for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
                let bvec = bs_tile[kk * 16u + lid.x];
                // A tap outside the input is skipped, as in `direct`: its zero
                // B times an infinite or NaN weight would give NaN.
                let j4 = vec4<u32>((k0 + kk) / pc.c_in);
                let skip = (q4 < j4) | (q4 - j4 >= vec4<u32>(pc.l_in));
                acc0 = select(fma(vec4<f32>(as_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0), acc0, skip);
                acc1 = select(fma(vec4<f32>(as_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1), acc1, skip);
                acc2 = select(fma(vec4<f32>(as_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2), acc2, skip);
                acc3 = select(fma(vec4<f32>(as_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3), acc3, skip);
            }
        } else {
            for (var kk = 0u; kk < KSTEP; kk = kk + 1u) {
                let bvec = bs_tile[kk * 16u + lid.x];
                acc0 = fma(vec4<f32>(as_tile[(arow + 0u) * KSTEP + kk]), bvec, acc0);
                acc1 = fma(vec4<f32>(as_tile[(arow + 1u) * KSTEP + kk]), bvec, acc1);
                acc2 = fma(vec4<f32>(as_tile[(arow + 2u) * KSTEP + kk]), bvec, acc2);
                acc3 = fma(vec4<f32>(as_tile[(arow + 3u) * KSTEP + kk]), bvec, acc3);
            }
        }
        workgroupBarrier();
    }

    // --- epilogue: out[m, q*s + phase - pad], bias added last like `direct`
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = row0 + lid.y * 4u + i;
        if (m >= pc.c_out) { continue; }
        var accv = acc0;
        if (i == 1u) { accv = acc1; }
        if (i == 2u) { accv = acc2; }
        if (i == 3u) { accv = acc3; }
        for (var e = 0u; e < 4u; e = e + 1u) {
            let q = col0 + lid.x * 4u + e;
            let oh = i32(q * pc.s + phase) - i32(pc.pad);
            if (q >= pc.nq || oh < 0 || oh >= i32(pc.l_out)) { continue; }
            var value = accv[e];
            if (pc.has_bias != 0u) { value = value + bias[m]; }
            out[m * pc.l_out + u32(oh)] = value;
        }
    }
}
"#;

/// Every dimension [`POLYPHASE_1D`] reads, of a validated 1D node.
#[derive(Clone, Copy, Debug)]
pub struct PolyphaseDims {
    pub c_in: usize,
    pub c_out: usize,
    pub l_in: usize,
    pub l_out: usize,
    pub kernel: usize,
    pub stride: usize,
    /// Leading pad.
    pub pad: usize,
}

impl PolyphaseDims {
    /// Columns `q` of each phase GEMM: the last output sits at
    /// `q·stride + φ = l_out - 1 + pad`. `None` without a stride or an output.
    pub fn nq(&self) -> Option<usize> {
        Some(
            self.l_out
                .checked_sub(1)?
                .checked_add(self.pad)?
                .checked_div(self.stride)?
                + 1,
        )
    }

    /// Workgroups `[q tiles, C_out tiles, phases]`.
    pub fn grid(&self) -> Option<[usize; 3]> {
        let tile = TILE_SIZE as usize;
        Some([
            self.nq()?.div_ceil(tile),
            self.c_out.div_ceil(tile),
            self.stride,
        ])
    }
}

/// Whether [`POLYPHASE_1D`] addresses `d` exactly, over the whole grid
/// (columns and rows rounded up to the tile):
///
/// - the push fields fit `u32`, and `l_out` and `pad` fit `i32`;
/// - `taps`, through `k − φ + s − 1`, and the largest `K` (phase 0's) rounded
///   up to `KSTEP` fit `u32`;
/// - the `W`, `X` and output lengths fit `u32`, so do the flat indices below
///   them (`X`'s `ic·l_in + q − j` may wrap before the `− j`, and comes back
///   exact: `u32` arithmetic is modular);
/// - the epilogue row `q·s + φ` of the last grid column fits `i32`, so its
///   cast keeps the sign and `− pad` and the bounds test are exact.
///
/// Under these, every output `oh` is written once, by `q = (oh + pad) / s`
/// (`< nq`) in phase `(oh + pad) mod s`, provided the device launches the
/// grid ([`polyphase_fits_device`]).
pub fn polyphase_indices_fit(d: &PolyphaseDims) -> bool {
    let fit = || -> Option<bool> {
        let tile = TILE_SIZE as usize;
        let cols = d.nq()?.div_ceil(tile).checked_mul(tile)?;
        let rows = d.c_out.div_ceil(tile).checked_mul(tile)?;
        let taps = d.kernel.checked_add(d.stride - 1)?;
        let kdim = d
            .kernel
            .div_ceil(d.stride)
            .checked_mul(d.c_in)?
            .checked_add(K_STEP as usize - 1)?;
        let w_len = d.c_in.checked_mul(d.c_out)?.checked_mul(d.kernel)?;
        let x_len = d.c_in.checked_mul(d.l_in)?;
        let out_len = d.c_out.checked_mul(d.l_out)?;
        let last_row = cols.checked_mul(d.stride)? - 1;
        Some(
            [d.c_in, d.l_in, taps, kdim, rows, w_len, x_len, out_len]
                .iter()
                .all(|&v| v <= U32_MAX)
                && [d.l_out, d.pad, last_row].iter().all(|&v| v <= I32_MAX),
        )
    };
    fit() == Some(true)
}

/// Whether the device launches [`POLYPHASE_1D`]'s grid for `d`: the phases
/// ride on `z`, so the stride is bounded by `maxComputeWorkGroupCount[2]`.
pub fn polyphase_fits_device(d: &PolyphaseDims, max_workgroup_count: [u32; 3]) -> bool {
    d.grid().is_some_and(|grid| {
        grid.iter()
            .zip(max_workgroup_count)
            .all(|(&n, max)| n <= max as usize)
    })
}

pub const PACK_BINDINGS: u32 = 2;
/// 3 u32 fields in the push constant struct.
pub const PACK_PUSH_BYTES: u32 = 12;

/// One phase of `W`, `[C_in, C_out, kH, kW]` → `[K = C_in][M = C_out]`.
///
/// That is the layout the GEMM reads with `trans_a`, and since `group == 1` the
/// two leading axes are exactly `(ic, m)` in row-major order: element `i` of the
/// slice is `w[i * kH*kW + r*kW + s]`.
pub const PACK_PHASE: &str = r#"
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct Push { total: u32, span: u32, off: u32 }   // span = kH*kW, off = r*kW + s
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.total) { return; }
    out[i] = w[i * pc.span + pc.off];
}
"#;

pub const INTERLEAVE_BINDINGS: u32 = 2;
/// 9 u32 fields in the push constant struct.
pub const INTERLEAVE_PUSH_BYTES: u32 = 36;

/// Writes a phase's `[C_out, H_in, W_in]` result to its strided home,
/// `out[m, ih*sh + r, iw*sw + s]`.
pub const INTERLEAVE: &str = r#"
@group(0) @binding(0) var<storage, read> phase: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct Push {
    total: u32, h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    sh: u32, sw: u32, r: u32, s: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.total) { return; }
    let iw = i % pc.w_in;
    let t = i / pc.w_in;
    let ih = t % pc.h_in;
    let ch = t / pc.h_in;
    let oh = ih * pc.sh + pc.r;
    let ow = iw * pc.sw + pc.s;
    if (oh >= pc.h_out || ow >= pc.w_out) { return; }
    out[(ch * pc.h_out + oh) * pc.w_out + ow] = phase[i];
}
"#;

pub const FILL_BINDINGS: u32 = 2;
/// 3 u32 fields in the push constant struct.
pub const FILL_PUSH_BYTES: u32 = 12;

/// Bias (or zero) over the whole output, run before the phases.
///
/// Only needed when the stride is strictly larger than the kernel: then the
/// footprints leave holes that no phase writes, and those pixels are just the
/// bias. When stride equals kernel the phases tile the output exactly and this
/// pass is skipped.
pub const FILL: &str = r#"
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
struct Push { total: u32, plane: u32, has_bias: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.total) { return; }
    var v = 0.0;
    if (pc.has_bias != 0u) { v = bias[i / pc.plane]; }
    out[i] = v;
}
"#;

#[cfg(test)]
mod tests {
    use super::{PhaseGeom, phase_gemm_applies};

    #[test]
    fn source_compiles() {
        vk_compute::compile_wgsl(&super::direct()).expect("valid ConvTranspose f32 shader");
        vk_compute::compile_wgsl(super::PACK_PHASE).expect("valid pack phase shader");
        vk_compute::compile_wgsl(super::INTERLEAVE).expect("valid interleave shader");
        vk_compute::compile_wgsl(super::FILL).expect("valid fill shader");
        vk_compute::compile_wgsl(super::POLYPHASE_1D).expect("valid polyphase shader");
    }

    /// Piper's upsamplers (k16/s8/p4, k4/s2/p1) overlap and are padded: they
    /// must leave the gather kernel for the polyphase route, and the
    /// geometries it is not tested on must not follow them there.
    #[test]
    fn route_sends_1d_overlap_and_pads_to_polyphase_only() {
        use super::{Route, route};
        let piper = PhaseGeom {
            kernel: (16, 1),
            stride: (8, 1),
            zero_pads: false,
            one_d: true,
            ..sam3()
        };
        assert_eq!(route(&sam3(), true), Route::Phase);
        assert_eq!(route(&sam3(), false), Route::Gather);
        assert_eq!(route(&piper, true), Route::Polyphase);
        assert_eq!(route(&piper, false), Route::Polyphase);
        let gather = |why: &str, tweak: fn(&mut PhaseGeom)| {
            let mut g = piper;
            tweak(&mut g);
            assert_eq!(route(&g, true), Route::Gather, "without {why}: {g:?}");
        };
        gather("1D", |g| g.one_d = false);
        gather("batch 1", |g| g.batch = 2);
        gather("group 1", |g| g.group = 2);
        gather("null output_padding", |g| g.zero_output_padding = false);
        gather("dilation 1", |g| g.dilation = (1, 2));
        gather("enough work", |g| g.macs = super::PHASE_GEMM_MIN_MACS - 1);
    }

    /// sam3's shape: 2×2 kernel, stride 2, nothing else set.
    fn sam3() -> PhaseGeom {
        PhaseGeom {
            batch: 1,
            group: 1,
            kernel: (2, 2),
            stride: (2, 2),
            dilation: (1, 1),
            zero_pads: true,
            zero_output_padding: true,
            one_d: false,
            macs: 1 << 30,
        }
    }

    /// Each condition on its own must send the node back to the gather kernel:
    /// none of them is implied by the others, and a route taken on a geometry
    /// that does not satisfy them is silently wrong, not slow.
    #[test]
    fn every_precondition_is_load_bearing() {
        assert!(phase_gemm_applies(&sam3()));
        let reject = |why: &str, tweak: fn(&mut PhaseGeom)| {
            let mut g = sam3();
            tweak(&mut g);
            assert!(!phase_gemm_applies(&g), "accepted without {why}: {g:?}");
        };
        reject("batch 1", |g| g.batch = 2);
        reject("group 1", |g| g.group = 2);
        reject("null pads", |g| g.zero_pads = false);
        reject("null output_padding", |g| g.zero_output_padding = false);
        reject("dilation h 1", |g| g.dilation = (2, 1));
        reject("dilation w 1", |g| g.dilation = (1, 2));
        reject("stride h >= kernel h", |g| g.stride = (1, 2));
        reject("stride w >= kernel w", |g| g.stride = (2, 1));
        reject("enough work", |g| g.macs = super::PHASE_GEMM_MIN_MACS - 1);
    }

    /// Stride strictly greater than the kernel is still a valid decomposition —
    /// the footprints only get further apart — and it is the case that needs
    /// [`FILL`], so it must not be rejected by accident.
    #[test]
    fn stride_larger_than_kernel_still_applies() {
        assert!(phase_gemm_applies(&PhaseGeom {
            kernel: (2, 2),
            stride: (3, 4),
            ..sam3()
        }));
    }

    use super::{
        DIRECT_WORKGROUP, DirectDims, I32_MAX, K_STEP, POLYPHASE_1D, PolyphaseDims, TILE_SIZE,
        U32_MAX, direct_fits_device, direct_indices_fit, polyphase_fits_device,
        polyphase_indices_fit,
    };

    /// The checks bound the shaders' tile, K step and workgroup through the
    /// Rust constants: they must be the ones compiled in.
    #[test]
    fn checks_use_the_shaders_constants() {
        assert!(POLYPHASE_1D.contains(&format!("const TILE = {TILE_SIZE}u;")));
        assert!(POLYPHASE_1D.contains(&format!("const KSTEP = {K_STEP}u;")));
        assert!(super::direct().contains(&format!("@workgroup_size({DIRECT_WORKGROUP})")));
    }

    fn poly(
        c_in: usize,
        c_out: usize,
        l_in: usize,
        l_out: usize,
        k: usize,
        s: usize,
        pad: usize,
    ) -> PolyphaseDims {
        PolyphaseDims {
            c_in,
            c_out,
            l_in,
            l_out,
            kernel: k,
            stride: s,
            pad,
        }
    }

    /// Piper's four upsamplers at the profiled 560-frame input.
    fn piper_upsamplers() -> [PolyphaseDims; 4] {
        [
            poly(512, 256, 560, 4480, 16, 8, 4),
            poly(256, 128, 4480, 35840, 16, 8, 4),
            poly(128, 64, 35840, 71680, 4, 2, 1),
            poly(64, 32, 71680, 143360, 4, 2, 1),
        ]
    }

    /// Each bound of [`polyphase_indices_fit`] accepted exactly at its limit
    /// and rejected one past it, the other bounds kept clear.
    #[test]
    fn polyphase_indices_are_bounded_exactly() {
        for d in piper_upsamplers() {
            assert!(polyphase_indices_fit(&d), "{d:?}");
        }
        let edge = |why: &str, ok: PolyphaseDims, past: PolyphaseDims| {
            assert!(polyphase_indices_fit(&ok), "{why} at the limit: {ok:?}");
            assert!(
                !polyphase_indices_fit(&past),
                "{why} past the limit: {past:?}"
            );
        };
        let u = U32_MAX;
        // the epilogue row `q·s + φ` of the last column reaches i32::MAX
        edge(
            "epilogue row",
            poly(1, 1, 1, I32_MAX, 2, 2, 1),
            poly(1, 1, 1, I32_MAX, 2, 2, 2),
        );
        // 65537 · 65535 = u32::MAX
        edge(
            "W length",
            poly(65537, 65535, 1, 1, 1, 1, 0),
            poly(65537, 65536, 1, 1, 1, 1, 0),
        );
        edge(
            "X length",
            poly(65537, 1, 65535, 1, 1, 1, 0),
            poly(65537, 1, 65536, 1, 1, 1, 0),
        );
        edge(
            "output length",
            poly(1, 65537, 1, 65535, 1, 1, 0),
            poly(1, 65537, 1, 65536, 1, 1, 0),
        );
        edge(
            "K rounded up to KSTEP",
            poly(1, 1, 1, 1, u - 15, 1, 0),
            poly(1, 1, 1, 1, u - 14, 1, 0),
        );
        edge(
            "taps, k + s - 1",
            poly(1, 1, 1, 1, u - 15, 16, 0),
            poly(1, 1, 1, 1, u - 15, 17, 0),
        );
        edge(
            "C_out rounded up to the tile",
            poly(1, u - 63, 1, 1, 1, 1, 0),
            poly(1, u - 62, 1, 1, 1, 1, 0),
        );
        // without input channels `l_in` is no index, but still a u32 field
        edge(
            "l_in field",
            poly(0, 1, u, 1, 1, 1, 0),
            poly(0, 1, u + 1, 1, 1, 1, 0),
        );
        // no grid at all: the caller must never divide by these
        assert!(
            !polyphase_indices_fit(&poly(1, 1, 1, 1, 1, 0, 0)),
            "stride 0"
        );
        assert!(
            !polyphase_indices_fit(&poly(1, 1, 1, 0, 1, 1, 0)),
            "no output"
        );
    }

    /// The grid is `[q tiles, C_out tiles, stride]`; each axis is bounded by
    /// its own device limit, not by Vulkan's minimum.
    #[test]
    fn polyphase_grid_is_bounded_by_the_device() {
        let minimum = [65_535; 3];
        for d in piper_upsamplers() {
            assert!(polyphase_fits_device(&d, minimum), "{d:?}");
        }
        let edge = |why: &str, limits: [u32; 3], ok: PolyphaseDims, past: PolyphaseDims| {
            assert!(
                polyphase_fits_device(&ok, limits),
                "{why} at the limit: {ok:?}"
            );
            assert!(
                !polyphase_fits_device(&past, limits),
                "{why} past the limit: {past:?}"
            );
        };
        edge(
            "phases on z",
            minimum,
            poly(1, 1, 2, 1, 16, 65_535, 0),
            poly(1, 1, 2, 1, 16, 65_536, 0),
        );
        edge(
            "q tiles on x",
            [10, 1, 1],
            poly(1, 1, 640, 640, 1, 1, 0),
            poly(1, 1, 641, 641, 1, 1, 0),
        );
        edge(
            "C_out tiles on y",
            [1, 10, 1],
            poly(1, 640, 1, 1, 1, 1, 0),
            poly(1, 641, 1, 1, 1, 1, 0),
        );
    }

    fn direct_1d(
        c_in: usize,
        c_out: usize,
        l_in: usize,
        l_out: usize,
        k: usize,
        s: usize,
        pad: usize,
    ) -> DirectDims {
        DirectDims {
            batch: 1,
            c_in,
            c_out,
            group: 1,
            input: (l_in, 1),
            output: (l_out, 1),
            kernel: (k, 1),
            stride: (s, 1),
            pad: (pad, 0),
            dilation: (1, 1),
        }
    }

    /// A node the polyphase route refuses only for its grid is still run, by
    /// the gather kernel, on a device with Vulkan's minimum limits.
    #[test]
    fn a_stride_past_the_z_limit_falls_back_to_a_runnable_gather() {
        let minimum = [65_535; 3];
        // l_out = (2 - 1)·65536 + 16
        let p = poly(16, 16, 2, 65_552, 16, 65_536, 0);
        assert!(polyphase_indices_fit(&p) && !polyphase_fits_device(&p, minimum));
        let d = direct_1d(16, 16, 2, 65_552, 16, 65_536, 0);
        assert!(direct_indices_fit(&d) && direct_fits_device(&d, minimum));
    }

    /// Each bound of [`direct_indices_fit`] accepted exactly at its limit and
    /// rejected one past it.
    #[test]
    fn direct_indices_are_bounded_exactly() {
        for p in piper_upsamplers() {
            let d = direct_1d(p.c_in, p.c_out, p.l_in, p.l_out, p.kernel, p.stride, p.pad);
            assert!(direct_indices_fit(&d), "{d:?}");
        }
        let edge = |why: &str, ok: DirectDims, past: DirectDims| {
            assert!(direct_indices_fit(&ok), "{why} at the limit: {ok:?}");
            assert!(!direct_indices_fit(&past), "{why} past the limit: {past:?}");
        };
        let one = direct_1d(1, 1, 1, 1, 1, 1, 0);
        edge(
            "oh + pad",
            DirectDims {
                output: (I32_MAX, 1),
                pad: (1, 0),
                ..one
            },
            DirectDims {
                output: (I32_MAX, 1),
                pad: (2, 0),
                ..one
            },
        );
        edge(
            "r·dilation",
            DirectDims {
                kernel: (2, 1),
                dilation: (I32_MAX, 1),
                ..one
            },
            DirectDims {
                kernel: (2, 1),
                dilation: (I32_MAX + 1, 1),
                ..one
            },
        );
        edge(
            "stride as i32",
            DirectDims {
                stride: (1, I32_MAX),
                ..one
            },
            DirectDims {
                stride: (1, I32_MAX + 1),
                ..one
            },
        );
        edge(
            "outputs",
            DirectDims {
                c_out: 65537,
                output: (65535, 1),
                ..one
            },
            DirectDims {
                c_out: 65537,
                output: (65536, 1),
                ..one
            },
        );
        edge(
            "X length",
            DirectDims {
                c_in: 65537,
                input: (65535, 1),
                ..one
            },
            DirectDims {
                c_in: 65537,
                input: (65536, 1),
                ..one
            },
        );
        edge(
            "W length",
            DirectDims {
                c_in: 65537,
                c_out: 65535,
                ..one
            },
            DirectDims {
                c_in: 65537,
                c_out: 65536,
                ..one
            },
        );
        assert!(
            !direct_indices_fit(&DirectDims { group: 0, ..one }),
            "group 0"
        );
    }

    #[test]
    fn direct_grid_is_bounded_by_the_device() {
        let at = DirectDims {
            c_out: 65_535 * 256,
            ..direct_1d(1, 1, 1, 1, 1, 1, 0)
        };
        assert!(direct_fits_device(&at, [65_535; 3]));
        let past = DirectDims {
            c_out: at.c_out + 1,
            ..at
        };
        assert!(!direct_fits_device(&past, [65_535; 3]));
    }
}
