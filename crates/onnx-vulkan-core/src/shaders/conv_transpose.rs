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
}
