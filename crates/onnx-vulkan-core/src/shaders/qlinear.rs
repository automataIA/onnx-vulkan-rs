//! Shared shaders for the QOperator (static int8) family.
//!
//! These operators differ from the QDQ form in *where* the requantization
//! happens, not in what it computes. QDQ writes it out as a graph — dequantize
//! to f32, convolve, quantize back — and pays a full f32 round trip; the
//! QOperator form keeps the accumulator in int32 and requantizes once, which
//! is also what the reference implementation does. That is the 3% the QDQ
//! variant of resnet50 diverges from its own golden by.
//!
//! So there is no `QLinearConv` kernel here: the integer part is exactly
//! `ConvInteger` (and `MatMulInteger` for the matmul), and what is new is the
//! **epilogue** — [`REQUANTIZE`] — applied to the int32 accumulator.

/// The int32 accumulator, requantized to u8/i8.
///
/// `y = saturate(round(f32(acc + bias[c]) · ratio[c]) + y_zp)`, where
/// `ratio[c] = x_scale · w_scale[c] / y_scale` is precomputed on the host: it
/// is a constant of the graph, and computing it here would put a division in
/// the inner loop of every output element for no reason.
///
/// The channel index is `(i / inner) % axis_len`, the same expression
/// `QuantizeLinear` uses, so one shader serves both the convolution
/// (`inner = H·W`, `axis_len = C_out`) and the matmul (`inner = 1`,
/// `axis_len = N`), per-tensor being the degenerate `axis_len = 1`.
///
/// One thread per `u32`, i.e. per four elements, so the packed bytes are
/// written without atomics — the same trick as [`super::quantize_linear::QUANTIZE`].
pub const REQUANTIZE: &str = r#"
@group(0) @binding(0) var<storage, read> acc: array<i32>;
@group(0) @binding(1) var<storage, read> bias: array<i32>;
@group(0) @binding(2) var<storage, read> ratio: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

struct Push {
    n: u32, inner: u32, axis_len: u32,
    signed: u32, has_bias: u32, y_zp: i32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    let base = word * 4u;
    if (base >= pc.n) { return; }
    var packed = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let i = base + j;
        if (i >= pc.n) { break; }
        let c = (i / pc.inner) % pc.axis_len;
        var a = acc[i];
        if (pc.has_bias != 0u) { a = a + bias[c]; }
        // round() in WGSL is ties-to-even, as the ONNX spec requires
        var q = i32(round(f32(a) * ratio[c])) + pc.y_zp;
        if (pc.signed != 0u) {
            q = clamp(q, -128, 127);
        } else {
            q = clamp(q, 0, 255);
        }
        packed = packed | ((u32(q) & 0xffu) << (j * 8u));
    }
    out[word] = packed;
}
"#;

pub const REQUANTIZE_BINDINGS: u32 = 4;
/// 6 fields in the push constant struct.
pub const REQUANTIZE_PUSH_BYTES: u32 = 24;

/// `com.microsoft::QLinearAdd`: two quantized tensors summed on a third scale.
///
/// `C = saturate(round(ra·(A − a_zp) + rb·(B − b_zp)) + c_zp)`, with
/// `ra = A_scale / C_scale` and `rb = B_scale / C_scale` precomputed on the
/// host. Written as one pass rather than as dequantize-dequantize-add-quantize:
/// the decomposition is four passes over the largest tensors in the model
/// (resnet50 adds at 256×56×56) and produces the same numbers.
///
/// Broadcasting is the general ONNX one, with the same push layout as the f32
/// elementwise template — the last `QLinearAdd` of both models adds a `[1000]`
/// bias to a `[N, 1000]` logit tensor.
pub const QLINEAR_ADD: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;   // u8/i8 packed
@group(0) @binding(1) var<storage, read> b: array<u32>;   // u8/i8 packed
@group(0) @binding(2) var<storage, read_write> out: array<u32>;

struct Push {
    n: u32, rank: u32, pad0: u32, pad1: u32,
    os0: vec4<u32>, os1: vec4<u32>,
    as0: vec4<u32>, as1: vec4<u32>,
    bs0: vec4<u32>, bs1: vec4<u32>,
    ra: f32, rb: f32,
    a_zp: i32, b_zp: i32, c_zp: i32, signed: u32,
}
var<immediate> pc: Push;

fn dim(v0: vec4<u32>, v1: vec4<u32>, d: u32) -> u32 {
    if (d < 4u) { return v0[d]; }
    return v1[d - 4u];
}

fn as_signed(v: i32, signed: u32) -> i32 {
    if (signed != 0u && v > 127) { return v - 256; }
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    let base = word * 4u;
    if (base >= pc.n) { return; }
    var packed = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let i = base + j;
        if (i >= pc.n) { break; }
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
        let av = as_signed(i32((a[off_a >> 2u] >> ((off_a & 3u) * 8u)) & 0xffu), pc.signed);
        let bv = as_signed(i32((b[off_b >> 2u] >> ((off_b & 3u) * 8u)) & 0xffu), pc.signed);
        let sum = pc.ra * f32(av - pc.a_zp) + pc.rb * f32(bv - pc.b_zp);
        var q = i32(round(sum)) + pc.c_zp;
        if (pc.signed != 0u) {
            q = clamp(q, -128, 127);
        } else {
            q = clamp(q, 0, 255);
        }
        packed = packed | ((u32(q) & 0xffu) << (j * 8u));
    }
    out[word] = packed;
}
"#;

pub const QLINEAR_ADD_BINDINGS: u32 = 3;
/// The 112 bytes of the broadcast layout plus six scalars of quantization.
pub const QLINEAR_ADD_PUSH_BYTES: u32 = 136;

/// `MaxPool` on a quantized tensor, in the quantized domain.
///
/// A QOperator graph pools *between* two convolutions without leaving int8:
/// resnet50-int8's `MaxPool` takes a `uint8` tensor, and the f32 pooling kernel
/// would read the packed bytes as floats. It needs no scale and no zero point —
/// quantization is monotone, so the maximum of the codes is the code of the
/// maximum — which is also why this is a pooling shader and not a member of
/// the QLinear family proper.
///
/// One thread per output `u32`, four elements, for the same packing reason as
/// [`REQUANTIZE`].
pub const MAXPOOL_Q: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;   // u8/i8 packed
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
struct Push {
    total: u32, c: u32,
    h_in: u32, w_in: u32, h_out: u32, w_out: u32,
    kh: u32, kw: u32, sh: u32, sw: u32,
    phb: u32, pwb: u32, dh: u32, dw: u32, signed: u32,
}
var<immediate> pc: Push;

fn as_signed(v: i32, signed: u32) -> i32 {
    if (signed != 0u && v > 127) { return v - 256; }
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    let base = word * 4u;
    if (base >= pc.total) { return; }
    var packed = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let o = base + j;
        if (o >= pc.total) { break; }
        let ow = o % pc.w_out;
        let t1 = o / pc.w_out;
        let oh = t1 % pc.h_out;
        let t2 = t1 / pc.h_out;
        let ch = t2 % pc.c;
        let bn = t2 / pc.c;
        let plane = (bn * pc.c + ch) * pc.h_in * pc.w_in;

        // the padding of a quantized MaxPool contributes nothing, exactly as
        // in the f32 kernel: out-of-bound taps are skipped, not read as zero
        var acc = -2147483648;
        for (var r = 0u; r < pc.kh; r = r + 1u) {
            let ih = i32(oh) * i32(pc.sh) - i32(pc.phb) + i32(r) * i32(pc.dh);
            if (ih < 0 || ih >= i32(pc.h_in)) { continue; }
            for (var s = 0u; s < pc.kw; s = s + 1u) {
                let iw = i32(ow) * i32(pc.sw) - i32(pc.pwb) + i32(s) * i32(pc.dw);
                if (iw < 0 || iw >= i32(pc.w_in)) { continue; }
                let idx = plane + u32(ih) * pc.w_in + u32(iw);
                let v = as_signed(i32((x[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu), pc.signed);
                acc = max(acc, v);
            }
        }
        packed = packed | ((u32(acc) & 0xffu) << (j * 8u));
    }
    out[word] = packed;
}
"#;

pub const MAXPOOL_Q_BINDINGS: u32 = 2;
/// 15 fields, the same geometry the f32 pooling kernel takes.
pub const MAXPOOL_Q_PUSH_BYTES: u32 = 60;

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        vk_compute::compile_wgsl(super::REQUANTIZE).expect("valid REQUANTIZE shader");
        vk_compute::compile_wgsl(super::QLINEAR_ADD).expect("valid QLinearAdd shader");
        vk_compute::compile_wgsl(super::MAXPOOL_Q).expect("valid quantized MaxPool shader");
    }
}
