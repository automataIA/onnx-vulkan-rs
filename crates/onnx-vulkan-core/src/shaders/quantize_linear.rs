//! Shared shaders and dispatch layouts for `QuantizeLinear` /
//! `DequantizeLinear` (static quantization, QDQ format).
//!
//! Single shader per direction covers both per-tensor and per-axis:
//! channel index is `(i / inner) % axis_len`, and per-tensor is degenerate
//! case `axis_len = 1`. Quantized data is packed in VRAM, 4 bytes per
//! `u32`, as `DynamicQuantizeLinear` already does.

pub const BINDINGS: u32 = 4;
/// 5 u32 fields in the push constant struct.
pub const PUSH_BYTES: u32 = 20;

/// `DequantizeLinear` on **int32** input, without unpacking.
///
/// This is the shape of convolution bias in QDQ graphs: weights and
/// activations are int8, the bias is int32 because it is already an
/// accumulation. One i32 per element fills the whole word, so the byte
/// extraction needed by the u8/i8 case would be wrong here.
pub const DEQUANTIZE_I32: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<i32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> zp: array<i32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Push { n: u32, inner: u32, axis_len: u32, signed: u32, has_zp: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    let c = (i / pc.inner) % pc.axis_len;
    var zero = 0;
    if (pc.has_zp != 0u) { zero = zp[c]; }
    out[i] = f32(x[i] - zero) * scale[c];
}
"#;

/// `DequantizeLinear`: `y = (x - zero_point) * scale`, with `x` packed u8/i8.
pub const DEQUANTIZE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<u32>;      // u8/i8 packed
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> zp: array<u32>;     // u8/i8 packed
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Push { n: u32, inner: u32, axis_len: u32, signed: u32, has_zp: u32 }
var<immediate> pc: Push;

// naga does not allow storage pointers as arguments: byte extraction is
// repeated inline on the two buffers.
fn as_signed(v: i32, signed: u32) -> i32 {
    if (signed != 0u && v > 127) { return v - 256; }
    return v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pc.n) { return; }
    let c = (i / pc.inner) % pc.axis_len;
    var zero = 0;
    if (pc.has_zp != 0u) {
        zero = as_signed(i32((zp[c >> 2u] >> ((c & 3u) * 8u)) & 0xffu), pc.signed);
    }
    let q = as_signed(i32((x[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu), pc.signed);
    out[i] = f32(q - zero) * scale[c];
}
"#;

/// `QuantizeLinear`: `y = saturate(round_ties_even(x / scale) + zero_point)`.
/// One thread produces a full `u32` (4 elements) for packed writing,
/// so no atomics are needed.
pub const QUANTIZE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> zp: array<u32>;     // u8/i8 packed
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

struct Push { n: u32, inner: u32, axis_len: u32, signed: u32, has_zp: u32 }
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
        var zero = 0;
        if (pc.has_zp != 0u) {
            let raw = i32((zp[c >> 2u] >> ((c & 3u) * 8u)) & 0xffu);
            if (pc.signed != 0u && raw > 127) { zero = raw - 256; } else { zero = raw; }
        }
        // round() in WGSL is ties-to-even, as required by the ONNX spec
        var q = i32(round(x[i] / scale[c])) + zero;
        if (pc.signed != 0u) {
            q = clamp(q, -128, 127);
            packed = packed | ((u32(q) & 0xffu) << (j * 8u));
        } else {
            q = clamp(q, 0, 255);
            packed = packed | (u32(q) << (j * 8u));
        }
    }
    out[word] = packed;
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        vk_compute::compile_wgsl(super::DEQUANTIZE).expect("valid DequantizeLinear shader");
        vk_compute::compile_wgsl(super::QUANTIZE).expect("valid QuantizeLinear shader");
    }
}
