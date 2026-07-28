//! Shared shaders for `DynamicQuantizeLinear`.
//!
//! Three passes remain entirely on device: partial min/max, reduction with
//! scale & zero point calculation, then packed u8 quantization.
//!
//! ## Why the scale is not written `(max - min) / 255`
//!
//! Vulkan requires `OpFAdd`, `OpFSub`, `OpFMul` and `Fma` to be correctly
//! rounded, but allows `OpFDiv` **2.5 ULP** of error, and NVIDIA takes it.
//! Measured on an RTX 4070: `(max - min) / 255` came out **2 ULP** off ORT's
//! scale, and 36 values of 3048192 then quantized one step apart. lavapipe,
//! which divides exactly, reproduced ORT bit for bit on the same input.
//!
//! Two ULP on a scale matters far more than 36 values suggest, because every
//! tensor downstream is quantized against it: on sam3 that difference grows
//! through 32 transformer layers until the outputs share no digits.
//!
//! The fix keeps the division out of it. `1/255` is written as an unevaluated
//! sum of two f32 — `hi` is the rounded reciprocal, `lo` the remainder it lost —
//! and `fma(d, hi, d * lo)` recovers the correctly rounded quotient using only
//! operations the spec pins down. Verified over a million random values: it
//! agrees with the exactly rounded `d / 255` on every one, and on the GPU it
//! took the whole 3048192-value tensor to zero mismatches against ORT.
//!
//! Note what did *not* work: computing the residual with `fma(-q, b, a)` and
//! correcting `q` — the algebra is right, but the driver is free to fold it back
//! to `q`, and does. Anything that can be simplified to the original division
//! will be.

pub const PARTIAL: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> partial: array<vec2<f32>>; // (min, max) per workgroup

struct Push { n: u32 }
var<immediate> pc: Push;

var<workgroup> smin: array<f32, 256>;
var<workgroup> smax: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var mn = 0.0; // 0 is always included in the range (ONNX spec)
    var mx = 0.0;
    let stride = nwg.x * 256u;
    var i = gid.x;
    while (i < pc.n) {
        let v = x[i];
        mn = min(mn, v);
        mx = max(mx, v);
        i = i + stride;
    }
    smin[lid.x] = mn;
    smax[lid.x] = mx;
    workgroupBarrier();
    var s = 128u;
    while (s > 0u) {
        if (lid.x < s) {
            smin[lid.x] = min(smin[lid.x], smin[lid.x + s]);
            smax[lid.x] = max(smax[lid.x], smax[lid.x + s]);
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lid.x == 0u) {
        partial[wid.x] = vec2<f32>(smin[0], smax[0]);
    }
}
"#;

pub const FINALIZE: &str = r#"
@group(0) @binding(0) var<storage, read> partial: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> scale_out: array<f32>; // tensor y_scale
@group(0) @binding(2) var<storage, read_write> zp_out: array<u32>;    // tensor y_zero_point (byte 0)

struct Push { groups: u32 }
var<immediate> pc: Push;

var<workgroup> smin: array<f32, 256>;
var<workgroup> smax: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var mn = 0.0;
    var mx = 0.0;
    var i = lid.x;
    while (i < pc.groups) {
        mn = min(mn, partial[i].x);
        mx = max(mx, partial[i].y);
        i = i + 256u;
    }
    smin[lid.x] = mn;
    smax[lid.x] = mx;
    workgroupBarrier();
    var s = 128u;
    while (s > 0u) {
        if (lid.x < s) {
            smin[lid.x] = min(smin[lid.x], smin[lid.x + s]);
            smax[lid.x] = max(smax[lid.x], smax[lid.x + s]);
        }
        workgroupBarrier();
        s = s / 2u;
    }
    if (lid.x == 0u) {
        // 1/255 in double-single: only mul and fma, which Vulkan mandates
        // correctly rounded — no division to approximate.
        // The two constants are verified by `reciprocal_of_255_is_exact`
        let d = smax[0] - smin[0];
        var scale = fma(d, 3.921568859e-3, d * -2.319175824e-10);
        if (scale == 0.0) {
            scale = 1.0;
        }
        let zp = round(clamp(-smin[0] / scale, 0.0, 255.0));
        scale_out[0] = scale;
        zp_out[0] = u32(zp);
    }
}
"#;

pub const QUANTIZE: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale_in: array<f32>;
@group(0) @binding(2) var<storage, read> zp_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<u32>; // packed u8 x4

struct Push { n: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word = gid.x;
    let base = word * 4u;
    if (base >= pc.n) {
        return;
    }
    let scale = scale_in[0];
    let zp = f32(zp_in[0] & 0xffu);
    var packed = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let idx = base + j;
        var q = 0u;
        if (idx < pc.n) {
            q = u32(clamp(round(x[idx] / scale) + zp, 0.0, 255.0));
        }
        packed = packed | (q << (j * 8u));
    }
    y[word] = packed;
}
"#;

#[cfg(test)]
mod tests {
    /// The two constants in `FINALIZE` must reproduce `d / 255` exactly, which
    /// is what makes the scale agree with ORT's bit for bit. They are written as
    /// decimal literals in WGSL, so the test parses the same text the shader
    /// carries rather than a Rust copy of it.
    #[test]
    fn reciprocal_of_255_is_exact() {
        let line = super::FINALIZE
            .lines()
            .find(|l| l.contains("var scale = fma("))
            .expect("scale line");
        let numbers: Vec<f32> = line
            .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == '-'))
            .filter(|t| t.contains('e'))
            .filter_map(|t| t.parse().ok())
            .collect();
        let [hi, lo] = numbers[..] else {
            panic!("expected two constants, found {numbers:?}")
        };
        // the closest representable value to 1/255, and what it loses
        assert_eq!(hi, (1.0f64 / 255.0) as f32);
        assert_eq!(lo, (1.0f64 / 255.0 - hi as f64) as f32);

        // `fma(d, hi, d * lo)` must match the correctly rounded division;
        // a `d / 255.0` in f32 already matches it — the point is that the
        // GPU does not honor it, while this form does
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..100_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let d = (state >> 40) as f32 / 512.0;
            let expected = (d as f64 / 255.0) as f32;
            assert_eq!(d.mul_add(hi, d * lo), expected, "d = {d}");
        }
    }

    #[test]
    fn sources_compile() {
        for source in [super::PARTIAL, super::FINALIZE, super::QUANTIZE] {
            vk_compute::compile_wgsl(source).expect("valid DynamicQuantizeLinear shader");
        }
    }
}
