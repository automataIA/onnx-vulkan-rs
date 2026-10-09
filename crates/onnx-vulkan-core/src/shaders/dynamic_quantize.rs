//! Shared shaders for `DynamicQuantizeLinear`.
//!
//! Two passes remain entirely on device, after the range is cleared: min/max,
//! then packed u8 quantization, whose workgroups each derive the scale and
//! zero point from the range.
//!
//! ## The range
//!
//! An 8-byte buffer, cleared to zero before the first pass: word 0 holds the
//! bits of `-min`, word 1 those of `max`. ONNX includes 0 in the range, so
//! both are ≥ 0, and the bits of non-negative floats order like their values:
//! `atomicMax` on the words is a max on the values, and the cleared buffer is
//! already the range `[0, 0]`. The sign bit is masked so a `-0.0` counts as
//! `0`. Min and max do not depend on the order they are taken in, so the
//! result is the one a single reduction would give, bit for bit.
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
//!
//! ## Why the zero point and the quantized values do not use `/` either
//!
//! `-min / scale` and `x / scale` are divisions too. With a symmetric range
//! (`[-0.6, 0.6]`, `[-1, 1]`) the zero point is 127.5 up to rounding: ORT's
//! exact quotient rounds to 128, and an RTX 4070 quotient one ULP low rounded
//! to 127, shifting every value. `div_rn` takes the approximate quotient, then
//! picks among its neighbours the one with the smallest residual `a - q·b`,
//! which `fma` computes exactly. A quotient of two floats is never halfway
//! between two floats, so the smallest residual is the correctly rounded one.
//! The candidates are offset by a push-constant `zero` the driver cannot see,
//! so `fma(-(a / b), b, a)` cannot be folded to 0 as above. On the RTX 4070
//! the form without the offset passed the same tests; the offset stays because
//! the fold is legal and was observed for the analogous pattern.

/// Correctly rounded `a / b` for finite `a` and `b > 0`; `zero` must be 0.
macro_rules! div_rn {
    () => {
        r#"
fn div_rn(a: f32, b: f32, zero: u32) -> f32 {
    let n = abs(a);
    let bits = bitcast<u32>(n / b) + zero;
    var best = bitcast<f32>(bits);
    var err = abs(fma(-best, b, n));
    for (var k = 1u; k <= 3u; k = k + 1u) {
        let up = bitcast<f32>(bits + k);
        let up_err = abs(fma(-up, b, n));
        if (up_err < err) {
            best = up;
            err = up_err;
        }
        if (bits >= k) {
            let down = bitcast<f32>(bits - k);
            let down_err = abs(fma(-down, b, n));
            if (down_err < err) {
                best = down;
                err = down_err;
            }
        }
    }
    return select(best, -best, a < 0.0);
}
"#
    };
}
pub(crate) use div_rn;

/// The binding, push constants and `load` of a plain `DynamicQuantizeLinear`,
/// for [`partial_body`] and [`quantize_body`].
macro_rules! plain_head {
    ($($push:tt)*) => {
        concat!(
            r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
struct Push { "#,
            $($push)*,
            r#" }
var<immediate> pc: Push;
fn load(i: u32) -> f32 { return x[i]; }
"#
        )
    };
}

/// Pass 1, after a head declaring `x`, `pc.n` and `load`: the (min, max) of
/// `load(i)` folded into the range (see the module documentation).
macro_rules! partial_body {
    () => {
        r#"
@group(0) @binding(1) var<storage, read_write> range: array<atomic<u32>, 2>; // bits of -min, max

var<workgroup> smin: array<f32, 256>;
var<workgroup> smax: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var mn = 0.0; // 0 is always included in the range (ONNX spec)
    var mx = 0.0;
    let stride = nwg.x * 256u;
    var i = gid.x;
    while (i < pc.n) {
        let v = load(i);
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
        // both ≥ 0 (the loop starts from 0); the mask turns -0.0 into 0
        atomicMax(&range[0], bitcast<u32>(-smin[0]) & 0x7fffffffu);
        atomicMax(&range[1], bitcast<u32>(smax[0]) & 0x7fffffffu);
    }
}
"#
    };
}

pub const PARTIAL: &str = concat!(plain_head!("n: u32"), partial_body!());

/// Pass 2, after a head declaring `x`, `pc.n`, `pc.zero` and `load`: `load(i)`
/// quantized, four bytes per word. The first invocation of each workgroup
/// derives the scale and zero point from the range, the same way in every
/// workgroup, and the first workgroup's also writes them out. One per
/// invocation would add a fifth `div_rn` to the four of the values: measured
/// on SAM 3's vision encoder, +1 ms on `DQL_quantize`.
macro_rules! quantize_body {
    () => {
        r#"
@group(0) @binding(1) var<storage, read> range: array<u32, 2>;        // bits of -min, max
@group(0) @binding(2) var<storage, read_write> scale_out: array<f32>; // tensor y_scale
@group(0) @binding(3) var<storage, read_write> zp_out: array<u32>;    // tensor y_zero_point (byte 0)
@group(0) @binding(4) var<storage, read_write> y: array<u32>;         // packed u8 x4

var<workgroup> shared_scale: f32;
var<workgroup> shared_zp: f32;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    if (lid.x == 0u) {
        let mn = -bitcast<f32>(range[0]);
        let mx = bitcast<f32>(range[1]);
        // 1/255 in double-single: only mul and fma, which Vulkan mandates
        // correctly rounded — no division to approximate.
        // The two constants are verified by `reciprocal_of_255_is_exact`
        let d = mx - mn;
        var scale = fma(d, 3.921568859e-3, d * -2.319175824e-10);
        if (scale == 0.0) {
            scale = 1.0;
        }
        let zp = round(clamp(div_rn(-mn, scale, pc.zero), 0.0, 255.0));
        shared_scale = scale;
        shared_zp = zp;
        if (gid.x == 0u) {
            scale_out[0] = scale;
            zp_out[0] = u32(zp);
        }
    }
    workgroupBarrier();
    let scale = shared_scale;
    let zp = shared_zp;
    let word = gid.x;
    let base = word * 4u;
    if (base >= pc.n) {
        return;
    }
    var packed = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let idx = base + j;
        var q = 0u;
        if (idx < pc.n) {
            q = u32(clamp(round(div_rn(load(idx), scale, pc.zero)) + zp, 0.0, 255.0));
        }
        packed = packed | (q << (j * 8u));
    }
    y[word] = packed;
}
"#
    };
}

pub const QUANTIZE: &str = concat!(
    div_rn!(),
    plain_head!("n: u32, zero: u32"),
    quantize_body!()
);

/// The head of a pass that quantizes `DecomposedGelu(x)`: the same push
/// constants as the plain pass, then the GELU's.
fn gelu_head(push: &str) -> String {
    use super::elementwise::{DECOMPOSED_GELU_FN, UNARY_HELPERS_ERF};
    format!(
        "{UNARY_HELPERS_ERF}{DECOMPOSED_GELU_FN}
@group(0) @binding(0) var<storage, read> x: array<f32>;
struct Push {{ {push}, divisor: f32, offset: f32, factor: f32, factor_first: u32 }}
var<immediate> pc: Push;
fn load(i: u32) -> f32 {{ return decomposed_gelu(x[i]); }}
"
    )
}

/// [`PARTIAL`] over `DecomposedGelu(x)`, which is never stored: the GELU is
/// computed again by [`gelu_quantize`].
pub fn gelu_partial() -> String {
    gelu_head("n: u32") + partial_body!()
}

/// [`QUANTIZE`] over `DecomposedGelu(x)`.
pub fn gelu_quantize() -> String {
    div_rn!().to_owned() + &gelu_head("n: u32, zero: u32") + quantize_body!()
}

#[cfg(test)]
mod tests {
    /// The two constants in `QUANTIZE` must reproduce `d / 255` exactly, which
    /// is what makes the scale agree with ORT's bit for bit. They are written as
    /// decimal literals in WGSL, so the test parses the same text the shader
    /// carries rather than a Rust copy of it.
    #[test]
    fn reciprocal_of_255_is_exact() {
        let line = super::QUANTIZE
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
        for source in [super::PARTIAL, super::QUANTIZE] {
            vk_compute::compile_wgsl(source).expect("valid DynamicQuantizeLinear shader");
        }
        for source in [super::gelu_partial(), super::gelu_quantize()] {
            vk_compute::compile_wgsl(&source).expect("valid quantizing GELU shader");
        }
    }
}
