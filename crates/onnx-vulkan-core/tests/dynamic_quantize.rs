//! `DynamicQuantizeLinear` against the operator's definition on the host,
//! where `/` is a correctly rounded IEEE division, as in ONNX Runtime.
//!
//! A symmetric input range puts the zero point on a tie: `-min / scale` is
//! 127.5 up to rounding, so a quotient one ULP low rounds to 127 instead of
//! 128 and shifts every quantized value. Vulkan allows `OpFDiv` 2.5 ULP of
//! error, so the shaders must not rely on `/` being exact.

use onnx_vulkan_core::{ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, execute};
use std::collections::HashMap;
use vk_compute::VkContext;

const FLOAT: i32 = 1;

/// The ONNX definition: the range always includes 0, ties round to even.
fn reference(x: &[f32]) -> (Vec<i64>, f32, i64) {
    let min = x.iter().copied().fold(0.0f32, f32::min);
    let max = x.iter().copied().fold(0.0f32, f32::max);
    let scale = if max == min { 1.0 } else { (max - min) / 255.0 };
    let zp = (-min / scale).clamp(0.0, 255.0).round_ties_even();
    let y = x
        .iter()
        .map(|v| ((v / scale).round_ties_even() + zp).clamp(0.0, 255.0) as i64)
        .collect();
    (y, scale, zp as i64)
}

fn run(x: &[f32]) -> (Vec<i64>, f32, i64) {
    let node = NodeIr {
        domain: String::new(),
        op: "DynamicQuantizeLinear".to_string(),
        opset: 11,
        name: "dql".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".into(), "y_scale".into(), "y_zp".into()],
        attrs: HashMap::new(),
    };
    let ir = GraphIr {
        nodes: vec![node],
        initializers: HashMap::from([(
            "x".to_string(),
            InitializerIr {
                dtype: FLOAT,
                shape: vec![x.len() as i64],
                data: x.iter().flat_map(|v| v.to_le_bytes()).collect(),
            },
        )]),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".into(), "y_scale".into(), "y_zp".into()],
        ..Default::default()
    };
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    execute(&ir, &mut env).expect("graph execution");
    let y = env.host("y").unwrap().to_i64().unwrap();
    let scale = env.host("y_scale").unwrap().to_f32().unwrap()[0];
    let zp = env.host("y_zp").unwrap().to_i64().unwrap()[0];
    env.finish();
    (y, scale, zp)
}

fn assert_matches_reference(x: &[f32]) {
    let (want_y, want_scale, want_zp) = reference(x);
    let (y, scale, zp) = run(x);
    assert_eq!(scale.to_bits(), want_scale.to_bits(), "scale");
    assert_eq!(zp, want_zp, "zero point");
    let wrong = y.iter().zip(&want_y).filter(|(a, b)| a != b).count();
    assert_eq!(
        wrong,
        0,
        "{wrong} of {} values quantized differently",
        y.len()
    );
}

/// `[-r, r]` ranges, as produced by normalized images or `tanh`.
#[test]
fn a_symmetric_range_rounds_the_zero_point_tie_to_even() {
    for r in [0.6f32, 1.0, 0.5, 2.0, 3.7] {
        let n = 2048;
        let x: Vec<f32> = (0..n)
            .map(|i| -r + 2.0 * r * i as f32 / (n - 1) as f32)
            .collect();
        assert_matches_reference(&x);
    }
}

/// Many values per quantization step, so quotients land near every `.5`.
#[test]
fn quantized_values_follow_the_correctly_rounded_quotient() {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let x: Vec<f32> = (0..1 << 20)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 5.3 - 1.9
        })
        .collect();
    assert_matches_reference(&x);
}
