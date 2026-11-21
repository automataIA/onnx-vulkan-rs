//! `Resize` of a rank-3 tensor [N, C, W]: the 1-D upsampling of audio models
//! (the Kokoro and StyleTTS 2 decoders). It runs the 2-D kernel with H = 1, so
//! the result must match the rank-4 resize of [N, C, 1, W], and for `nearest`
//! the plain repetition of every sample.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

/// [1, 2, 3]
const X: [f32; 6] = [0.5, -1.0, 2.0, 3.0, 0.25, -0.75];

fn run(mode: &str, shape: Vec<i64>, scales: &[f32], use_sizes: bool) -> (Vec<f32>, Vec<i64>) {
    let (inputs, name, data): (Vec<String>, &str, Vec<f32>) = if use_sizes {
        let sizes: Vec<f32> = shape
            .iter()
            .zip(scales)
            .map(|(&d, &s)| (d as f32 * s).floor())
            .collect();
        (
            vec!["x".into(), String::new(), String::new(), "sizes".into()],
            "sizes",
            sizes,
        )
    } else {
        (vec!["x".into(), String::new(), "scales".into()], "scales", scales.to_vec())
    };
    let init = if use_sizes {
        InitializerIr {
            dtype: onnx_vulkan_core::host_ops::INT64,
            shape: vec![data.len() as i64],
            data: data.iter().flat_map(|v| (*v as i64).to_le_bytes()).collect(),
        }
    } else {
        InitializerIr {
            dtype: FLOAT,
            shape: vec![data.len() as i64],
            data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    };
    let ir = GraphIr {
        nodes: vec![NodeIr {
            domain: String::new(),
            op: "Resize".to_string(),
            opset: 13,
            name: "Resize_0".to_string(),
            inputs,
            outputs: vec!["out".to_string()],
            attrs: HashMap::from([("mode".to_string(), AttrValue::String(mode.to_string()))]),
        }],
        initializers: HashMap::from([(name.to_string(), init)]),
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        ..Default::default()
    };
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(HostTensor::from_f32(shape, &X)));
    execute(&ir, &mut env).expect("graph execution");
    let out = env.host("out").expect("output on host").clone();
    env.finish();
    (out.to_f32().expect("output f32"), out.shape)
}

#[test]
fn nearest_rank3_repeats_every_sample() {
    let (got, shape) = run("nearest", vec![1, 2, 3], &[1.0, 1.0, 2.0], false);
    assert_eq!(shape, vec![1, 2, 6]);
    let want: Vec<f32> = X.iter().flat_map(|&v| [v, v]).collect();
    assert_eq!(got, want);
}

#[test]
fn rank3_equals_rank4_with_unit_height() {
    for mode in ["nearest", "linear", "cubic"] {
        for use_sizes in [false, true] {
            let (got3, shape3) = run(mode, vec![1, 2, 3], &[1.0, 1.0, 3.0], use_sizes);
            let (got4, shape4) = run(mode, vec![1, 2, 1, 3], &[1.0, 1.0, 1.0, 3.0], use_sizes);
            assert_eq!(shape3, vec![1, 2, 9], "{mode}");
            assert_eq!(shape4, vec![1, 2, 1, 9], "{mode}");
            // same kernel and push constants; the driver may still contract
            // the cubic weights differently per pipeline build, so allow 1e-5
            assert_eq!(got3.len(), got4.len());
            for (a, b) in got3.iter().zip(&got4) {
                assert!((a - b).abs() <= 1e-5, "{mode} sizes={use_sizes}: {a} != {b}");
            }
        }
    }
}
