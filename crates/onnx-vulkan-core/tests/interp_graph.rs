//! Execution of a synthetic `GraphIr` directly from the core, without ONNX
//! Runtime: host shape-math + GPU dispatch in the same graph.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT64};
use onnx_vulkan_core::{
    ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        since_version: 13,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: HashMap::new(),
    }
}

fn f32_initializer(shape: Vec<i64>, values: &[f32]) -> InitializerIr {
    InitializerIr {
        dtype: FLOAT,
        shape,
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn i64_initializer(values: &[i64]) -> InitializerIr {
    InitializerIr {
        dtype: INT64,
        shape: vec![values.len() as i64],
        data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// `x` [2,4] → Mul by a broadcast initializer [4] → Relu → Reshape to [8],
/// plus a host-resolved `Shape`: covers both interpreter paths.
#[test]
fn runs_a_synthetic_graph_without_ort() {
    let context = VkContext::new().expect("Vulkan context");
    let x = [1.0f32, -2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0];
    let w = [0.5f32, 2.0, 1.0, -1.0];

    let mut initializers = HashMap::new();
    initializers.insert("w".to_string(), f32_initializer(vec![4], &w));
    initializers.insert("new_shape".to_string(), i64_initializer(&[8]));

    let ir = GraphIr {
        nodes: vec![
            node("Shape", &["x"], &["x_shape"]),
            node("Mul", &["x", "w"], &["prod"]),
            node("Relu", &["prod"], &["act"]),
            node("Reshape", &["act", "new_shape"], &["out"]),
        ],
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string(), "x_shape".to_string()],
    };

    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(HostTensor::from_f32(vec![2, 4], &x)));
    execute(&ir, &mut env).expect("graph execution");

    let expected: Vec<f32> = x
        .iter()
        .zip(w.iter().cycle())
        .map(|(a, b)| (a * b).max(0.0))
        .collect();
    // the Mul/Relu/Reshape branch must have ended up in VRAM, not on the host
    // path; `out` is enough to tell, because a host `Mul` would carry the whole
    // chain to the host. The intermediates themselves are gone by now: liveness
    // releases each one at its last reader.
    assert!(env.on_device("out"), "output must remain device-resident");
    assert!(!env.on_device("prod"), "the intermediate must be released");
    assert!(!env.on_device("x_shape"), "Shape stays on host");

    let out = env.host("out").expect("output on host");
    assert_eq!(out.shape, vec![8]);
    assert_eq!(out.to_f32().expect("output f32"), expected);
    assert_eq!(
        env.host("x_shape")
            .expect("host shape")
            .to_i64()
            .expect("i64"),
        vec![2, 4]
    );
    env.finish();
}
