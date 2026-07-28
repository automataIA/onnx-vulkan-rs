//! Liveness: an intermediate dies at its last reader, not at the end of the block.
//!
//! The measurement is the point of the test. A chain of N nodes produces N
//! intermediates, but only two are alive at any moment (the node's input and its
//! output), so the VRAM the run holds must not grow with N. Before the buffer
//! pool it did, linearly: that is what made a 4066-node block ask for 9.6 GB.
//!
//! `stats::storage_peak_bytes` counts only device-local tensor buffers, so the
//! assertion is on the quantity that actually runs out on a GPU.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

/// 4 MB per tensor: big enough that a per-node leak is unmistakable next to the
/// noise of the small buffers the interpreter allocates for its own bookkeeping.
const ELEMS: usize = 1 << 20;
const BYTES: u64 = (ELEMS * 4) as u64;
const CHAIN: usize = 32;

fn relu_chain() -> GraphIr {
    let nodes = (0..CHAIN)
        .map(|i| NodeIr {
            domain: String::new(),
            op: "Relu".to_string(),
            since_version: 14,
            name: format!("relu_{i}"),
            inputs: vec![if i == 0 {
                "x".to_string()
            } else {
                format!("v{i}")
            }],
            outputs: vec![if i + 1 == CHAIN {
                "out".to_string()
            } else {
                format!("v{}", i + 1)
            }],
            attrs: HashMap::new(),
        })
        .collect();
    GraphIr {
        nodes,
        initializers: HashMap::<String, InitializerIr>::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    }
}

#[test]
fn a_chain_of_intermediates_does_not_grow_the_peak() {
    let ir = relu_chain();
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);

    let input: Vec<f32> = (0..ELEMS).map(|i| (i % 97) as f32 + 1.0).collect();
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set(
        "x",
        Tensor::Host(HostTensor::from_f32(vec![ELEMS as i64], &input)),
    );

    vk_compute::stats::reset_storage_peak();
    execute(&ir, &mut env).expect("chain execution");
    let peak = vk_compute::stats::storage_peak_bytes();

    let out = env
        .host("out")
        .expect("output on host")
        .to_f32()
        .expect("f32");
    env.finish();

    // Relu on positive values is the identity: the chain must not change the data
    assert_eq!(out, input, "the chain altered the values");

    // input + output + a few service buffers; without reuse they would be 33
    let buffers = peak.div_ceil(BYTES);
    assert!(
        buffers <= 4,
        "peak {peak} bytes = {buffers} 4 MB tensors on a chain of {CHAIN}: \
         intermedi are not being reused"
    );
}
