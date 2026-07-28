//! The session cache (pipelines + packed weights) is owned by the caller:
//! two executions on the same `KernelCache` reuse everything, two distinct
//! caches remain independent.

use onnx_vulkan_core::host_ops::{HostTensor, UINT8};
use onnx_vulkan_core::{
    ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, execute,
};
use std::collections::HashMap;
use vk_compute::VkContext;

const M: usize = 2;
const K: usize = 8;
const N: usize = 4;
const A_ZP: u8 = 3;
const B_ZP: u8 = 5;

fn u8_initializer(shape: Vec<i64>, data: Vec<u8>) -> InitializerIr {
    InitializerIr {
        dtype: UINT8,
        shape,
        data,
    }
}

fn matmul_integer_graph() -> GraphIr {
    let b: Vec<u8> = (0..K * N).map(|i| (i * 7 % 251) as u8).collect();
    let mut initializers = HashMap::new();
    initializers.insert("b".to_string(), u8_initializer(vec![K as i64, N as i64], b));
    initializers.insert("a_zp".to_string(), u8_initializer(vec![], vec![A_ZP]));
    initializers.insert("b_zp".to_string(), u8_initializer(vec![], vec![B_ZP]));

    GraphIr {
        nodes: vec![NodeIr {
            domain: String::new(),
            op: "MatMulInteger".to_string(),
            since_version: 10,
            name: "mmi".to_string(),
            inputs: vec![
                "a".to_string(),
                "b".to_string(),
                "a_zp".to_string(),
                "b_zp".to_string(),
            ],
            outputs: vec!["out".to_string()],
            attrs: HashMap::new(),
        }],
        initializers,
        inputs: vec!["a".to_string()],
        outputs: vec!["out".to_string()],
    }
}

fn expected(a: &[u8], b: &[u8]) -> Vec<i32> {
    let mut out = vec![0i32; M * N];
    for row in 0..M {
        for col in 0..N {
            out[row * N + col] = (0..K)
                .map(|k| {
                    (i32::from(a[row * K + k]) - i32::from(A_ZP))
                        * (i32::from(b[k * N + col]) - i32::from(B_ZP))
                })
                .sum();
        }
    }
    out
}

fn run_once(cache: &KernelCache<'_>, ir: &GraphIr, a: &[u8]) -> Vec<i32> {
    let mut env = ExecutionEnv::new(cache, &ir.initializers);
    env.set(
        "a",
        Tensor::Host(HostTensor::new(UINT8, vec![M as i64, K as i64], a.to_vec())),
    );
    execute(ir, &mut env).expect("MatMulInteger execution");
    let out = env.host("out").expect("output on host");
    assert_eq!(out.shape, vec![M as i64, N as i64]);
    let values: Vec<i32> = out
        .data
        .chunks_exact(4)
        .map(|w| i32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    env.finish();
    values
}

#[test]
fn a_warm_cache_is_reused_across_runs() {
    let context = VkContext::new().expect("Vulkan context");
    let ir = matmul_integer_graph();
    let b = &ir.initializers["b"].data;
    let a: Vec<u8> = (0..M * K).map(|i| (i * 13 % 199) as u8).collect();
    let want = expected(&a, b);

    let cache = KernelCache::new(&context);
    assert_eq!(cache.builds(), (0, 0), "cold cache");

    assert_eq!(run_once(&cache, &ir, &a), want);
    let after_first = cache.builds();
    assert_eq!(after_first.1, 1, "B packed only once");
    assert!(
        after_first.0 >= 2,
        "pack + matmul compiled: {after_first:?}"
    );

    // second execution: same results, no recompilation or packing
    assert_eq!(run_once(&cache, &ir, &a), want);
    assert_eq!(cache.builds(), after_first, "warm cache: zero rebuild");
}

#[test]
fn separate_caches_do_not_share_state() {
    let context = VkContext::new().expect("Vulkan context");
    let ir = matmul_integer_graph();
    let b = &ir.initializers["b"].data;
    let a: Vec<u8> = (0..M * K).map(|i| (i * 5 % 197) as u8).collect();
    let want = expected(&a, b);

    let first = KernelCache::new(&context);
    assert_eq!(run_once(&first, &ir, &a), want);

    let second = KernelCache::new(&context);
    assert_eq!(second.builds(), (0, 0), "new cache starts empty");
    assert_eq!(run_once(&second, &ir, &a), want);
    assert_eq!(second.builds(), first.builds());
}
