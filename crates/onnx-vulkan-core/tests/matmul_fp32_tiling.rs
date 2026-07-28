//! f32 `MatMul` against a CPU reference.
//!
//! The kernel stages a 64×64 tile and gives each thread a 4×4 micro-tile, so
//! the shapes that matter are the ones that cross a tile boundary and the ones
//! that do not divide it — plus batching, since the batch offsets for A and B
//! are computed independently and broadcasting lets them differ. The last cases
//! are thin enough that the dispatch picks the 16×16 kernel instead, so both
//! geometries are covered.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute};
use std::collections::HashMap;
use vk_compute::VkContext;

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

/// `[ba, m, k] × [bb, k, n]`, with `ba` or `bb` allowed to be 1 (broadcast).
fn reference(a: &[f32], b: &[f32], ba: usize, bb: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let batch = ba.max(bb);
    let mut out = vec![0.0; batch * m * n];
    for z in 0..batch {
        let (za, zb) = (if ba == 1 { 0 } else { z }, if bb == 1 { 0 } else { z });
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a[za * m * k + i * k + p] * b[zb * k * n + p * n + j];
                }
                out[z * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

#[test]
fn matmul_tiles_edges_and_broadcast_match_the_cpu_reference() {
    // (ba, bb, m, k, n): a tile-exact case, one that divides nothing, one
    // single-tile-column shape like attention's N=64, and both broadcasts.
    let cases = [
        (1usize, 1usize, 128usize, 64usize, 128usize),
        (1, 1, 130, 70, 67),
        (2, 2, 72, 72, 64),
        (3, 1, 65, 33, 65),
        (1, 3, 65, 33, 65),
        // Thin enough that the dispatch falls back to the 16×16 kernel.
        (1, 1, 1, 64, 96),
        (2, 2, 3, 40, 80),
        // M = 1 and K large enough to split: the GEMV path, whose partials are
        // summed by a second dispatch. roberta's own shape, then two that
        // divide neither the 32-column tile nor the split evenly.
        (1, 1, 1, 768, 768),
        (1, 1, 1, 256, 100),
        (1, 1, 1, 130, 33),
    ];
    for (ba, bb, m, k, n) in cases {
        let a = pseudo(ba * m * k, 7);
        let b = pseudo(bb * k * n, 11);
        let ir = GraphIr {
            nodes: vec![NodeIr {
                domain: String::new(),
                op: "MatMul".to_string(),
                since_version: 13,
                name: "mm".to_string(),
                inputs: vec!["A".to_string(), "B".to_string()],
                outputs: vec!["Y".to_string()],
                attrs: HashMap::new(),
            }],
            initializers: HashMap::new(),
            inputs: vec!["A".to_string(), "B".to_string()],
            outputs: vec!["Y".to_string()],
        };
        let context = VkContext::new().expect("contesto Vulkan");
        let cache = KernelCache::new(&context);
        let mut env = ExecutionEnv::new(&cache, &ir.initializers);
        env.set(
            "A",
            Tensor::Host(HostTensor::from_f32(
                vec![ba as i64, m as i64, k as i64],
                &a,
            )),
        );
        env.set(
            "B",
            Tensor::Host(HostTensor::from_f32(
                vec![bb as i64, k as i64, n as i64],
                &b,
            )),
        );
        execute(&ir, &mut env).expect("MatMul execution");
        let got = env
            .host("Y")
            .expect("output on host")
            .clone()
            .to_f32()
            .expect("output f32");
        env.finish();

        let want = reference(&a, &b, ba, bb, m, k, n);
        assert_eq!(got.len(), want.len(), "{ba}x{bb} {m}x{k}x{n}: lunghezze");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-5 + 1e-4 * w.abs(),
                "{ba}x{bb} {m}x{k}x{n} element {i}: {g} != {w}"
            );
        }
    }
}
