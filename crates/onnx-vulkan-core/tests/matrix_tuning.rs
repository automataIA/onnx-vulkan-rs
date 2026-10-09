//! Artifact-selected matrix tiles must run and record an exact cache hit.
use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::tuning::*;
use onnx_vulkan_core::{ElementType, Executor, GraphIr, NodeIr, Tensor};
use std::collections::HashMap;
use std::sync::Arc;
use vk_compute::VkContext;

#[test]
fn matrix_artifacts_select_both_tiles_and_reject_unknown_variants() {
    let context = VkContext::new().unwrap();
    let (m, k, n) = (3usize, 5usize, 7usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 7.0) / 11.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 17.0) / 19.0).collect();
    let reference: Vec<f32> = (0..m * n)
        .map(|i| (0..k).map(|j| a[(i / n) * k + j] * b[j * n + i % n]).sum())
        .collect();
    for (op, variants) in [
        ("MatMul", vec!["matmul16", "matmul64", "unknown"]),
        ("Gemm", vec!["gemm16", "gemm64", "unknown"]),
    ] {
        let node = NodeIr {
            domain: String::new(),
            op: op.into(),
            opset: 21,
            name: "matrix".into(),
            inputs: vec!["a".into(), "b".into()],
            outputs: vec!["y".into()],
            attrs: HashMap::new(),
        };
        let graph = GraphIr {
            nodes: vec![node.clone()],
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            ..Default::default()
        };
        let tensor = |dims| TensorSignature {
            dtype: ElementType::Float32,
            dimensions: dims,
        };
        for variant in variants {
            let key = TuningKey {
                device: context.device_fingerprint().into(),
                workload: WorkloadSignature {
                    domain: String::new(),
                    op: op.into(),
                    inputs: vec![
                        tensor(vec![m as u64, k as u64]),
                        tensor(vec![k as u64, n as u64]),
                    ],
                    outputs: vec![tensor(vec![m as u64, n as u64])],
                    attributes_digest: attributes_digest(&node),
                },
                implementation: onnx_vulkan_core::shaders::matrix::implementation_fingerprint(),
            };
            let mut table = TuningTable::new();
            table.insert_best(
                key,
                TuningRecord::new(
                    TacticId::new("matrix-f32".into(), variant.into()),
                    Default::default(),
                    Measurement::new(20, 100, 90, 110).unwrap(),
                ),
            );
            let resolver = Arc::new(TacticResolver::new(TuningMode::RequireCache, table));
            let executor =
                Executor::with_tuning(&context, graph.clone(), resolver.clone()).unwrap();
            let result = executor.run(vec![
                (
                    "a",
                    Tensor::Host(HostTensor::from_f32(vec![m as i64, k as i64], &a)),
                ),
                (
                    "b",
                    Tensor::Host(HostTensor::from_f32(vec![k as i64, n as i64], &b)),
                ),
            ]);
            if variant == "unknown" {
                assert!(result.is_err());
                continue;
            }
            let output = result.unwrap().host("y").unwrap().to_f32().unwrap();
            assert!(
                output
                    .iter()
                    .zip(&reference)
                    .all(|(a, b)| (a - b).abs() < 1e-5)
            );
            assert_eq!(resolver.summary()[0].kind, SelectionKind::Hit);
        }
    }
}

#[test]
fn gemm16_matches_gemm64_for_transposes_coefficients_and_bias() {
    use onnx_vulkan_core::shaders::{gemm, matrix};
    use vk_compute::compile_wgsl;
    let ctx = VkContext::new().unwrap();
    let (m, k, n) = (17usize, 19usize, 23usize);
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 % 17.0 - 8.0) / 19.0).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 % 13.0 - 6.0) / 17.0).collect();
    for ta in [false, true] {
        for tb in [false, true] {
            for (cr, cc) in [(0, 0), (1, 1), (1, n), (m, 1), (m, n)] {
                let c: Vec<f32> = (0..(cr * cc).max(1)).map(|i| i as f32 / 100.0).collect();
                let expected: Vec<f32> = (0..m * n)
                    .map(|i| {
                        let (r, col) = (i / n, i % n);
                        let sum: f32 = (0..k)
                            .map(|j| {
                                a[if ta { j * m + r } else { r * k + j }]
                                    * b[if tb { col * k + j } else { j * n + col }]
                            })
                            .sum();
                        0.5 * sum
                            + if cr == 0 {
                                0.0
                            } else {
                                -0.25
                                    * c[(if cr == 1 { 0 } else { r }) * cc
                                        + if cc == 1 { 0 } else { col }]
                            }
                    })
                    .collect();
                let mut buffers = Vec::new();
                for values in [&a, &b, &c] {
                    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                    let buf = ctx.create_storage_buffer(bytes.len() as u64).unwrap();
                    ctx.stream_upload(&buf, &bytes).unwrap();
                    buffers.push(buf);
                }
                buffers.push(ctx.create_storage_buffer((m * n * 4) as u64).unwrap());
                let mut push = Vec::new();
                for x in [
                    m as u32,
                    k as u32,
                    n as u32,
                    u32::from(ta) | (u32::from(tb) << 1) | (u32::from(cr > 0) << 2),
                    0.5f32.to_bits(),
                    (-0.25f32).to_bits(),
                    cr.max(1) as u32,
                    cc.max(1) as u32,
                ] {
                    push.extend_from_slice(&x.to_le_bytes());
                }
                for (source, tile) in [(matrix::GEMM16, 16), (gemm::GEMM, 64)] {
                    let pipe = ctx
                        .create_pipeline(&compile_wgsl(source).unwrap(), 4, 32)
                        .unwrap();
                    ctx.dispatch(
                        &pipe,
                        &buffers.iter().collect::<Vec<_>>(),
                        &push,
                        [(n as u32).div_ceil(tile), (m as u32).div_ceil(tile), 1],
                    )
                    .unwrap();
                    let bytes = ctx.stream_download(&buffers[3], m * n * 4).unwrap();
                    for (raw, expected) in bytes.chunks_exact(4).zip(&expected) {
                        let actual = f32::from_le_bytes(raw.try_into().unwrap());
                        assert!(
                            (actual - expected).abs() < 1e-5,
                            "ta={ta} tb={tb} bias={cr}x{cc}: {actual} != {expected}"
                        );
                    }
                    ctx.destroy_pipeline(pipe);
                }
                for buf in buffers {
                    ctx.destroy_buffer(buf);
                }
            }
        }
    }
}
