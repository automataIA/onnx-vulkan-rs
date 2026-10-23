//! ONNX `NonZero`: row-major stable coordinates with value-dependent output size.

use onnx_vulkan_core::host_ops::{BOOL, FLOAT, HostTensor, INT64};
use onnx_vulkan_core::{ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute};
use std::collections::HashMap;
use vk_compute::VkContext;

fn run(dtype: i32, shape: Vec<i64>, data: Vec<u8>) -> (Vec<i64>, Vec<i64>) {
    let node = NodeIr {
        domain: String::new(),
        op: "NonZero".into(),
        opset: 13,
        name: "nonzero".into(),
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        attrs: HashMap::new(),
    };
    let graph = GraphIr {
        nodes: vec![node],
        inputs: vec!["x".into()],
        outputs: vec!["y".into()],
        ..Default::default()
    };
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("x", Tensor::Host(HostTensor::new(dtype, shape, data)));
    execute(&graph, &mut env).expect("NonZero execution");
    let output = env.host("y").expect("output").clone();
    assert_eq!(output.dtype, INT64);
    let values = output
        .data
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("i64")))
        .collect();
    let shape = output.shape.clone();
    env.finish();
    (shape, values)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[test]
fn boolean_coordinates_are_stable_across_workgroups() {
    let mut input = vec![0u8; 3 * 260];
    for (row, column) in [(0usize, 0usize), (0, 259), (1, 3), (1, 255), (2, 259)] {
        input[row * 260 + column] = 1;
    }
    let (shape, values) = run(BOOL, vec![3, 260], input);
    assert_eq!(shape, vec![2, 5]);
    assert_eq!(values, vec![0, 0, 1, 1, 2, 0, 259, 3, 255, 259]);
}

#[test]
fn float_nan_is_nonzero_and_coordinates_follow_row_major_order() {
    let (shape, values) = run(
        FLOAT,
        vec![2, 3],
        f32_bytes(&[0.0, -1.0, f32::NAN, 0.0, 2.5, -0.0]),
    );
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(values, vec![0, 0, 1, 1, 2, 1]);
}

#[test]
fn int64_values_use_both_words_and_empty_output_has_dynamic_shape() {
    let values = [0i64, 1i64 << 40, -7, 0];
    let data: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let (shape, coordinates) = run(INT64, vec![2, 2], data);
    assert_eq!(shape, vec![2, 2]);
    assert_eq!(coordinates, vec![0, 1, 1, 0]);

    let (empty_shape, empty) = run(BOOL, vec![0, 3], vec![]);
    assert_eq!(empty_shape, vec![2, 0]);
    assert!(empty.is_empty());
}

#[test]
fn scalar_nonzero_has_rank_zero_result_and_zero_scalar_is_empty() {
    let (shape, values) = run(FLOAT, vec![], f32_bytes(&[f32::NAN]));
    assert_eq!(shape, vec![0, 1]);
    assert!(values.is_empty());

    let (shape, values) = run(FLOAT, vec![], f32_bytes(&[0.0]));
    assert_eq!(shape, vec![0, 0]);
    assert!(values.is_empty());
}

#[test]
fn cached_pipelines_are_specialized_by_input_dtype() {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = |dtype: i32, shape: Vec<i64>, data: Vec<u8>| {
        let node = NodeIr {
            domain: String::new(),
            op: "NonZero".into(),
            opset: 13,
            name: "nonzero_mixed_dtype".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attrs: HashMap::new(),
        };
        let graph = GraphIr {
            nodes: vec![node],
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        };
        let mut env = ExecutionEnv::new(&cache, &graph.initializers);
        env.set("x", Tensor::Host(HostTensor::new(dtype, shape, data)));
        execute(&graph, &mut env).expect("NonZero execution");
        let output = env.host("y").expect("output").clone();
        let coordinates: Vec<i64> = output
            .data
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("i64")))
            .collect();
        env.finish();
        coordinates
    };

    assert_eq!(graph(BOOL, vec![4], vec![0, 1, 0, 1]), vec![1, 3]);
    assert_eq!(
        graph(FLOAT, vec![4], f32_bytes(&[0.0, 0.0, 2.0, 0.0])),
        vec![2]
    );
}
