//! `ArgMax`, as the ONNX op and as the runtime helper a decode step calls.
//!
//! Two properties are worth pinning down and neither is "the maximum is found".
//! The first is **which** index comes back when values tie: the kernel reduces a
//! tree, so the tie-break has to survive every level of it, and the answer ONNX
//! wants is the first occurrence. The second is that the split path and the
//! single-workgroup path agree — above `ARGMAX_MIN_SPLIT` elements the axis is
//! cut across ~150 workgroups and reduced twice, which is a different code path
//! from a row that fits one workgroup.

use onnx_vulkan_core::host_ops::{HostTensor, INT64};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, argmax_of,
    execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: "ArgMax".to_string(),
        opset: 13,
        name: "argmax_0".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn graph(node: NodeIr) -> GraphIr {
    GraphIr {
        nodes: vec![node],
        initializers: HashMap::<String, InitializerIr>::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
        ..Default::default()
    }
}

fn run(ir: &GraphIr, x: HostTensor) -> (Vec<i64>, Vec<i64>) {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(x));
    execute(ir, &mut env).expect("graph execution");
    let out = env.host("out").expect("output on host").clone();
    env.finish();
    assert_eq!(out.dtype, INT64, "ArgMax outputs int64");
    (out.to_i64().expect("indices"), out.shape)
}

/// Host reference over one axis, first occurrence.
fn argmax_ref(x: &[f32], shape: &[i64], axis: usize) -> Vec<i64> {
    let c = shape[axis] as usize;
    let inner: usize = shape[axis + 1..].iter().product::<i64>().max(1) as usize;
    let rows = x.len() / c;
    (0..rows)
        .map(|r| {
            let base = (r / inner) * c * inner + r % inner;
            let mut best = f32::NEG_INFINITY;
            let mut at = 0i64;
            for k in 0..c {
                if x[base + k * inner] > best {
                    best = x[base + k * inner];
                    at = k as i64;
                }
            }
            at
        })
        .collect()
}

fn pseudo(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}

/// One row as wide as a vocabulary: the axis is split across workgroups and
/// reduced twice, which is the path a decode step takes.
#[test]
fn a_vocabulary_wide_row_is_split_across_workgroups() {
    let shape = vec![1i64, 1, 151936];
    let mut x = pseudo(151936, 11);
    // a unique maximum, deep in the last split, above anything `pseudo` emits
    x[140_003] = 9.0;
    let (got, out_shape) = run(
        &graph(node(&[
            ("axis", AttrValue::Int(-1)),
            ("keepdims", AttrValue::Int(0)),
        ])),
        HostTensor::from_f32(shape.clone(), &x),
    );
    assert_eq!(out_shape, vec![1, 1]);
    assert_eq!(got, vec![140_003]);
    assert_eq!(got, argmax_ref(&x, &shape, 2));
}

/// A row short enough to stay in one workgroup: no second reduction level, so
/// the tie-break inside a single tree is all that decides.
#[test]
fn a_classifier_row_stays_in_one_workgroup() {
    let shape = vec![4i64, 1000];
    let x = pseudo(4000, 3);
    let (got, out_shape) = run(
        &graph(node(&[("axis", AttrValue::Int(1))])),
        HostTensor::from_f32(shape.clone(), &x),
    );
    assert_eq!(out_shape, vec![4, 1], "keepdims defaults to 1");
    assert_eq!(got, argmax_ref(&x, &shape, 1));
}

/// Ties: with every element equal the answer is index 0, and with the maximum
/// repeated it is the **first** of them. A tree reduction gets this wrong unless
/// every level prefers the lower index, and the failure is invisible on random
/// data — hence a test that has nothing but ties.
#[test]
fn a_repeated_maximum_answers_its_first_index() {
    let flat = vec![0.5f32; 4096];
    let (got, _) = run(
        &graph(node(&[
            ("keepdims", AttrValue::Int(0)),
            ("axis", AttrValue::Int(0)),
        ])),
        HostTensor::from_f32(vec![4096], &flat),
    );
    assert_eq!(got, vec![0], "all equal: the first index wins");

    let mut twice = pseudo(4096, 5);
    twice[700] = 3.0;
    twice[3000] = 3.0;
    let (got, _) = run(
        &graph(node(&[
            ("keepdims", AttrValue::Int(0)),
            ("axis", AttrValue::Int(0)),
        ])),
        HostTensor::from_f32(vec![4096], &twice),
    );
    assert_eq!(got, vec![700], "two maxima: the earlier one");
}

/// A non-last axis, where consecutive elements of the reduced axis sit `inner`
/// apart: the strided read is the part the contiguous cases never exercise.
#[test]
fn a_channel_axis_reads_with_a_stride() {
    let shape = vec![2i64, 5, 3, 3];
    let x = pseudo(90, 17);
    let (got, out_shape) = run(
        &graph(node(&[
            ("axis", AttrValue::Int(1)),
            ("keepdims", AttrValue::Int(0)),
        ])),
        HostTensor::from_f32(shape.clone(), &x),
    );
    assert_eq!(out_shape, vec![2, 3, 3]);
    assert_eq!(got, argmax_ref(&x, &shape, 1));
}

/// Not every `ArgMax` is on floats — the support check reads the node and cannot
/// see the dtype, so an int64 input must have a path that works rather than one
/// that binds an int64 buffer to an `array<f32>`.
#[test]
fn an_integer_input_goes_host_side() {
    let x: Vec<i64> = vec![3, 9, 2, 9, 1, 0, 4, 7];
    let (got, out_shape) = run(
        &graph(node(&[
            ("axis", AttrValue::Int(1)),
            ("keepdims", AttrValue::Int(0)),
        ])),
        HostTensor::from_i64(vec![2, 4], &x),
    );
    assert_eq!(out_shape, vec![2]);
    assert_eq!(
        got,
        vec![1, 3],
        "row 0 ties at 1 and 3; row 1 max is 7 at 3"
    );
}

/// `select_last_index = 1` asks for the other index of a tie, which this kernel
/// does not produce. Refusing is the contract: a claimed node that answers the
/// first index would be silently wrong.
#[test]
fn the_reversed_tie_break_is_refused() {
    assert!(is_implemented_node(&node(&[])));
    assert!(!is_implemented_node(&node(&[(
        "select_last_index",
        AttrValue::Int(1)
    )])));
}

/// The runtime path: no `ArgMax` node in the graph at all, the reduction
/// enqueued onto a value a run already produced. This is what a decode step
/// calls instead of downloading the logits, and what it must agree with is the
/// op above.
#[test]
fn the_runtime_helper_reduces_a_value_the_graph_produced() {
    let shape = vec![1i64, 2, 8192];
    let mut x = pseudo(16384, 23);
    x[5000] = 4.0; // row 0
    x[8192 + 77] = 4.0; // row 1
    let ir = GraphIr {
        nodes: vec![NodeIr {
            domain: String::new(),
            op: "Relu".to_string(),
            opset: 13,
            name: "relu_0".to_string(),
            inputs: vec!["x".to_string()],
            outputs: vec!["logits".to_string()],
            attrs: HashMap::new(),
        }],
        initializers: HashMap::new(),
        inputs: vec!["x".to_string()],
        outputs: vec!["logits".to_string()],
        ..Default::default()
    };

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(HostTensor::from_f32(shape.clone(), &x)));
    execute(&ir, &mut env).expect("graph execution");
    let got = argmax_of(&mut env, "logits").expect("device argmax");
    env.finish();

    assert_eq!(got, vec![5000, 77]);
}
