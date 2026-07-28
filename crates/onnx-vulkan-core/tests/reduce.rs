//! Reduction over one axis: `ReduceMean`, `ReduceSum`, `ReduceMax`.
//!
//! The case that matters is `axes = [1]`, `keepdims = 1` on NCHW — the form
//! rfdetr uses to decompose the projector normalizations, which alone split
//! the graph in 18 places. Each test compares the kernel against a CPU
//! reference written here.

use onnx_vulkan_core::host_ops::{HostTensor, INT64};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor, constant_outputs,
    execute, fold_constant_params, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        since_version: 13,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: outputs.iter().map(|s| (*s).to_string()).collect(),
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn graph(nodes: Vec<NodeIr>, output: &str) -> GraphIr {
    GraphIr {
        nodes,
        initializers: HashMap::<String, InitializerIr>::new(),
        inputs: vec!["x".to_string()],
        outputs: vec![output.to_string()],
    }
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

fn run(ir: &GraphIr, x: HostTensor, output: &str) -> (Vec<f32>, Vec<i64>) {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(x));
    execute(ir, &mut env).expect("graph execution");
    let out = env.host(output).expect("output on host").clone();
    env.finish();
    (out.to_f32().expect("output f32"), out.shape)
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "different lengths");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 + 1e-4 * w.abs(),
            "element {i}: {g} != {w}"
        );
    }
}

/// CPU reference: reduction of `axis` of a tensor with shape `shape`.
fn reduce_ref(x: &[f32], shape: &[i64], axis: usize, kind: &str) -> Vec<f32> {
    let c = shape[axis] as usize;
    let inner: usize = shape[axis + 1..].iter().product::<i64>() as usize;
    let rows = x.len() / c;
    (0..rows)
        .map(|r| {
            let base = (r / inner) * c * inner + r % inner;
            let vals = (0..c).map(|k| x[base + k * inner]);
            match kind {
                "max" => vals.fold(f32::NEG_INFINITY, f32::max),
                "sum" => vals.sum(),
                _ => vals.sum::<f32>() / c as f32,
            }
        })
        .collect()
}

/// The rfdetr case: [1, 8, 5, 5], channel axis, `keepdims = 1`.
#[test]
fn reduce_mean_over_channels_keepdims() {
    let shape = vec![1i64, 8, 5, 5];
    let x = pseudo(200, 7);
    let ir = graph(
        vec![node(
            "ReduceMean",
            &["x"],
            &["out"],
            &[
                ("axes", AttrValue::Ints(vec![1])),
                ("keepdims", AttrValue::Int(1)),
            ],
        )],
        "out",
    );

    let (got, out_shape) = run(&ir, HostTensor::from_f32(shape.clone(), &x), "out");
    assert_eq!(out_shape, vec![1, 1, 5, 5]);
    assert_close(&got, &reduce_ref(&x, &shape, 1, "mean"));
}

/// Last axis (`inner = 1`, contiguous rows) with `keepdims = 0`: the form
/// rfdetr uses for `ReduceMax` in the transformer tail.
#[test]
fn reduce_max_last_axis_no_keepdims() {
    let shape = vec![4i64, 3, 16];
    let x = pseudo(192, 11);
    let ir = graph(
        vec![node(
            "ReduceMax",
            &["x"],
            &["out"],
            &[
                ("axes", AttrValue::Ints(vec![-1])),
                ("keepdims", AttrValue::Int(0)),
            ],
        )],
        "out",
    );

    let (got, out_shape) = run(&ir, HostTensor::from_f32(shape.clone(), &x), "out");
    assert_eq!(out_shape, vec![4, 3]);
    assert_close(&got, &reduce_ref(&x, &shape, 2, "max"));
}

/// Sum over the middle axis, default `keepdims` (= 1).
#[test]
fn reduce_sum_middle_axis_default_keepdims() {
    let shape = vec![2i64, 6, 7];
    let x = pseudo(84, 23);
    let ir = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["out"],
            &[("axes", AttrValue::Ints(vec![1]))],
        )],
        "out",
    );

    let (got, out_shape) = run(&ir, HostTensor::from_f32(shape.clone(), &x), "out");
    assert_eq!(out_shape, vec![2, 1, 7]);
    assert_close(&got, &reduce_ref(&x, &shape, 1, "sum"));
}

/// What the kernel **cannot** do must not be claimed: multiple axes, axes as
/// input, missing `axes`. Claiming them would split the block at runtime
/// instead of during capability resolution.
#[test]
fn unsupported_forms_are_not_claimed() {
    let single = node(
        "ReduceMean",
        &["x"],
        &["out"],
        &[("axes", AttrValue::Ints(vec![1]))],
    );
    assert!(is_implemented_node(&single), "a single axis is implemented");

    let multi = node(
        "ReduceMean",
        &["x"],
        &["out"],
        &[("axes", AttrValue::Ints(vec![2, 3]))],
    );
    assert!(
        !is_implemented_node(&multi),
        "multiple axes not implemented"
    );

    let axes_input = node("ReduceSum", &["x", "axes"], &["out"], &[]);
    assert!(
        !is_implemented_node(&axes_input),
        "axes as input, unresolved: not implemented"
    );
}

/// The form with axes as input (`ReduceSum` from opset 13, the others from 18)
/// becomes supported **after** canonicalization, which promotes the constant
/// input to an attribute. This is the rfdetr case, where the axes live in an
/// initializer.
#[test]
fn constant_axes_input_is_folded_into_the_attribute() {
    let initializers = HashMap::from([(
        "axes".to_string(),
        InitializerIr {
            dtype: INT64,
            shape: vec![1],
            data: 1i64.to_le_bytes().to_vec(),
        },
    )]);

    let mut n = node("ReduceSum", &["x", "axes"], &["out"], &[]);
    assert!(!is_implemented_node(&n), "before canonicalization");

    fold_constant_params(&mut n, &initializers);
    assert_eq!(n.attrs.get("axes"), Some(&AttrValue::Ints(vec![1])));
    assert_eq!(
        n.inputs,
        vec!["x".to_string()],
        "the input must be consumed"
    );
    assert!(is_implemented_node(&n), "after canonicalization");

    // and the canonicalized node actually runs
    let shape = vec![2i64, 6, 7];
    let x = pseudo(84, 31);
    let ir = GraphIr {
        nodes: vec![n],
        initializers,
        inputs: vec!["x".to_string()],
        outputs: vec!["out".to_string()],
    };
    let (got, out_shape) = run(&ir, HostTensor::from_f32(shape.clone(), &x), "out");
    assert_eq!(out_shape, vec![2, 1, 7]);
    assert_close(&got, &reduce_ref(&x, &shape, 1, "sum"));
}

/// Axes can also come from a `Constant` node, not just from an initializer:
/// this happens when ORT optimizations are disabled and the constants have
/// not been folded.
#[test]
fn axes_from_a_constant_node_are_also_folded() {
    let mut producer = node("Constant", &[], &["axes"], &[]);
    producer.attrs.insert(
        "value".to_string(),
        AttrValue::Tensor(InitializerIr {
            dtype: INT64,
            shape: vec![1],
            data: (-1i64).to_le_bytes().to_vec(),
        }),
    );
    let mut consumer = node("ReduceMax", &["x", "axes"], &["out"], &[]);

    let constants = constant_outputs(&[producer]);
    fold_constant_params(&mut consumer, &constants);
    assert_eq!(consumer.attrs.get("axes"), Some(&AttrValue::Ints(vec![-1])));
    assert!(is_implemented_node(&consumer));
}

/// An int64 input goes to the host: the GPU kernel is f32-only, but the
/// support check does not see dtypes, so the node is still claimed. This is
/// the case that used to drop rfdetr — `ReduceSum` on the int64 mask of the
/// transformer tail.
#[test]
fn integer_input_is_reduced_on_host() {
    let values: Vec<i64> = (0..12).collect();
    let ir = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["out"],
            &[
                ("axes", AttrValue::Ints(vec![1])),
                ("keepdims", AttrValue::Int(0)),
            ],
        )],
        "out",
    );

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(HostTensor::from_i64(vec![3, 4], &values)));
    execute(&ir, &mut env).expect("graph execution");

    assert!(!env.on_device("out"), "an int64 must not end up in VRAM");
    let out = env.host("out").expect("output on host");
    assert_eq!(out.dtype, INT64, "the input dtype must be preserved");
    assert_eq!(out.shape, vec![3]);
    // rows 0..3, 4..7, and 8..11
    assert_eq!(out.to_i64().expect("i64"), vec![6, 22, 38]);
    env.finish();
}

/// A non-constant input cannot be resolved: the node stays unclaimed, and
/// the block splits there instead of failing at runtime.
#[test]
fn non_constant_axes_input_stays_unclaimed() {
    let mut n = node("ReduceSum", &["x", "axes"], &["out"], &[]);
    fold_constant_params(&mut n, &HashMap::new());
    assert!(!n.attrs.contains_key("axes"));
    assert!(!is_implemented_node(&n));
}
