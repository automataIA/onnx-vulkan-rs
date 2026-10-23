//! Reduction over one or more axes: `ReduceMean`, `ReduceSum`, `ReduceMax`.
//!
//! The case that matters is `axes = [1]`, `keepdims = 1` on NCHW — the form
//! rfdetr uses to decompose the projector normalizations, which alone split
//! the graph in 18 places. Each test compares the kernel against a CPU
//! reference written here.

use onnx_vulkan_core::host_ops::{BOOL, FLOAT, HostTensor, INT8, INT32, INT64, UINT8};
use onnx_vulkan_core::{
    AttrValue, ElementType, ExecutionEnv, GraphIr, InitializerIr, KernelCache, NodeIr, Tensor,
    constant_outputs, execute, fold_constant_params, is_implemented_node, unsupported_dtype,
};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 13,
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
        ..Default::default()
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
                "min" => vals.fold(f32::INFINITY, f32::min),
                "sum" => vals.sum(),
                _ => vals.sum::<f32>() / c as f32,
            }
        })
        .collect()
}

fn reduce_ref_axes(x: &[f32], shape: &[i64], axes: &[usize], kind: &str) -> (Vec<f32>, Vec<i64>) {
    let mut values = x.to_vec();
    let mut shape = shape.to_vec();
    let mut axes = axes.to_vec();
    axes.sort_unstable_by(|a, b| b.cmp(a));
    for axis in axes {
        values = reduce_ref(&values, &shape, axis, kind);
        shape.remove(axis);
    }
    (values, shape)
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

#[test]
fn reductions_accept_multiple_axes() {
    let shape = vec![2i64, 3, 4];
    let x = pseudo(24, 29);
    for (op, kind) in [
        ("ReduceSum", "sum"),
        ("ReduceMean", "mean"),
        ("ReduceMax", "max"),
        ("ReduceMin", "min"),
    ] {
        let ir = graph(
            vec![node(
                op,
                &["x"],
                &["out"],
                &[
                    ("axes", AttrValue::Ints(vec![0, -1])),
                    ("keepdims", AttrValue::Int(0)),
                ],
            )],
            "out",
        );
        let (got, out_shape) = run(&ir, HostTensor::from_f32(shape.clone(), &x), "out");
        let (want, want_shape) = reduce_ref_axes(&x, &shape, &[0, 2], kind);
        assert_eq!(out_shape, want_shape, "{op}");
        assert_close(&got, &want);
    }
}

#[test]
fn omitted_axes_reduce_all_dimensions_to_a_scalar() {
    let shape = vec![2i64, 3, 4];
    let x = pseudo(24, 37);
    let ir = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["out"],
            &[("keepdims", AttrValue::Int(0))],
        )],
        "out",
    );
    let (got, out_shape) = run(&ir, HostTensor::from_f32(shape.clone(), &x), "out");
    let (want, want_shape) = reduce_ref_axes(&x, &shape, &[0, 1, 2], "sum");
    assert!(
        out_shape.is_empty(),
        "expected scalar shape, got {out_shape:?}"
    );
    assert_eq!(out_shape, want_shape);
    assert_close(&got, &want);
}

#[test]
fn explicit_empty_axes_reduce_all_unless_noop_is_requested() {
    let shape = vec![2i64, 3];
    let x = pseudo(6, 41);

    let reduce_all = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["out"],
            &[
                ("axes", AttrValue::Ints(vec![])),
                ("keepdims", AttrValue::Int(0)),
            ],
        )],
        "out",
    );
    let (got, out_shape) = run(&reduce_all, HostTensor::from_f32(shape.clone(), &x), "out");
    assert!(out_shape.is_empty());
    assert_close(&got, &[x.iter().sum()]);

    let noop = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["out"],
            &[
                ("axes", AttrValue::Ints(vec![])),
                ("noop_with_empty_axes", AttrValue::Int(1)),
            ],
        )],
        "out",
    );
    let (got, out_shape) = run(&noop, HostTensor::from_f32(shape.clone(), &x), "out");
    assert_eq!(out_shape, shape);
    assert_eq!(got, x);
}

#[test]
fn scalar_reduction_is_an_identity() {
    let x = [2.5f32];
    let ir = graph(vec![node("ReduceMean", &["x"], &["out"], &[])], "out");
    let (got, out_shape) = run(&ir, HostTensor::from_f32(vec![], &x), "out");
    assert!(out_shape.is_empty());
    assert_eq!(got, x);
}

/// Constant axes are handled after canonicalization. A dynamic axes input is
/// still unresolved and must stay outside the supported slice.
#[test]
fn supported_and_unresolved_forms_are_claimed_correctly() {
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
    assert!(is_implemented_node(&multi), "multiple axes are implemented");

    let all = node("ReduceSum", &["x"], &["out"], &[]);
    assert!(
        is_implemented_node(&all),
        "omitted axes reduce all dimensions"
    );

    let noop = node(
        "ReduceMean",
        &["x"],
        &["out"],
        &[
            ("axes", AttrValue::Ints(vec![])),
            ("noop_with_empty_axes", AttrValue::Int(1)),
        ],
    );
    assert!(
        is_implemented_node(&noop),
        "empty-axis no-op is implemented"
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
        ..Default::default()
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

#[test]
fn integer_multi_axis_mean_divides_once_after_accumulating() {
    use onnx_vulkan_core::host_ops::{RedOp, reduce_axes};

    let input = HostTensor::from_i64(vec![2, 2], &[0, 1, 1, 2]);
    let output = reduce_axes(&input, &[0, 1], RedOp::Mean, false).expect("integer ReduceMean");
    assert_eq!(output.shape, Vec::<i64>::new());
    assert_eq!(output.to_i64().expect("scalar mean"), vec![1]);
}

#[test]
fn empty_integer_sum_returns_zero_bins_and_sum_wraps_without_panicking() {
    use onnx_vulkan_core::host_ops::{RedOp, reduce_axes};

    let empty = HostTensor::from_i64(vec![2, 0, 3], &[]);
    let output = reduce_axes(&empty, &[1], RedOp::Sum, false).expect("empty ReduceSum");
    assert_eq!(output.shape, vec![2, 3]);
    assert_eq!(output.to_i64().expect("zero sums"), vec![0; 6]);

    let overflow = HostTensor::from_i64(vec![2], &[i64::MAX, 1]);
    let output = reduce_axes(&overflow, &[0], RedOp::Sum, false).expect("wrapping ReduceSum");
    assert_eq!(output.to_i64().expect("wrapped sum"), vec![i64::MIN]);
}

#[test]
fn empty_integer_extrema_use_dtype_specific_identities() {
    use onnx_vulkan_core::host_ops::{INT8, RedOp, UINT8, reduce_axes};

    let empty_i8 = HostTensor::new(INT8, vec![2, 0], vec![]);
    let max = reduce_axes(&empty_i8, &[1], RedOp::Max, false).expect("empty int8 max");
    assert_eq!(max.data, vec![i8::MIN as u8; 2]);

    let empty_u8 = HostTensor::new(UINT8, vec![2, 0], vec![]);
    let min = reduce_axes(&empty_u8, &[1], RedOp::Min, false).expect("empty uint8 min");
    assert_eq!(min.data, vec![u8::MAX; 2]);
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

#[test]
fn reduction_load_time_dtype_guard_accepts_supported_and_refuses_float64_float16() {
    let reductions = ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"];
    let supported = [FLOAT, BOOL, UINT8, INT8, INT32, INT64];
    for op in reductions {
        let n = node(op, &["x"], &["out"], &[]);
        for dtype in supported {
            let types = HashMap::from([(String::from("x"), dtype)]);
            assert_eq!(
                unsupported_dtype(&n, &types),
                None,
                "{op} should accept element type {dtype}"
            );
        }
        for dtype in [ElementType::Float16 as i32, ElementType::Float64 as i32] {
            let types = HashMap::from([(String::from("x"), dtype)]);
            assert!(
                unsupported_dtype(&n, &types).is_some(),
                "{op} should refuse unsupported element type {dtype}"
            );
        }
        assert_eq!(
            unsupported_dtype(&n, &HashMap::new()),
            None,
            "unresolved dtype stays allowed for {op}"
        );
    }
}

#[test]
fn absent_axes_noop_preserves_shape_and_default_keepdims_retains_rank() {
    let x = HostTensor::from_f32(vec![2, 2], &[1., 2., 3., 4.]);
    let ir = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["out"],
            &[("noop_with_empty_axes", AttrValue::Int(1))],
        )],
        "out",
    );
    let (got, shape) = run(&ir, x.clone(), "out");
    assert_eq!(shape, vec![2, 2]);
    assert_eq!(got, vec![1., 2., 3., 4.]);
    let ir = graph(vec![node("ReduceSum", &["x"], &["out"], &[])], "out");
    let (got, shape) = run(&ir, x, "out");
    assert_eq!(shape, vec![1, 1]);
    assert_eq!(got, vec![10.]);
}

#[test]
fn empty_float_sum_and_mean_follow_empty_reduction_semantics() {
    for (op, nan) in [("ReduceSum", false), ("ReduceMean", true)] {
        let ir = graph(
            vec![node(
                op,
                &["x"],
                &["out"],
                &[
                    ("axes", AttrValue::Ints(vec![1])),
                    ("keepdims", AttrValue::Int(0)),
                ],
            )],
            "out",
        );
        let (got, shape) = run(&ir, HostTensor::from_f32(vec![2, 0, 3], &[]), "out");
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(got.len(), 6);
        assert!(got.iter().all(|v| if nan { v.is_nan() } else { *v == 0. }));
    }
}
