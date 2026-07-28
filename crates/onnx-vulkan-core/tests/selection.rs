//! Selection ops of the rfdetr transformer tail: `GatherElements`,
//! `ScatterND`, `TopK`, `Greater`.
//!
//! All four run on the host: the tensors involved are small and three out of
//! four are int64. The cost we paid was not the computation but the splitting
//! of the convex block, so covering them is enough to close the fragmentation.
//!
//! The expected values come from `onnx.reference.ReferenceEvaluator`
//! (opset 17), not hand-written.

use onnx_vulkan_core::host_ops::{self, BOOL, CmpOp, FLOAT, HostTensor, INT64};
use onnx_vulkan_core::{AttrValue, NodeIr, is_implemented_node};

fn node(op: &str, attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        since_version: 17,
        name: format!("{op}_0"),
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
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

#[test]
fn gather_elements_axis_1() {
    const DATA: [f32; 30] = [
        -0.801931, -1.32436, -0.248362, 0.420445, 1.13605, 0.109706, -0.552647, -0.78478, 0.748746,
        1.63478, 0.272769, -1.23333, -0.958265, 1.60002, 0.202882, -1.73213, -0.0836962, -1.16323,
        -0.629288, -0.488006, -0.713313, 0.553378, -0.063086, -0.589431, 0.409638, 0.829855,
        -1.64302, -0.25673, -0.980747, -0.173155,
    ];
    const IDX: [i64; 24] = [
        3, 3, 3, 0, 0, 2, 1, 2, 4, 1, 2, 1, 1, 4, 0, 1, 0, 0, 1, 3, 2, 2, 0, 3,
    ];
    const WANT: [f32; 24] = [
        1.63478, 0.272769, -1.23333, -0.801931, -1.32436, 0.748746, 0.420445, -0.78478, 0.202882,
        0.420445, -0.78478, 0.109706, -0.629288, -0.980747, -1.16323, -0.629288, -0.0836962,
        -1.16323, -0.629288, 0.829855, -0.589431, 0.553378, -0.0836962, -1.64302,
    ];

    let out = host_ops::gather_elements(
        &HostTensor::from_f32(vec![2, 5, 3], &DATA),
        &HostTensor::from_i64(vec![2, 4, 3], &IDX),
        1,
    )
    .expect("GatherElements");
    assert_eq!(
        out.shape,
        vec![2, 4, 3],
        "the output has the shape of the indices"
    );
    assert_close(&out.to_f32().expect("f32"), &WANT);
}

/// Negative axis and negative indices: both must be normalized against the rank.
#[test]
fn gather_elements_normalizes_negative_axis_and_indices() {
    let data = HostTensor::from_f32(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let idx = HostTensor::from_i64(vec![2, 2], &[-1, 0, -3, 2]);
    let out = host_ops::gather_elements(&data, &idx, -1).expect("GatherElements");
    assert_eq!(out.to_f32().expect("f32"), vec![2.0, 0.0, 3.0, 5.0]);
}

#[test]
fn gather_elements_rejects_out_of_range_index() {
    let data = HostTensor::from_f32(vec![2, 2], &[0.0, 1.0, 2.0, 3.0]);
    let idx = HostTensor::from_i64(vec![2, 2], &[0, 2, 0, 0]);
    assert!(host_ops::gather_elements(&data, &idx, 1).is_err());
}

#[test]
fn scatter_nd_overwrites_addressed_slices() {
    const DATA: [f32; 12] = [
        -1.99782, 0.272129, -1.10172, 0.0330572, 0.043632, -1.98843, -0.233423, -0.25579, 0.962001,
        -1.18145, 0.738042, -1.09897,
    ];
    const UPD: [f32; 8] = [
        -0.331291, -0.840473, 1.44873, 0.568213, 2.43173, 0.641916, 0.844993, 0.840683,
    ];
    const WANT: [f32; 12] = [
        -0.331291, -0.840473, 1.44873, 0.568213, 0.043632, -1.98843, -0.233423, -0.25579, 2.43173,
        0.641916, 0.844993, 0.840683,
    ];

    let out = host_ops::scatter_nd(
        &HostTensor::from_f32(vec![3, 4], &DATA),
        &HostTensor::from_i64(vec![2, 1], &[0, 2]),
        &HostTensor::from_f32(vec![2, 4], &UPD),
    )
    .expect("ScatterND");
    assert_eq!(out.shape, vec![3, 4], "the shape is that of data");
    assert_close(&out.to_f32().expect("f32"), &WANT);
}

/// The rfdetr form: int64, `k` equal to the rank, so single elements are
/// written instead of slices.
#[test]
fn scatter_nd_writes_single_elements_when_k_equals_rank() {
    let out = host_ops::scatter_nd(
        &HostTensor::from_i64(vec![1, 2], &[7, 9]),
        &HostTensor::from_i64(vec![1, 2], &[0, 1]),
        &HostTensor::from_i64(vec![1], &[42]),
    )
    .expect("ScatterND");
    assert_eq!(out.dtype, INT64);
    assert_eq!(out.to_i64().expect("i64"), vec![7, 42]);
}

#[test]
fn scatter_nd_rejects_mismatched_updates() {
    let err = host_ops::scatter_nd(
        &HostTensor::from_f32(vec![3, 4], &[0.0; 12]),
        &HostTensor::from_i64(vec![2, 1], &[0, 2]),
        &HostTensor::from_f32(vec![4], &[0.0; 4]),
    );
    assert!(err.is_err(), "updates must cover all indexed slices");
}

const TK_X: [f32; 14] = [
    -0.606612, -0.0700284, 1.35039, -0.396551, 0.1888, -0.0212235, 0.609217, -0.364909, -0.152362,
    0.242381, 0.103023, -0.864973, 0.895783, -1.29848,
];

#[test]
fn top_k_largest_on_axis_1() {
    const WANT_V: [f32; 6] = [1.35039, 0.609217, 0.1888, 0.895783, 0.242381, 0.103023];
    const WANT_I: [i64; 6] = [2, 6, 4, 5, 2, 3];

    let (values, indices) =
        host_ops::top_k(&HostTensor::from_f32(vec![2, 7], &TK_X), 3, 1, true).expect("TopK");
    assert_eq!(values.shape, vec![2, 3]);
    assert_eq!(values.dtype, FLOAT);
    assert_close(&values.to_f32().expect("f32"), &WANT_V);
    assert_eq!(indices.dtype, INT64);
    assert_eq!(indices.to_i64().expect("i64"), WANT_I.to_vec());
}

/// Axis 0: rows are not contiguous (`inner = 7`), which is the case where it
/// is easy to get strides wrong.
#[test]
fn top_k_smallest_on_axis_0() {
    const WANT_V: [f32; 7] = [
        -0.606612, -0.152362, 0.242381, -0.396551, -0.864973, -0.0212235, -1.29848,
    ];
    const WANT_I: [i64; 7] = [0, 1, 1, 0, 1, 0, 1];

    let (values, indices) =
        host_ops::top_k(&HostTensor::from_f32(vec![2, 7], &TK_X), 1, 0, false).expect("TopK");
    assert_eq!(values.shape, vec![1, 7]);
    assert_close(&values.to_f32().expect("f32"), &WANT_V);
    assert_eq!(indices.to_i64().expect("i64"), WANT_I.to_vec());
}

/// Ties follow the lowest index, as in the ONNX reference.
#[test]
fn top_k_breaks_ties_by_lowest_index() {
    let (_, indices) = host_ops::top_k(
        &HostTensor::from_f32(vec![1, 4], &[1.0, 1.0, 1.0, 0.0]),
        2,
        1,
        true,
    )
    .expect("TopK");
    assert_eq!(indices.to_i64().expect("i64"), vec![0, 1]);
}

#[test]
fn top_k_rejects_k_larger_than_axis() {
    assert!(host_ops::top_k(&HostTensor::from_f32(vec![1, 3], &[0.0; 3]), 4, 1, true).is_err());
}

/// `Greater` goes through the same path as `Equal`/`Less`: ONNX broadcasting,
/// BOOL output, and an integer branch separate from the f32 one. In rfdetr
/// both cases exist — one on f32, one on an int64 mask.
#[test]
fn greater_broadcasts_and_returns_bool() {
    let a = HostTensor::from_f32(vec![2, 3], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let b = HostTensor::from_f32(vec![3], &[2.0, 2.0, 2.0]);
    let out = host_ops::compare(&a, &b, CmpOp::Greater).expect("Greater");
    assert_eq!(out.dtype, BOOL);
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.data, vec![0, 0, 0, 1, 1, 1]);
}

#[test]
fn greater_on_integers_stays_exact() {
    // i64 beyond the 24 bits of f32 mantissa: going through the float branch would give equal values
    let big = 1i64 << 40;
    let a = HostTensor::from_i64(vec![2], &[big + 1, big]);
    let b = HostTensor::from_i64(vec![1], &[big]);
    let out = host_ops::compare(&a, &b, CmpOp::Greater).expect("Greater");
    assert_eq!(out.data, vec![1, 0]);
}

/// `ScatterND` is the only one of the four with a capability constraint:
/// reductions accumulate instead of overwriting and are not implemented.
#[test]
fn scatter_nd_reduction_is_not_claimed() {
    assert!(is_implemented_node(&node("ScatterND", &[])));
    assert!(is_implemented_node(&node(
        "ScatterND",
        &[("reduction", AttrValue::String("none".to_string()))]
    )));
    assert!(!is_implemented_node(&node(
        "ScatterND",
        &[("reduction", AttrValue::String("add".to_string()))]
    )));
}

/// The other three have no rejected forms: if the name is claimed, the node is.
#[test]
fn the_other_three_have_no_rejected_forms() {
    for op in ["TopK", "GatherElements", "Greater"] {
        assert!(is_implemented_node(&node(op, &[])), "{op} not claimed");
    }
}
