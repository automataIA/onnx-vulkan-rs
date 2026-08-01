//! What the QOperator (static int8) family is allowed to claim.
//!
//! Coverage for these ops is decided in two places and both refuse at load
//! time: `is_implemented_node` reads the node (arity, attributes), and
//! `unsupported_quantization` reads the operands, because whether a scale is
//! per-tensor or per-channel — and whether a per-channel zero point is
//! symmetric — is a property of the values and not of the node.
//!
//! The constraints asserted here are the ones measured on the two QOperator
//! models in the matrix (`scripts/qlinear-census.py`): activations `uint8` with
//! per-tensor scalar parameters, weights `int8` with per-channel scale and a
//! zero point that is identically zero, `int32` bias. A model outside that
//! envelope must be refused rather than run on an epilogue that silently drops
//! a term.

use onnx_vulkan_core::{
    AttrValue, InitializerIr, NodeIr, is_implemented_node, unsupported_quantization,
};
use std::collections::HashMap;

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT8: i32 = 3;
const INT32: i32 = 6;

fn node(op: &str, inputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 10,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["out".to_string()],
        attrs: attrs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect(),
    }
}

fn tensor(dtype: i32, shape: &[i64], bytes: Vec<u8>) -> InitializerIr {
    InitializerIr {
        dtype,
        shape: shape.to_vec(),
        data: bytes,
    }
}

/// The inputs of a `QLinearConv` as resnet50-int8 and mobilenetv2-int8 write
/// them: `channels` output channels, symmetric per-channel weights.
fn qconv_params(channels: usize) -> HashMap<String, InitializerIr> {
    HashMap::from([
        ("x_scale".to_string(), tensor(FLOAT, &[], vec![0; 4])),
        ("x_zp".to_string(), tensor(UINT8, &[], vec![128])),
        (
            "w_scale".to_string(),
            tensor(FLOAT, &[channels as i64], vec![0; 4 * channels]),
        ),
        (
            "w_zp".to_string(),
            tensor(INT8, &[channels as i64], vec![0; channels]),
        ),
        ("y_scale".to_string(), tensor(FLOAT, &[], vec![0; 4])),
        ("y_zp".to_string(), tensor(UINT8, &[], vec![7])),
        (
            "bias".to_string(),
            tensor(INT32, &[channels as i64], vec![0; 4 * channels]),
        ),
    ])
}

const QCONV_INPUTS: [&str; 9] = [
    "x", "x_scale", "x_zp", "w", "w_scale", "w_zp", "y_scale", "y_zp", "bias",
];

#[test]
fn the_shape_of_a_real_qoperator_export_is_claimed() {
    let full = node("QLinearConv", &QCONV_INPUTS, &[]);
    assert!(is_implemented_node(&full));
    assert_eq!(unsupported_quantization(&full, &qconv_params(8)), None);

    // the bias is optional in the schema, and its absence is not a refusal
    let no_bias = node("QLinearConv", &QCONV_INPUTS[..8], &[]);
    assert!(is_implemented_node(&no_bias));
    assert_eq!(unsupported_quantization(&no_bias, &qconv_params(8)), None);
}

/// The one constraint that cost a measurement to establish: every
/// `QLinearConv` of both models has `w_zero_point` identically zero, so the
/// `w_zp[c]·Σx` correction term is never written. A model that needs it is
/// refused rather than run without it.
#[test]
fn asymmetric_per_channel_weights_are_refused() {
    let node = node("QLinearConv", &QCONV_INPUTS, &[]);
    let mut params = qconv_params(8);
    params.insert(
        "w_zp".to_string(),
        tensor(INT8, &[8], vec![0, 0, 3, 0, 0, 0, 0, 0]),
    );

    // the node itself is unremarkable: the refusal is in the values
    assert!(is_implemented_node(&node));
    let reason = unsupported_quantization(&node, &params).expect("must be refused");
    assert!(reason.contains("per-channel zero point"), "{reason}");
}

/// A **per-tensor** zero point may be non-zero: that is the case `ConvInteger`
/// already handles, and refusing it would refuse the ordinary asymmetric
/// activation quantization every one of these models uses.
#[test]
fn a_per_tensor_zero_point_may_be_asymmetric() {
    let node = node("QLinearConv", &QCONV_INPUTS, &[]);
    let mut params = qconv_params(8);
    params.insert("w_scale".to_string(), tensor(FLOAT, &[], vec![0; 4]));
    params.insert("w_zp".to_string(), tensor(INT8, &[], vec![9]));
    assert_eq!(unsupported_quantization(&node, &params), None);
}

#[test]
fn per_channel_activation_parameters_are_refused() {
    let node = node("QLinearConv", &QCONV_INPUTS, &[]);
    let mut params = qconv_params(8);
    params.insert("y_scale".to_string(), tensor(FLOAT, &[4], vec![0; 16]));
    let reason = unsupported_quantization(&node, &params).expect("must be refused");
    assert!(reason.contains("per-tensor"), "{reason}");
}

/// A scale produced at runtime cannot be folded into the epilogue at load time,
/// and reading it back mid-graph is what this engine does not do.
#[test]
fn a_runtime_scale_is_refused() {
    let node = node("QLinearConv", &QCONV_INPUTS, &[]);
    let mut params = qconv_params(8);
    params.remove("x_scale");
    let reason = unsupported_quantization(&node, &params).expect("must be refused");
    assert!(reason.contains("not a constant"), "{reason}");
}

#[test]
fn a_float_bias_is_refused() {
    let node = node("QLinearConv", &QCONV_INPUTS, &[]);
    let mut params = qconv_params(8);
    params.insert("bias".to_string(), tensor(FLOAT, &[8], vec![0; 32]));
    let reason = unsupported_quantization(&node, &params).expect("must be refused");
    assert!(reason.contains("int32"), "{reason}");
}

/// `QLinearMatMul` is the op where the two models disagree: resnet50 passes
/// `b_scale` as a scalar, mobilenetv2 as a per-channel vector. Both are
/// claimed, or one of the two models does not load.
#[test]
fn both_forms_of_qlinear_matmul_scale_are_claimed() {
    let inputs = [
        "a", "a_scale", "a_zp", "b", "b_scale", "b_zp", "y_scale", "y_zp",
    ];
    let node = node("QLinearMatMul", &inputs, &[]);
    assert!(is_implemented_node(&node));

    let base = |scale: InitializerIr, zero: InitializerIr| {
        HashMap::from([
            ("a_scale".to_string(), tensor(FLOAT, &[], vec![0; 4])),
            ("a_zp".to_string(), tensor(UINT8, &[], vec![128])),
            ("b_scale".to_string(), scale),
            ("b_zp".to_string(), zero),
            ("y_scale".to_string(), tensor(FLOAT, &[], vec![0; 4])),
            ("y_zp".to_string(), tensor(UINT8, &[], vec![64])),
        ])
    };
    let scalar = base(tensor(FLOAT, &[], vec![0; 4]), tensor(INT8, &[], vec![0]));
    assert_eq!(unsupported_quantization(&node, &scalar), None);
    let per_channel = base(
        tensor(FLOAT, &[16], vec![0; 64]),
        tensor(INT8, &[16], vec![0; 16]),
    );
    assert_eq!(unsupported_quantization(&node, &per_channel), None);
}

/// `channels_last = 1` is NHWC: a different reduction and a different output
/// layout, refused on the node rather than transposed underneath.
#[test]
fn nhwc_global_average_pool_is_not_claimed() {
    let inputs = ["x", "x_scale", "x_zp", "y_scale", "y_zp"];
    assert!(is_implemented_node(&node(
        "QLinearGlobalAveragePool",
        &inputs,
        &[]
    )));
    assert!(!is_implemented_node(&node(
        "QLinearGlobalAveragePool",
        &inputs,
        &[("channels_last", AttrValue::Int(1))]
    )));
}

/// The schema makes the zero points of `QLinearAdd` optional; every export
/// measured carries them, and defaulting a missing one to zero is an
/// assumption nothing would check.
#[test]
fn qlinear_add_wants_every_zero_point() {
    let full = [
        "A", "A_scale", "A_zp", "B", "B_scale", "B_zp", "C_scale", "C_zp",
    ];
    assert!(is_implemented_node(&node("QLinearAdd", &full, &[])));
    assert!(!is_implemented_node(&node("QLinearAdd", &full[..7], &[])));
    let mut missing = full;
    missing[5] = "";
    assert!(!is_implemented_node(&node("QLinearAdd", &missing, &[])));
}
