//! Load-time refusals on opset and operand type.
//!
//! Coverage used to be decided on the op name and its attributes alone, and
//! that let two classes through. A node from an opset whose contract moved
//! under the kernel was claimed as if it had not: `Pad` takes `pads` from
//! `inputs[1]`, which before opset 11 is an attribute and no input at all. And
//! a node whose *operands* are of a type the kernel cannot read was claimed
//! too: a `uint8` `MaxPool` went to the float kernel and answered with
//! reinterpreted bytes — `max|Δ| = 8.086`, argmax 489 → 611 — wrong, silent,
//! and caught only because that model ships golden data.
//!
//! Neither check needs a GPU: both decide on the IR.

use onnx_vulkan_core::{MAX_OPSET, NodeIr, is_implemented_node, unsupported_dtype};
use std::collections::HashMap;

fn node(op: &str, opset: i32, domain: &str, inputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: domain.to_string(),
        op: op.to_string(),
        opset,
        name: format!("{op}_0"),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: vec!["out".to_string()],
        attrs: HashMap::new(),
    }
}

/// `Pad`, `Slice`, `TopK` and `Resize` read a form that only exists from a
/// given opset on. Below it the kernel would read an input that is not there,
/// or default an attribute that does not exist to the wrong value.
#[test]
fn opset_floor_refuses_the_pre_migration_form() {
    for (op, first_good, inputs) in [
        ("Pad", 11, &["x", "pads"][..]),
        ("Slice", 10, &["x", "starts", "ends"][..]),
        ("TopK", 10, &["x", "k"][..]),
        ("Resize", 11, &["x", "roi", "scales"][..]),
    ] {
        assert!(
            !is_implemented_node(&node(op, first_good - 1, "", inputs)),
            "{op} at opset {} must be refused: the attribute form",
            first_good - 1
        );
        assert!(
            is_implemented_node(&node(op, first_good, "", inputs)),
            "{op} at opset {first_good} must be claimed"
        );
    }
}

/// An op whose contract never migrated has no floor: opset 1 is fine.
#[test]
fn ops_without_a_migration_have_no_floor() {
    for op in ["Relu", "Mul", "Transpose", "Concat"] {
        assert!(
            is_implemented_node(&node(op, 1, "", &["x"])),
            "{op} at opset 1 must be claimed"
        );
    }
}

/// Above the reviewed ceiling every standard op is refused, because nobody has
/// read the kernels against that spec. The statement is about us, not the model.
#[test]
fn opset_ceiling_refuses_an_unreviewed_spec() {
    for op in ["Relu", "MatMul", "Conv", "Softmax"] {
        assert!(
            is_implemented_node(&node(op, MAX_OPSET, "", &["x"])),
            "{op} at the ceiling must be claimed"
        );
        assert!(
            !is_implemented_node(&node(op, MAX_OPSET + 1, "", &["x"])),
            "{op} above the ceiling must be refused"
        );
    }
}

/// Contrib ops live on their own version axis at 1, where the `ai.onnx` bounds
/// are meaningless — applying them would refuse every model that has one.
#[test]
fn contrib_domain_is_exempt_from_the_ai_onnx_bounds() {
    let n = node(
        "SimplifiedLayerNormalization",
        1,
        "com.microsoft",
        &["x", "w"],
    );
    assert!(
        is_implemented_node(&n),
        "a contrib op at domain version 1 must not be refused by the ai.onnx floor"
    );
    // and the ceiling does not reach it either, whatever version it declares
    let n = node(
        "SimplifiedLayerNormalization",
        MAX_OPSET + 5,
        "com.microsoft",
        &["x", "w"],
    );
    assert!(is_implemented_node(&n));
}

/// An opset of 0 means the producer could not resolve one. Unknown is not
/// wrong: refusing here would reject working models to guard a hypothetical.
#[test]
fn unresolved_opset_is_allowed() {
    assert!(is_implemented_node(&node("Relu", 0, "", &["x"])));
    assert!(
        is_implemented_node(&node("Pad", 0, "", &["x", "pads"])),
        "even an op with a floor: 0 is 'unknown', not 'old'"
    );
}

/// A non-float operand on a kernel that has no integer path.
///
/// `MaxPool` is deliberately *not* the example any more: the incident that
/// motivated this check was fixed in `pool()`, which now dispatches
/// `MaxPool_q` on packed bytes. Using it here would test a rule that no longer
/// holds — see `ops_that_are_not_float_only_are_untouched`, where it belongs.
#[test]
fn a_non_float_operand_is_refused() {
    let n = node("Softmax", 12, "", &["x"]);
    let float: HashMap<String, i32> = [("x".to_string(), 1)].into();
    let uint8: HashMap<String, i32> = [("x".to_string(), 2)].into();

    assert_eq!(unsupported_dtype(&n, &float), None, "f32 is what it reads");
    let reason = unsupported_dtype(&n, &uint8).expect("a non-float input must be refused");
    assert!(
        reason.contains("Softmax") && reason.contains("2"),
        "the message must name the op and the element type, got: {reason}"
    );
}

/// A value whose type nobody resolved is allowed through, on the same
/// principle as an unresolved opset: this check refuses what is known to be
/// wrong, never what is merely unknown.
#[test]
fn an_unknown_operand_type_is_allowed() {
    let n = node("Softmax", 12, "", &["x"]);
    assert_eq!(unsupported_dtype(&n, &HashMap::new()), None);
}

/// Ops outside the table are unconstrained: `Cast` exists to change the type,
/// and the integer kernels read integers on purpose.
#[test]
fn ops_that_are_not_float_only_are_untouched() {
    let uint8: HashMap<String, i32> = [("x".to_string(), 2)].into();
    for op in [
        "Cast",
        "MatMulInteger",
        "ConvInteger",
        "Reshape",
        "Gather",
        "MaxPool",
        "CumSum",
    ] {
        assert_eq!(
            unsupported_dtype(&node(op, 12, "", &["x"]), &uint8),
            None,
            "{op} must not be constrained to f32"
        );
    }
}
