//! GPU-free graph compatibility checks shared by the runtime and tooling.

use crate::{GraphIr, NodeIr, interp};

/// Version of the machine-readable compatibility report.
pub const COMPATIBILITY_SCHEMA_VERSION: u32 = 1;

/// One reason a node may be refused by the Vulkan interpreter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatibilityIssue {
    /// Stable node name when the model supplies one. Empty names are retained
    /// here because that is the exact identity available in the graph IR.
    pub node: String,
    pub op: String,
    /// `execution`, `quantization`, or `dtype`.
    pub kind: String,
    pub reason: String,
}

/// The complete result of the checks performed before an executor allocates a
/// Vulkan resource. Unknown metadata is deliberately not an issue: dynamic
/// shapes and unresolved dtypes are valid at runtime and are checked against
/// the concrete tensors when a run is made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompatibilityReport {
    pub schema_version: u32,
    pub compatible: bool,
    pub total_nodes: usize,
    pub supported_nodes: usize,
    pub issues: Vec<CompatibilityIssue>,
}

impl CompatibilityReport {
    pub fn execution_coverage(&self) -> (usize, usize) {
        (self.supported_nodes, self.total_nodes)
    }
}

/// Runs the exact node, quantization, and dtype predicates used by
/// [`crate::Executor`], without creating a Vulkan context or allocating GPU
/// memory. Callers that also perform load-time rewrites should call
/// [`crate::prepare_graph`] first, just as the executor does.
pub fn compatibility_report(graph: &GraphIr) -> CompatibilityReport {
    let mut issues = Vec::new();
    let mut supported_nodes = 0;
    for node in &graph.nodes {
        let mut node_supported = true;
        if !interp::is_implemented_node(node) {
            node_supported = false;
            issues.push(issue(
                node,
                "execution",
                format!(
                    "operator or operator form is not implemented by the interpreter (domain `{}`, opset {})",
                    if node.domain.is_empty() { "ai.onnx" } else { &node.domain },
                    node.opset
                ),
            ));
        }
        if let Some(reason) = interp::unsupported_quantization(node, &graph.initializers) {
            node_supported = false;
            issues.push(issue(node, "quantization", reason));
        }
        if let Some(reason) = interp::unsupported_dtype(node, &graph.value_types) {
            node_supported = false;
            issues.push(issue(node, "dtype", reason));
        }
        if node_supported {
            supported_nodes += 1;
        }
    }
    CompatibilityReport {
        schema_version: COMPATIBILITY_SCHEMA_VERSION,
        compatible: issues.is_empty(),
        total_nodes: graph.nodes.len(),
        supported_nodes,
        issues,
    }
}

fn issue(node: &NodeIr, kind: &str, reason: impl Into<String>) -> CompatibilityIssue {
    CompatibilityIssue {
        node: node.name.clone(),
        op: node.op.clone(),
        kind: kind.to_owned(),
        reason: reason.into(),
    }
}
