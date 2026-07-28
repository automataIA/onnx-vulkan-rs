//! Compiling EP: fuses encoder subgraph into single block executed
//! by single Vulkan command buffer (1 upload / 1 submit / 1 download), instead of
//! per-node kernel-registry model (which syncs across every CPU↔GPU boundary).
//! See `cronologia.md` for diagnosis.
//!
//! Optional path enabled with `VULKAN_EP_COMPILE=1`; by default EP still uses
//! the kernel registry (validated behavior unchanged).

pub mod graph_ir;
mod ort_io;

use crate::ort_util::{error_status, to_status};
use anyhow::Result;
use onnx_vulkan_core::Executor;
pub use onnx_vulkan_core::fusion;
use ort_ep_sys as sys;
use std::ffi::c_void;
use std::ptr;

/// Canonicalized IR of graph nodes in `nodes` order.
///
/// Canonicalization (`fold_constant_params`) moves parameters moved to inputs
/// back into attributes — reduction `axes` —
/// resolving them from initializers and `Constant` nodes. This must be done
/// **here**, before support checking, because in isolated nodes that value is unknown.
/// The same transformation is applied by `graph_ir::extract` before execution,
/// so capability and execution cannot diverge.
///
/// # Safety
/// `graph` and `nodes` valid for the call duration.
pub unsafe fn canonical_nodes(
    graph: *const sys::OrtGraph,
    nodes: &[*const sys::OrtNode],
) -> Result<Vec<graph_ir::NodeIr>> {
    let api = crate::ort_util::apis().ort;
    let mut irs = Vec::with_capacity(nodes.len());
    for &node in nodes {
        irs.push(unsafe { graph_ir::extract_node(api, node)? });
    }
    let mut constants = onnx_vulkan_core::constant_outputs(&irs);
    constants.extend(unsafe { graph_ir::extract_initializers(graph)? });
    for ir in &mut irs {
        graph_ir::fold_constant_params(ir, &constants);
    }
    Ok(irs)
}

/// Nodes that the compiling EP can fuse and execute (delegates to the interpreter,
/// the single source of truth on coverage). The check is on the **node**, not
/// on the op name: attributes decide whether the kernel can handle it.
pub fn is_fusible_node(node: &graph_ir::NodeIr) -> bool {
    onnx_vulkan_core::is_implemented_node(node)
}

/// Enables the compiling EP path (`VULKAN_EP_COMPILE=1`).
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("VULKAN_EP_COMPILE")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    })
}

/// `OrtNodeComputeInfo` owned by the plugin: `base` first (`#[repr(C)]`),
/// plus the executor of the fused subgraph.
#[repr(C)]
struct VulkanNodeComputeInfo {
    base: sys::OrtNodeComputeInfo,
    /// Graph and session resources of the subgraph: pipelines and packed
    /// weights live as long as the session, so WGSL compilation and B packing
    /// happen only once.
    executor: Executor<'static>,
    /// Value names of the fused node's inputs/outputs (ORT binding order).
    boundary_inputs: Vec<String>,
    boundary_outputs: Vec<String>,
}

/// `OrtEp::Compile` callback: extracts the IR of each fused subgraph and
/// returns an `OrtNodeComputeInfo` with `CreateState`/`Compute`/`ReleaseState`.
///
/// # Safety
/// Called by ORT under the Plugin EP API contract.
pub unsafe extern "C" fn compile(
    _this: *mut sys::OrtEp,
    graphs: *mut *const sys::OrtGraph,
    fused_nodes: *mut *const sys::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut sys::OrtNodeComputeInfo,
    _ep_context_nodes: *mut *mut sys::OrtNode,
) -> sys::OrtStatusPtr {
    to_status(unsafe { compile_impl(graphs, fused_nodes, count, node_compute_infos) })
}

unsafe fn compile_impl(
    graphs: *mut *const sys::OrtGraph,
    fused_nodes: *mut *const sys::OrtNode,
    count: usize,
    node_compute_infos: *mut *mut sys::OrtNodeComputeInfo,
) -> Result<()> {
    unsafe {
        let graphs = std::slice::from_raw_parts(graphs, count);
        let fused = std::slice::from_raw_parts(fused_nodes, count);
        for (i, &graph) in graphs.iter().enumerate() {
            let ir = graph_ir::extract(graph)?;
            let (boundary_inputs, boundary_outputs) = graph_ir::fused_node_io(fused[i])?;
            log::info!(
                "VulkanEP compile: subgraph {i} — {} nodes, {} initializers, \
                 {} boundary inputs, {} boundary outputs",
                ir.nodes.len(),
                ir.initializers.len(),
                boundary_inputs.len(),
                boundary_outputs.len()
            );
            // the interpreter frees an intermediate at its last reader and
            // keeps only the graph outputs alive: if ORT asked the fused node
            // for a value the subgraph does not declare as output, liveness
            // would have already recycled it
            for name in boundary_outputs.iter().filter(|n| !n.is_empty()) {
                anyhow::ensure!(
                    ir.outputs.contains(name),
                    "boundary output '{name}' is not a subgraph output"
                );
            }
            let mut base: sys::OrtNodeComputeInfo = std::mem::zeroed();
            base.ort_version_supported = sys::ORT_API_VERSION;
            base.CreateState = Some(create_state);
            base.Compute = Some(compute);
            base.ReleaseState = Some(release_state);
            // `Executor::new` rejects an unimplemented node: if
            // `GetCapability` claimed something the interpreter cannot
            // execute, it is discovered here and not midway through the first inference
            let executor = Executor::new(crate::vk::context()?, ir)?;
            let boxed = Box::new(VulkanNodeComputeInfo {
                base,
                executor,
                boundary_inputs,
                boundary_outputs,
            });
            *node_compute_infos.add(i) = Box::into_raw(boxed).cast::<sys::OrtNodeComputeInfo>();
        }
        Ok(())
    }
}

/// Releases the `OrtNodeComputeInfo` allocated by [`compile`].
///
/// # Safety
/// Called by ORT with the same pointers produced by `Compile`.
pub unsafe extern "C" fn release_node_compute_infos(
    _this: *mut sys::OrtEp,
    node_compute_infos: *mut *mut sys::OrtNodeComputeInfo,
    count: usize,
) {
    unsafe {
        for i in 0..count {
            let p = *node_compute_infos.add(i);
            if !p.is_null() {
                drop(Box::from_raw(p.cast::<VulkanNodeComputeInfo>()));
            }
        }
    }
}

/// Compute state for a subgraph: pointer to the `VulkanNodeComputeInfo`
/// (lives for the whole session), from which the interpreter reads IR and boundaries.
struct ComputeState {
    nci: *const VulkanNodeComputeInfo,
}

unsafe extern "C" fn create_state(
    this_ptr: *mut sys::OrtNodeComputeInfo,
    _ctx: *mut sys::OrtNodeComputeContext,
    compute_state: *mut *mut c_void,
) -> sys::OrtStatusPtr {
    let nci = this_ptr.cast::<VulkanNodeComputeInfo>().cast_const();
    let state = Box::new(ComputeState { nci });
    unsafe { *compute_state = Box::into_raw(state).cast::<c_void>() };
    ptr::null_mut()
}

unsafe extern "C" fn release_state(_this: *mut sys::OrtNodeComputeInfo, state: *mut c_void) {
    if !state.is_null() {
        drop(unsafe { Box::from_raw(state.cast::<ComputeState>()) });
    }
}

unsafe extern "C" fn compute(
    _this: *mut sys::OrtNodeComputeInfo,
    compute_state: *mut c_void,
    kernel_context: *mut sys::OrtKernelContext,
) -> sys::OrtStatusPtr {
    let state = unsafe { &*compute_state.cast::<ComputeState>() };
    let nci = unsafe { &*state.nci };
    match unsafe {
        ort_io::run(
            &nci.executor,
            &nci.boundary_inputs,
            &nci.boundary_outputs,
            kernel_context,
        )
    } {
        Ok(()) => ptr::null_mut(),
        Err(e) => {
            log::error!("VulkanEP compile/compute: {e:#}");
            error_status(&format!("{e:#}"))
        }
    }
}
