//! `OrtEp`: execution provider instance used by the session.
//!
//! Claims nodes with registered kernels (MatMulInteger, DynamicQuantizeLinear)
//! via kernel registry; remaining graph stays on CPU EP.

use crate::ort_util::{apis, check, to_status};
use crate::registry::SUPPORTED_OPS;
use anyhow::Result;
use ort_ep_sys as sys;
use std::ffi::{CStr, c_char};
use std::ptr;

#[repr(C)]
pub struct VulkanEp {
    base: sys::OrtEp,
    name: &'static CStr,
    /// Factory that created the EP: owns shared kernel registry.
    factory: *mut crate::factory::VulkanEpFactory,
}

impl VulkanEp {
    pub fn new(name: &'static CStr, factory: *mut crate::factory::VulkanEpFactory) -> Self {
        let mut base: sys::OrtEp = unsafe { std::mem::zeroed() };
        base.ort_version_supported = sys::ORT_API_VERSION;
        base.GetName = Some(get_name);
        base.GetCapability = Some(get_capability);
        base.GetKernelRegistry = Some(get_kernel_registry);
        base.OnRunEnd = Some(on_run_end);
        // Compiling EP (opt-in VULKAN_EP_COMPILE=1): fuses the subgraph into a
        // single command buffer. Harmless when GetCapability fuses nothing.
        base.Compile = Some(crate::compile::compile);
        base.ReleaseNodeComputeInfos = Some(crate::compile::release_node_compute_infos);
        Self {
            base,
            name,
            factory,
        }
    }
}

unsafe extern "C" fn get_name(this_ptr: *const sys::OrtEp) -> *const c_char {
    let ep = unsafe { &*this_ptr.cast::<VulkanEp>() };
    ep.name.as_ptr()
}

/// End of a session Run: prints the profile (if `VULKAN_EP_STATS`).
unsafe extern "C" fn on_run_end(
    _this_ptr: *mut sys::OrtEp,
    _run_options: *const sys::OrtRunOptions,
    _sync_stream: bool,
) -> sys::OrtStatusPtr {
    vk_compute::stats::dump_and_reset();
    ptr::null_mut()
}

/// Subset of ops to claim (debug), from `VULKAN_EP_OPS`.
fn op_filter() -> Option<Vec<String>> {
    static FILTER: std::sync::OnceLock<Option<Vec<String>>> = std::sync::OnceLock::new();
    FILTER
        .get_or_init(|| {
            std::env::var("VULKAN_EP_OPS")
                .ok()
                .map(|s| s.split(',').map(|o| o.trim().to_string()).collect())
        })
        .clone()
}

unsafe extern "C" fn get_capability(
    _this_ptr: *mut sys::OrtEp,
    graph: *const sys::OrtGraph,
    graph_support_info: *mut sys::OrtEpGraphSupportInfo,
) -> sys::OrtStatusPtr {
    to_status(unsafe { get_capability_impl(graph, graph_support_info) })
}

unsafe fn get_capability_impl(
    graph: *const sys::OrtGraph,
    graph_support_info: *mut sys::OrtEpGraphSupportInfo,
) -> Result<()> {
    let api = apis().ort;
    let ep_api = apis().ep;
    unsafe {
        let mut num_nodes = 0usize;
        check(
            (api.Graph_GetNumNodes.expect("Graph_GetNumNodes"))(graph, &mut num_nodes),
            "Graph_GetNumNodes",
        )?;
        if num_nodes == 0 {
            return Ok(());
        }
        let mut nodes: Vec<*const sys::OrtNode> = vec![ptr::null(); num_nodes];
        check(
            (api.Graph_GetNodes.expect("Graph_GetNodes"))(graph, nodes.as_mut_ptr(), num_nodes),
            "Graph_GetNodes",
        )?;

        // Compiling EP path: fuses supported ops into connected components
        // (one command buffer per component). Opt-in VULKAN_EP_COMPILE=1.
        if crate::compile::enabled() {
            return fuse_supported(graph, graph_support_info, &nodes);
        }

        let mut claimed = 0usize;
        for node in nodes {
            let mut op_type: *const c_char = ptr::null();
            check(
                (api.Node_GetOperatorType.expect("Node_GetOperatorType"))(node, &mut op_type),
                "Node_GetOperatorType",
            )?;
            let op = CStr::from_ptr(op_type).to_string_lossy();
            if !SUPPORTED_OPS.contains(&op.as_ref()) {
                continue;
            }
            // debug filter: VULKAN_EP_OPS=MatMulInteger,DynamicQuantizeLinear
            // to claim only a subset (A/B measurements).
            if let Some(only) = op_filter()
                && !only.iter().any(|o| o == op.as_ref())
            {
                continue;
            }
            // the registry applies the type constraints: claim only if a kernel exists
            let mut kernel_def: *const sys::OrtKernelDef = ptr::null();
            check(
                (ep_api
                    .EpGraphSupportInfo_LookUpKernel
                    .expect("LookUpKernel"))(
                    graph_support_info, node, &mut kernel_def
                ),
                "EpGraphSupportInfo_LookUpKernel",
            )?;
            if kernel_def.is_null() {
                continue;
            }
            check(
                (ep_api
                    .EpGraphSupportInfo_AddSingleNode
                    .expect("AddSingleNode"))(graph_support_info, node),
                "EpGraphSupportInfo_AddSingleNode",
            )?;
            claimed += 1;
        }
        log::info!("VulkanEP: claimed {claimed}/{num_nodes} nodes");
        Ok(())
    }
}

/// Op-type of a node.
unsafe fn node_op(node: *const sys::OrtNode) -> Result<String> {
    let api = apis().ort;
    let mut op: *const c_char = ptr::null();
    check(
        unsafe { (api.Node_GetOperatorType.expect("Node_GetOperatorType"))(node, &mut op) },
        "Node_GetOperatorType",
    )?;
    Ok(unsafe { CStr::from_ptr(op) }.to_string_lossy().into_owned())
}

/// Compiling EP: partitions supported nodes into **convex blocks** and
/// claims them for fusion. Each block becomes a single command buffer
/// (1 upload / 1 submit / 1 download), reducing CPU↔GPU boundaries.
unsafe fn fuse_supported(
    graph: *const sys::OrtGraph,
    gsi: *mut sys::OrtEpGraphSupportInfo,
    nodes: &[*const sys::OrtNode],
) -> Result<()> {
    let ep_api = apis().ep;
    // constant parameters passed as inputs (the reduction `axes`) decide
    // whether a node is supported: they must be resolved before the check
    let irs = unsafe { crate::compile::canonical_nodes(graph, nodes)? };

    // Per-node I/O and support (indices aligned with `nodes`).
    let mut node_inputs = Vec::with_capacity(nodes.len());
    let mut node_outputs = Vec::with_capacity(nodes.len());
    let mut supported = Vec::with_capacity(nodes.len());
    let only = op_filter();
    for (&node, ir) in nodes.iter().zip(&irs) {
        let (ins, outs) = unsafe { crate::compile::graph_ir::fused_node_io(node)? };
        let op = unsafe { node_op(node) }?;
        // `VULKAN_EP_OPS=Op1,Op2` restricts fusion to those ops: useful to
        // bisect which kernel introduces a divergence on a real model.
        let claimed = crate::compile::is_fusible_node(ir)
            && only.as_ref().is_none_or(|list| list.contains(&op));
        supported.push(claimed);
        node_inputs.push(ins);
        node_outputs.push(outs);
    }
    if log::log_enabled!(log::Level::Debug) {
        use std::collections::BTreeMap;
        let mut hist: BTreeMap<String, usize> = BTreeMap::new();
        for (i, &node) in nodes.iter().enumerate() {
            if !supported[i] {
                *hist.entry(unsafe { node_op(node) }?).or_default() += 1;
            }
        }
        let mut v: Vec<_> = hist.into_iter().collect();
        v.sort_by_key(|b| std::cmp::Reverse(b.1));
        log::debug!("VulkanEP compile: unsupported ops (freq): {v:?}");
    }

    let n_supported = supported.iter().filter(|&&s| s).count();
    if n_supported == 0 {
        log::info!("VulkanEP compile: no fusable nodes");
        return Ok(());
    }

    let groups =
        crate::compile::fusion::convex_groups(nodes.len(), &node_inputs, &node_outputs, &supported);

    let add = ep_api
        .EpGraphSupportInfo_AddNodesToFuse
        .expect("EpGraphSupportInfo_AddNodesToFuse");
    let mut multi = 0usize;
    for group in &groups {
        let ptrs: Vec<*const sys::OrtNode> = group.iter().map(|&i| nodes[i]).collect();
        check(
            unsafe { add(gsi, ptrs.as_ptr(), ptrs.len(), ptr::null()) },
            "EpGraphSupportInfo_AddNodesToFuse",
        )?;
        if ptrs.len() > 1 {
            multi += 1;
        }
    }
    log::info!(
        "VulkanEP compile: {n_supported} nodes in {} convex blocks ({multi} multi-node)",
        groups.len()
    );
    Ok(())
}

unsafe extern "C" fn get_kernel_registry(
    this_ptr: *mut sys::OrtEp,
    kernel_registry: *mut *const sys::OrtKernelRegistry,
) -> sys::OrtStatusPtr {
    let ep = unsafe { &*this_ptr.cast::<VulkanEp>() };
    let factory = unsafe { &mut *ep.factory };
    match factory.kernel_registry() {
        Ok(reg) => {
            unsafe { *kernel_registry = reg };
            ptr::null_mut()
        }
        Err(e) => to_status(Err(e)),
    }
}
