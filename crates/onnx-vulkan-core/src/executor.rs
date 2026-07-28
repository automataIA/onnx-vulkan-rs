//! Public execution API: one graph, one set of session resources, one run.
//!
//! Before this type existed, running a graph meant assembling three pieces by
//! hand — a `KernelCache`, an `ExecutionEnv` built on it and its initializers,
//! and a call to `execute` — and every host had to do it the same way to get
//! the same behaviour. The ORT plugin did exactly that inside its own compute
//! info. `Executor` is that assembly promoted to a type, so there is one path
//! and not one per host.
//!
//! Two lifetimes matter and they are independent: the executor borrows the
//! `VkContext` for as long as it lives, and each run borrows the executor for
//! as long as its outputs are read.

use crate::{
    ExecutionEnv, GraphIr, HostTensor, KernelCache, Result, Tensor, execute, is_implemented_node,
};
use vk_compute::VkContext;

/// A graph plus the GPU resources reused across its runs.
///
/// Pipelines and packed weights live here, so a second run of the same graph
/// compiles no shader and re-packs no weight. Dropping the executor frees them.
pub struct Executor<'context> {
    ir: GraphIr,
    cache: KernelCache<'context>,
}

impl<'context> Executor<'context> {
    /// Builds the executor, **rejecting** a graph with any node the interpreter
    /// cannot run.
    ///
    /// The check happens here rather than at the first dispatch on purpose: a
    /// model either runs entirely on the GPU or it fails loud, and the useful
    /// moment to fail is at load, not halfway through an inference.
    pub fn new(context: &'context VkContext, mut ir: GraphIr) -> Result<Self> {
        // Load-time rewrites live here and not in each host, so the standalone
        // path and the ORT plugin cannot end up running different graphs.
        let fused = crate::rewrite::fuse_layernorm(&mut ir);
        let folded = crate::rewrite::fold_constants(&mut ir);
        if fused > 0 || folded > 0 {
            let pruned = crate::rewrite::prune_dead_nodes(&mut ir);
            let released = crate::rewrite::prune_dead_initializers(&mut ir);
            log::info!(
                "rewrite: {fused} decomposed LayerNormalization fused, \
                 {folded} constant nodes folded, {pruned} orphaned nodes pruned, \
                 {:.1} MB of initializers released, {} nodes left",
                released as f64 / 1e6,
                ir.nodes.len()
            );
        }
        // Every unsupported node, not the first: a caller deciding whether this
        // engine can run their model needs the whole list, and discovering it
        // one recompile at a time is not a report.
        let unsupported: Vec<&crate::NodeIr> = ir
            .nodes
            .iter()
            .filter(|n| !is_implemented_node(n))
            .collect();
        if !unsupported.is_empty() {
            let mut by_op: std::collections::BTreeMap<&str, (usize, &str)> = Default::default();
            for node in &unsupported {
                let entry = by_op.entry(&node.op).or_insert((0, node.name.as_str()));
                entry.0 += 1;
            }
            let detail = by_op
                .iter()
                .map(|(op, (count, first))| format!("{op} ×{count} (e.g. '{first}')"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(crate::Error::Unsupported(format!(
                "{} nodes of {} types are not implemented: {detail}",
                unsupported.len(),
                by_op.len()
            )));
        }
        Ok(Self {
            cache: KernelCache::new(context),
            ir,
        })
    }

    pub fn graph(&self) -> &GraphIr {
        &self.ir
    }

    pub fn context(&self) -> &'context VkContext {
        self.cache.context()
    }

    /// Session resources, exposed for the hosts that need to inspect them
    /// (counters of compiled pipelines and packed weights).
    pub fn cache(&self) -> &KernelCache<'context> {
        &self.cache
    }

    /// Runs the graph with the given named inputs.
    ///
    /// The dispatches are **enqueued**, not submitted: whoever reads the
    /// outputs decides when to flush. That is what lets a host copy a device
    /// output into a buffer of its own inside the same command buffer, without
    /// a round trip through host memory.
    pub fn run<'a>(&'a self, inputs: Vec<(&str, Tensor<'a>)>) -> Result<Outputs<'a>> {
        let mut env = ExecutionEnv::new(&self.cache, &self.ir.initializers);
        for (name, tensor) in inputs {
            env.set(name, tensor);
        }
        execute(&self.ir, &mut env)?;
        Ok(Outputs { env })
    }
}

/// Values produced by a run, readable until dropped.
///
/// Holds the whole environment and not just the graph outputs: a host adapter
/// needs shape and dtype of a value to allocate its own destination.
///
/// Only the graph's declared outputs are readable. Every other value is
/// released at its last reader, so its VRAM can serve the rest of the block —
/// a graph whose outputs are not declared holds nothing at the end.
pub struct Outputs<'a> {
    env: ExecutionEnv<'a, 'a>,
}

impl<'a> Outputs<'a> {
    pub fn value(&self, name: &str) -> Option<&Tensor<'a>> {
        self.env.value(name)
    }

    pub fn shape_of(&self, name: &str) -> Result<Vec<i64>> {
        self.env.shape_of(name)
    }

    pub fn dtype_of(&self, name: &str) -> Result<i32> {
        self.env.dtype_of(name)
    }

    /// Reads a value on host, downloading it if it lives in VRAM. **Forces a
    /// flush** — this is the synchronization point.
    pub fn host(&self, name: &str) -> Result<HostTensor> {
        self.env.host(name)
    }

    pub fn on_device(&self, name: &str) -> bool {
        self.env.on_device(name)
    }

    /// Releases the run's buffers. Consuming instead of `Drop` because freeing
    /// device memory can fail, and swallowing that in a destructor would hide
    /// a leak.
    pub fn finish(self) {
        self.env.finish();
    }
}
