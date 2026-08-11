//! A decode step recorded once and issued again, without walking the graph.
//!
//! Measured on a token of qwen2.5-VL, the interpreter spends ~2.55 ms deciding
//! ~1000 dispatches and ~0.45 ms in nodes that dispatch nothing — `Reshape`,
//! `Shape`, `Range`, `Gather` — which build the mask and the positions. Of the
//! first figure, 1.40 ms is the Vulkan recording itself. What is left, ~1.15 ms,
//! is the framework: name lookups, shape arithmetic, a `Vec` per push constant,
//! a match per op. That is the part a plan removes, and only that part: the
//! recording still happens (see `vk_compute::capture`), and the host nodes still
//! run, because their results change with every token.
//!
//! The plan is built by **observing**, not by declaring. Three consecutive
//! steps are captured; the first two say which bytes of which push constant
//! move with the step and by how much, and the third is used to check the
//! prediction against what the interpreter actually produced. Anything that
//! does not fit — a grid that changes, a buffer that is not the same one, a
//! payload with no value behind it — is refused with a reason instead of
//! replayed. A wrong plan is silent corruption, so the failure has to be loud
//! and at build time.
//!
//! What has to hold for a plan to exist at all, and what earned it:
//!
//! - **buffer identity**: the same request must be served the same `VkBuffer`
//!   at every step, or a replayed binding points at the wrong memory. That is
//!   the retained pool (FIFO, see `vk_compute::buffer::StoragePool`).
//! - **step-invariant grids and sizes**: the attention scratch is laid out for
//!   the cache's physical extent, not for the token's length.
//! - **nothing allocated per step**: an allocation would hand out a buffer the
//!   plan is still pointing at.

use crate::{Error, GraphIr, Result, Tensor};
use std::ops::Range;
use vk_compute::StreamOp;

/// One captured step: the commands it issued, and which node issued each.
pub struct StepTrace {
    pub(crate) ops: Vec<StreamOp>,
    /// Ops produced by node `i`, as a range into `ops`.
    pub(crate) nodes: Vec<Range<usize>>,
}

impl StepTrace {
    pub(crate) fn new(ops: Vec<StreamOp>, nodes: Vec<Range<usize>>) -> Self {
        Self { ops, nodes }
    }

    pub fn ops(&self) -> &[StreamOp] {
        &self.ops
    }
}

/// A push-constant word that moves with the step: `base + delta · steps`.
struct PushPatch {
    op: usize,
    /// Byte offset of the word inside the push-constant block. Every kernel
    /// here lays its push constants out as 4-byte words, which is what makes a
    /// word the unit a patch can be expressed in.
    offset: usize,
    base: u32,
    delta: i64,
}

/// An upload whose payload is a value the host nodes rebuild every step.
struct PayloadPatch {
    op: usize,
    value: String,
}

/// A decode step, ready to be issued for a later token.
pub struct StepPlan {
    ops: Vec<StreamOp>,
    /// Nodes that must run again at every step: they dispatch nothing and
    /// compute the mask, the positions and the shapes the uploads carry.
    host_nodes: Vec<usize>,
    push: Vec<PushPatch>,
    payloads: Vec<PayloadPatch>,
    /// The step the captured ops belong to, so a later one can be expressed as
    /// a distance from it.
    origin: i64,
    /// Buffers taken out of the pool so nothing else can be served them, see
    /// `hold_pool`.
    held: Vec<vk_compute::GpuBuffer>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepPlanStats {
    pub dispatches: usize,
    pub host_nodes: usize,
    pub temporary_buffer_sizes: Vec<u64>,
    pub temporary_bytes: u64,
}

impl StepPlan {
    /// Builds a plan from consecutive captured steps, `steps[i]` being the trace
    /// of step `origin + i`.
    ///
    /// Three traces minimum: two to see what moves, one to check that what was
    /// inferred from them predicts a step nobody looked at. A plan that fails
    /// the check is not returned.
    pub fn build(ir: &GraphIr, traces: &[StepTrace], origin: i64) -> Result<Self> {
        let borrowed: Vec<&StepTrace> = traces.iter().collect();
        Self::build_from(ir, &borrowed, origin)
    }

    /// As [`Self::build`], for a caller that keeps its traces somewhere that
    /// does not hand out a slice — a decode loop holds only the last three.
    pub fn build_from(ir: &GraphIr, traces: &[&StepTrace], origin: i64) -> Result<Self> {
        if traces.len() < 3 {
            return Err(Error::Unsupported(format!(
                "a plan needs three captured steps to be checked, {} given",
                traces.len()
            )));
        }
        let (first, second) = (&traces[0], &traces[1]);
        same_structure(ir, first, second)?;
        let host_nodes = host_nodes(first);
        let payloads = payloads(traces)?;
        let push = push_patches(first, second)?;
        let plan = Self {
            ops: first.ops.clone(),
            host_nodes,
            push,
            payloads,
            origin,
            held: Vec::new(),
        };
        plan.check(ir, traces)?;
        Ok(plan)
    }

    /// Every dispatch the plan issues, for the caller that wants to know what a
    /// step costs without running it.
    pub fn dispatches(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, StreamOp::Dispatch(_)))
            .count()
    }

    pub fn host_node_count(&self) -> usize {
        self.host_nodes.len()
    }

    /// Metadata safe to persist in an execution-plan manifest. It describes
    /// the verified live plan without exposing commands, handles, or buffers.
    pub fn stats(&self) -> StepPlanStats {
        let mut temporary_buffer_sizes = self
            .held
            .iter()
            .map(|buffer| buffer.size)
            .collect::<Vec<_>>();
        temporary_buffer_sizes.sort_unstable();
        StepPlanStats {
            dispatches: self.dispatches(),
            host_nodes: self.host_node_count(),
            temporary_bytes: temporary_buffer_sizes.iter().copied().sum(),
            temporary_buffer_sizes,
        }
    }

    /// Takes the free buffers out of the pool and keeps them.
    ///
    /// The plan's bindings name buffers the captured step returned to the pool
    /// when it was done with them. Nothing must be handed those buffers again
    /// while the plan is alive — an `ArgMax` allocating a scratch of the same
    /// size would be given one, and the next replayed step would write its
    /// attention scores into it. Holding them is what makes the pool an arena
    /// for as long as the plan lives.
    pub fn hold_pool(&mut self, context: &vk_compute::VkContext) {
        self.held.extend(context.take_storage_pool());
    }

    /// Returns the held buffers to the pool. Consuming, and not `Drop`, for the
    /// same reason `Outputs::finish` is: it needs the context.
    pub fn release(self, context: &vk_compute::VkContext) {
        for buffer in self.held {
            context.recycle_storage_buffer(buffer);
        }
    }

    /// Predicts the ops of `step` and compares them against what the traces
    /// recorded, so the caller finds out here and not through a wrong logit.
    fn check(&self, ir: &GraphIr, traces: &[&StepTrace]) -> Result<()> {
        for (index, trace) in traces.iter().enumerate().skip(1) {
            same_structure(ir, traces[0], trace)?;
            let ops = self.patched(self.origin + index as i64);
            for (position, (predicted, actual)) in ops.iter().zip(&trace.ops).enumerate() {
                let (StreamOp::Dispatch(predicted), StreamOp::Dispatch(actual)) =
                    (predicted, actual)
                else {
                    continue;
                };
                if predicted.push != actual.push {
                    return Err(Error::Unsupported(format!(
                        "the plan predicts push constants {:?} for op {position} of step {}, \
                         the interpreter produced {:?}: the step does not move linearly and \
                         replaying it would compute the wrong thing",
                        predicted.push,
                        self.origin + index as i64,
                        actual.push
                    )));
                }
            }
        }
        Ok(())
    }

    /// The plan's ops with this step's scalars in them.
    fn patched(&self, step: i64) -> Vec<StreamOp> {
        let mut ops = self.ops.clone();
        let distance = step - self.origin;
        for patch in &self.push {
            let StreamOp::Dispatch(dispatch) = &mut ops[patch.op] else {
                continue;
            };
            let value = (patch.base as i64 + patch.delta * distance) as u32;
            dispatch.push[patch.offset..patch.offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        ops
    }
}

/// Two traces issue the same commands on the same memory, differing only in
/// the scalars and payloads a step is allowed to change.
fn same_structure(ir: &GraphIr, first: &StepTrace, second: &StepTrace) -> Result<()> {
    if first.ops.len() != second.ops.len() {
        return Err(Error::Unsupported(format!(
            "one step issued {} commands and the next {}: the graph is not doing the same work \
             from token to token",
            first.ops.len(),
            second.ops.len()
        )));
    }
    for (index, (a, b)) in first.ops.iter().zip(&second.ops).enumerate() {
        if !a.same_shape(b) {
            let node = first
                .nodes
                .iter()
                .position(|range| range.contains(&index))
                .map(|node| format!("{} '{}'", ir.nodes[node].op, ir.nodes[node].name))
                .unwrap_or_else(|| "no node".into());
            return Err(Error::Unsupported(format!(
                "command {index} ({node}) is a {} in one step and a {} in the next: a plan can \
                 carry different numbers, not a different pipeline, buffer or grid",
                a.kind(),
                b.kind()
            )));
        }
    }
    Ok(())
}

/// Nodes that issued no command at all: pure host computation, which the plan
/// re-runs because its results are what the uploads carry.
///
/// A host node reading a value some *device* node produced is allowed, and it
/// happens: qwen2.5-VL's mRoPE path asks for the shape of a projection. Only
/// the metadata survives the step — the buffer went back to the pool at the
/// last reader — so a node that wants the shape gets it and a node that wants
/// the bytes fails, loudly, rather than reading the captured token's data.
///
/// Which of the two it is cannot be read off the graph, and is not guessed
/// here: `verify_against` re-runs these nodes under exactly the conditions a
/// replay gives them, and that is where the difference shows up.
fn host_nodes(trace: &StepTrace) -> Vec<usize> {
    trace
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, range)| range.is_empty())
        .map(|(index, _)| index)
        .collect()
}

/// For every upload whose bytes are not the same in all traces, the value whose
/// bytes they are.
///
/// The value comes from the label the environment attached when it uploaded it,
/// not from the node that issued the upload: a node uploads several of its
/// inputs and they are the same thing once they are bytes. An upload that
/// changes and carries no label is refused — replaying it would upload the
/// captured token's data at every later token.
///
/// An upload whose bytes are identical in all three traces is left alone. Those
/// are the constants a kernel passes in a buffer because they do not fit a push
/// constant, and they are already shared by content across the session.
fn payloads(traces: &[&StepTrace]) -> Result<Vec<PayloadPatch>> {
    let mut patches = Vec::new();
    for (position, op) in traces[0].ops.iter().enumerate() {
        let StreamOp::Upload(first) = op else {
            continue;
        };
        let varies = traces[1..].iter().any(|trace| match &trace.ops[position] {
            StreamOp::Upload(later) => later.bytes != first.bytes,
            _ => true,
        });
        if !varies {
            continue;
        }
        let value = first.label.clone().ok_or_else(|| {
            Error::Unsupported(format!(
                "upload {position} carries {} bytes that change at every step and no value name: \
                 replaying it would upload the captured token's data",
                first.bytes.len()
            ))
        })?;
        patches.push(PayloadPatch {
            op: position,
            value,
        });
    }
    Ok(patches)
}

/// Push-constant words that differ between two consecutive steps, as an affine
/// law in the step. The prediction is checked in `StepPlan::check`.
fn push_patches(first: &StepTrace, second: &StepTrace) -> Result<Vec<PushPatch>> {
    let mut patches = Vec::new();
    for (op, (a, b)) in first.ops.iter().zip(&second.ops).enumerate() {
        let (StreamOp::Dispatch(a), StreamOp::Dispatch(b)) = (a, b) else {
            continue;
        };
        if a.push.len() % 4 != 0 {
            return Err(Error::Unsupported(format!(
                "op {op} pushes {} bytes, which is not a whole number of words: a plan patches \
                 words",
                a.push.len()
            )));
        }
        for offset in (0..a.push.len()).step_by(4) {
            let word = |bytes: &[u8]| {
                u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("four bytes"))
            };
            let (before, after) = (word(&a.push), word(&b.push));
            if before == after {
                continue;
            }
            patches.push(PushPatch {
                op,
                offset,
                base: before,
                delta: after as i64 - before as i64,
            });
        }
    }
    Ok(patches)
}

impl StepPlan {
    /// Checks the plan's own output against a step the interpreter ran.
    ///
    /// `check` compares the push constants, which are inferred; this compares
    /// **everything the plan would issue**, payloads included, against what the
    /// interpreter actually issued for the same step. It is the difference
    /// between trusting that the host nodes rebuild the mask the same way and
    /// having seen them do it.
    pub fn verify_against(
        &self,
        trace: &StepTrace,
        step: i64,
        host: &dyn Fn(&str) -> Result<Vec<u8>>,
    ) -> Result<()> {
        let ops = self.ops_for(step, host)?;
        for (index, (planned, actual)) in ops.iter().zip(&trace.ops).enumerate() {
            let disagreement = match (planned, actual) {
                (StreamOp::Dispatch(a), StreamOp::Dispatch(b)) if a.push != b.push => {
                    Some(format!("push constants {:?} against {:?}", a.push, b.push))
                }
                (StreamOp::Upload(a), StreamOp::Upload(b)) if a.bytes != b.bytes => Some(format!(
                    "{} bytes of '{}' that are not the ones the step uploaded",
                    a.bytes.len(),
                    a.label.as_deref().unwrap_or("an unnamed value")
                )),
                _ => None,
            };
            if let Some(disagreement) = disagreement {
                return Err(Error::Unsupported(format!(
                    "the plan would issue, for command {index} of step {step}, {disagreement}"
                )));
            }
        }
        Ok(())
    }

    /// The ops to issue for `step`, given an environment in which the host
    /// nodes have already run.
    pub fn ops_for(
        &self,
        step: i64,
        host: &dyn Fn(&str) -> Result<Vec<u8>>,
    ) -> Result<Vec<StreamOp>> {
        let mut ops = self.patched(step);
        for patch in &self.payloads {
            let StreamOp::Upload(upload) = &mut ops[patch.op] else {
                continue;
            };
            upload.bytes = host(&patch.value)?;
        }
        Ok(ops)
    }

    pub fn host_nodes(&self) -> &[usize] {
        &self.host_nodes
    }
}

/// A tensor's bytes as they are on the host, for an upload the plan refreshes.
pub fn host_bytes(tensor: &Tensor<'_>) -> Option<Vec<u8>> {
    match tensor {
        Tensor::Host(host) => Some(host.data.clone()),
        Tensor::Device(_) => None,
    }
}
