//! Recording the stream as data, so a step can be issued again without the
//! caller deciding what to issue.
//!
//! The stream API is imperative: whoever walks a graph calls `stream_dispatch`
//! once per kernel, and the cost of *deciding* those calls is paid again at
//! every step of a decode loop even though the calls are identical. Capture
//! turns one step's calls into a `Vec<StreamOp>` that can be replayed directly.
//!
//! This is a **software** replay: every command is recorded into a fresh
//! command buffer each time, exactly as the interpreter would have recorded it.
//! What it removes is the deciding, not the recording — a pre-recorded command
//! buffer would remove that too, at the price of baking push constants (which
//! change every token) into it.
//!
//! Two things a replay stands on, and neither is checked here:
//!
//! - the `VkBuffer` handles must still be alive and still hold the tensors the
//!   captured step put in them, so whoever owns the buffers has to keep them
//!   out of any pool that could hand them to something else;
//! - the pipeline handles must outlive the capture, which they do while the
//!   session's `KernelCache` lives.
//!
//! Both are the caller's contract. `onnx-vulkan-core` is where they are made
//! good; here the ops are plain data.

use crate::pipeline::ComputePipeline;
use ash::vk;

/// A dispatch, with everything resolved to Vulkan handles.
///
/// The push constants are owned and public: they are the part a decode step
/// changes (the cache length and the token's position), and patching them is
/// what makes the same plan valid at the next token.
#[derive(Clone)]
pub struct DispatchOp {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) bindings: Vec<vk::DescriptorBufferInfo>,
    pub push: Vec<u8>,
    pub groups: [u32; 3],
}

/// A host→device copy. The payload is owned because a replay refreshes it: the
/// bytes of a token's mask are not the bytes of the next one's.
#[derive(Clone)]
pub struct UploadOp {
    pub(crate) dst: vk::Buffer,
    pub(crate) dst_offset: u64,
    pub bytes: Vec<u8>,
    /// What was uploaded, when the caller knew a name for it.
    ///
    /// A replay has to refresh the payloads that change from token to token —
    /// the mask, the positions — and refreshing means asking whoever owns the
    /// values for the current bytes of *this* one. Guessing the value from the
    /// node that issued the upload does not work: a node uploads several of its
    /// inputs and they are indistinguishable once they are bytes. So the label
    /// is attached where the name is still in hand.
    pub label: Option<String>,
}

/// A device→device copy. Nothing in it varies with the step.
#[derive(Clone)]
pub struct CopyOp {
    pub(crate) src: vk::Buffer,
    pub(crate) src_offset: u64,
    pub(crate) dst: vk::Buffer,
    pub(crate) dst_offset: u64,
    pub(crate) bytes: u64,
}

#[derive(Clone)]
pub enum StreamOp {
    Dispatch(DispatchOp),
    Upload(UploadOp),
    Copy(CopyOp),
}

impl StreamOp {
    /// Two ops issue the same work on the same memory, ignoring push constants
    /// and upload payloads — the two things a step is expected to change.
    ///
    /// This is how a plan is validated instead of assumed: capture two steps,
    /// and if the structure differs at all, replaying the first would not be
    /// the second with different numbers in it.
    pub fn same_shape(&self, other: &Self) -> bool {
        match (self, other) {
            (StreamOp::Dispatch(a), StreamOp::Dispatch(b)) => {
                a.pipeline == b.pipeline
                    && a.groups == b.groups
                    && a.push.len() == b.push.len()
                    && a.bindings.len() == b.bindings.len()
                    && a.bindings.iter().zip(&b.bindings).all(|(x, y)| {
                        x.buffer == y.buffer && x.offset == y.offset && x.range == y.range
                    })
            }
            (StreamOp::Upload(a), StreamOp::Upload(b)) => {
                a.dst == b.dst && a.dst_offset == b.dst_offset
            }
            (StreamOp::Copy(a), StreamOp::Copy(b)) => {
                a.src == b.src
                    && a.dst == b.dst
                    && a.src_offset == b.src_offset
                    && a.dst_offset == b.dst_offset
                    && a.bytes == b.bytes
            }
            _ => false,
        }
    }

    /// A one-line description, for the message a rejected plan prints.
    pub fn kind(&self) -> String {
        match self {
            StreamOp::Dispatch(op) => format!(
                "dispatch of {:?} on {:?}",
                op.groups,
                op.bindings.iter().map(|b| b.buffer).collect::<Vec<_>>()
            ),
            StreamOp::Upload(op) => format!(
                "upload of {} bytes into {:?} ({})",
                op.bytes.len(),
                op.dst,
                op.label.as_deref().unwrap_or("unnamed")
            ),
            StreamOp::Copy(op) => format!("copy of {} bytes {:?}->{:?}", op.bytes, op.src, op.dst),
        }
    }
}

impl DispatchOp {
    pub(crate) fn new(
        pipeline: &ComputePipeline,
        bindings: Vec<vk::DescriptorBufferInfo>,
        push: &[u8],
        groups: [u32; 3],
    ) -> Self {
        Self {
            pipeline: pipeline.pipeline,
            layout: pipeline.layout,
            set_layout: pipeline.set_layout,
            bindings,
            push: push.to_vec(),
            groups,
        }
    }
}

impl UploadOp {
    pub(crate) fn new(dst: vk::Buffer, dst_offset: u64, bytes: &[u8], label: Option<&str>) -> Self {
        Self {
            dst,
            dst_offset,
            bytes: bytes.to_vec(),
            label: label.map(str::to_owned),
        }
    }
}
