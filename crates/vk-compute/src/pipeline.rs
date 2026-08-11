//! Pipeline compute + dispatch sincrono.

use crate::buffer::GpuBuffer;
use crate::context::VkContext;
use anyhow::Result;
use ash::vk;

/// Range of buffer bound to binding: descriptor starts at `offset`,
/// so shader indexes starting from tensor origin. Needed for ORT's memory
/// pattern, which assigns tensor slices within a single allocation.
#[derive(Clone, Copy)]
pub struct BufferSlice<'a> {
    pub buf: &'a GpuBuffer,
    pub offset: u64,
}

impl<'a> From<&'a GpuBuffer> for BufferSlice<'a> {
    fn from(buf: &'a GpuBuffer) -> Self {
        Self { buf, offset: 0 }
    }
}

pub struct ComputePipeline {
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    shader_module: vk::ShaderModule,
    num_buffers: u32,
    push_const_size: u32,
}

impl VkContext {
    /// Compute pipeline from SPIR-V: `num_buffers` storage buffers (binding 0..n)
    /// + push constants of `push_const_size` bytes (0 = none).
    pub fn create_pipeline(
        &self,
        spirv: &[u32],
        num_buffers: u32,
        push_const_size: u32,
    ) -> Result<ComputePipeline> {
        let device = &self.device;
        unsafe {
            let shader_module = device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..num_buffers)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;

            let set_layouts = [set_layout];
            let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
            let push_range = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(push_const_size)];
            if push_const_size > 0 {
                layout_info = layout_info.push_constant_ranges(&push_range);
            }
            let layout = device.create_pipeline_layout(&layout_info, None)?;

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(c"main");
            let info = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipeline = device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| e)?[0];

            Ok(ComputePipeline {
                pipeline,
                layout,
                set_layout,
                shader_module,
                num_buffers,
                push_const_size,
            })
        }
    }

    /// Records a dispatch in the given command buffer. Descriptor set is
    /// allocated from the persistent arena (reset on flush).
    pub(crate) fn record_dispatch(
        &self,
        cmd: vk::CommandBuffer,
        pipeline: &ComputePipeline,
        buffers: &[BufferSlice<'_>],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        assert_eq!(buffers.len() as u32, pipeline.num_buffers);
        assert_eq!(push_constants.len() as u32, pipeline.push_const_size);
        for b in buffers {
            anyhow::ensure!(
                b.offset % self.storage_offset_alignment == 0,
                "offset {} not aligned to {} (minStorageBufferOffsetAlignment)",
                b.offset,
                self.storage_offset_alignment
            );
            anyhow::ensure!(
                b.offset < b.buf.size,
                "offset {} outside the buffer of {} bytes",
                b.offset,
                b.buf.size
            );
        }
        let set = self.acquire_descriptor_set(pipeline.set_layout)?;
        let device = &self.device;
        unsafe {
            let buffer_infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
                .iter()
                .map(|b| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(b.buf.buffer)
                        .offset(b.offset)
                        .range(vk::WHOLE_SIZE)]
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                0,
                &[set],
                &[],
            );
            if !push_constants.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            device.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            Ok(())
        }
    }

    /// Records a dispatch whose bindings are already resolved to handles.
    ///
    /// The validation `record_dispatch` performs — binding count, push size,
    /// offset alignment — was done when the op was captured, on the buffers
    /// themselves. Here there is nothing left to check against: the op holds
    /// handles, and a handle carries neither a size nor a layout.
    pub(crate) fn record_dispatch_op(
        &self,
        cmd: vk::CommandBuffer,
        op: &crate::DispatchOp,
    ) -> Result<()> {
        let set = self.acquire_descriptor_set(op.set_layout)?;
        let device = &self.device;
        unsafe {
            let infos: Vec<[vk::DescriptorBufferInfo; 1]> =
                op.bindings.iter().map(|info| [*info]).collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            device.update_descriptor_sets(&writes, &[]);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, op.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                op.layout,
                0,
                &[set],
                &[],
            );
            if !op.push.is_empty() {
                device.cmd_push_constants(
                    cmd,
                    op.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &op.push,
                );
            }
            device.cmd_dispatch(cmd, op.groups[0], op.groups[1], op.groups[2]);
        }
        Ok(())
    }

    /// Synchronous dispatch (test/standalone use): enqueues into the stream and flushes.
    pub fn dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        self.stream_dispatch(pipeline, buffers, push_constants, groups)?;
        self.flush()
    }

    /// Measures repeated dispatches with Vulkan timestamp queries.
    ///
    /// The returned vector contains one amortized GPU-duration sample per
    /// dispatch in nanoseconds. Pending stream work is flushed first; all
    /// measured batches then share one command buffer and one queue submission.
    /// Each interval contains `dispatches_per_sample` dispatches and is divided
    /// by that count, amortizing timestamp-command observer overhead while host
    /// recording, process startup, and fence latency remain excluded.
    pub fn measure_dispatch_gpu(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
        sample_count: u32,
        dispatches_per_sample: u32,
    ) -> Result<Vec<u64>> {
        anyhow::ensure!(
            sample_count > 0,
            "GPU timestamp sample count must be positive"
        );
        anyhow::ensure!(
            dispatches_per_sample > 0,
            "dispatches per GPU timestamp sample must be positive"
        );
        anyhow::ensure!(
            self.timestamp_valid_bits > 0 && self.timestamp_period > 0.0,
            "the selected Vulkan compute queue does not support timestamps"
        );
        anyhow::ensure!(
            self.timestamp_valid_bits <= 64,
            "invalid Vulkan timestampValidBits {}",
            self.timestamp_valid_bits
        );
        let query_count = sample_count
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("GPU timestamp sample count overflow"))?;
        anyhow::ensure!(
            query_count <= crate::context::TS_CAPACITY,
            "GPU timestamp sample count {sample_count} exceeds the limit {}",
            crate::context::TS_CAPACITY / 2
        );

        self.flush()?;
        let buffer_slices: Vec<BufferSlice<'_>> =
            buffers.iter().map(|buffer| (*buffer).into()).collect();
        // SAFETY: the device is live and the validated query count is non-zero.
        let query_pool = unsafe {
            self.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(query_count),
                None,
            )?
        };
        let result = (|| {
            self.run_commands(|cmd| {
                // SAFETY: `run_commands` supplies an actively recording command
                // buffer, and this query pool stays alive until its fence passes.
                unsafe {
                    self.device
                        .cmd_reset_query_pool(cmd, query_pool, 0, query_count);
                }
                for sample in 0..sample_count {
                    let barrier = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                        );
                    // SAFETY: the command buffer is recording and the barrier
                    // references no external memory or transient handles.
                    unsafe {
                        self.device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[barrier],
                            &[],
                            &[],
                        );
                        // Two independent endpoints keep the inter-dispatch
                        // barrier outside every measured interval.
                        self.device.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::TOP_OF_PIPE,
                            query_pool,
                            sample * 2,
                        );
                    }
                    for repetition in 0..dispatches_per_sample {
                        if repetition > 0 {
                            let barrier = vk::MemoryBarrier::default()
                                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                                .dst_access_mask(
                                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                                );
                            // SAFETY: as above, the command buffer is recording
                            // and the barrier references no external handles.
                            unsafe {
                                self.device.cmd_pipeline_barrier(
                                    cmd,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::PipelineStageFlags::COMPUTE_SHADER,
                                    vk::DependencyFlags::empty(),
                                    &[barrier],
                                    &[],
                                    &[],
                                );
                            }
                        }
                        self.record_dispatch(
                            cmd,
                            pipeline,
                            &buffer_slices,
                            push_constants,
                            groups,
                        )?;
                    }
                    // SAFETY: the query index is below `query_count`, and the
                    // pool remains alive through submission and readback.
                    unsafe {
                        self.device.cmd_write_timestamp(
                            cmd,
                            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                            query_pool,
                            sample * 2 + 1,
                        );
                    }
                }
                Ok(())
            })?;

            let mut timestamps = vec![0u64; query_count as usize];
            // SAFETY: `run_commands` waited for its submission fence, so every
            // requested query is complete and the destination slice is sized
            // to exactly `query_count` 64-bit results.
            unsafe {
                self.device.get_query_pool_results(
                    query_pool,
                    0,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64,
                )?;
            }
            timestamps
                .chunks_exact(2)
                .enumerate()
                .map(|(index, pair)| {
                    let ticks = timestamp_delta_ticks(pair[0], pair[1], self.timestamp_valid_bits);
                    let nanoseconds = (ticks as f64 * self.timestamp_period as f64
                        / dispatches_per_sample as f64)
                        .round() as u64;
                    anyhow::ensure!(
                        nanoseconds > 0,
                        "GPU timestamp sample {index} has zero duration"
                    );
                    Ok(nanoseconds)
                })
                .collect()
        })();
        self.reset_descriptors();
        // SAFETY: the measurement submission has completed (or was never
        // submitted), and no command buffer can still reference this pool.
        unsafe { self.device.destroy_query_pool(query_pool, None) };
        result
    }

    pub fn destroy_pipeline(&self, pipeline: ComputePipeline) {
        unsafe {
            self.device.destroy_pipeline(pipeline.pipeline, None);
            self.device.destroy_pipeline_layout(pipeline.layout, None);
            self.device
                .destroy_descriptor_set_layout(pipeline.set_layout, None);
            self.device
                .destroy_shader_module(pipeline.shader_module, None);
        }
    }
}

pub(crate) fn timestamp_delta_ticks(start: u64, end: u64, valid_bits: u32) -> u64 {
    let delta = end.wrapping_sub(start);
    if valid_bits == 64 {
        delta
    } else {
        delta & ((1u64 << valid_bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::timestamp_delta_ticks;

    #[test]
    fn timestamp_delta_handles_full_width_and_wrapping_counters() {
        assert_eq!(timestamp_delta_ticks(10, 25, 64), 15);
        assert_eq!(timestamp_delta_ticks(u64::MAX - 4, 3, 64), 8);
        assert_eq!(timestamp_delta_ticks(250, 5, 8), 11);
    }
}
