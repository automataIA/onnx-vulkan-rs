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
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
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
