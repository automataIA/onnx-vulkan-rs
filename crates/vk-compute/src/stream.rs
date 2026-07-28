//! Deferred command stream.
//!
//! Dispatches and copies are recorded in a single command buffer;
//! submit (with fence) occurs only on `flush()` — typically when a
//! result must return to CPU. Drastically reduces `vkQueueSubmit` calls
//! and per-op waits (NVIDIA/Khronos best practice).

use crate::buffer::GpuBuffer;
use crate::context::VkContext;
use crate::pipeline::{BufferSlice, ComputePipeline};
use anyhow::{Context as _, Result};
use ash::vk;

#[derive(Default)]
pub(crate) struct StreamState {
    /// Command buffer being recorded (None = empty stream).
    cmd: Option<vk::CommandBuffer>,
    /// Staging buffers kept alive until flush.
    staging: Vec<GpuBuffer>,
    /// Buffers to destroy after flush.
    graveyard: Vec<GpuBuffer>,
    /// Profiling: op for each timed dispatch (index = timestamp i→i+1).
    ts_labels: Vec<&'static str>,
    /// Timestamps written into the current command buffer (0 = no baseline).
    ts_count: u32,
}

impl VkContext {
    fn stream_cmd(&self, state: &mut StreamState) -> Result<vk::CommandBuffer> {
        if let Some(cmd) = state.cmd {
            return Ok(cmd);
        }
        let device = &self.device;
        let _guard = self.submit_lock.lock().unwrap();
        let cmd = unsafe {
            let cmd = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            cmd
        };
        state.cmd = Some(cmd);
        state.ts_labels.clear();
        state.ts_count = 0;
        // profiling: reset query pool + baseline timestamp at the start of the cmd buffer
        if self.profiling_on() {
            let pool = self.ensure_query_pool()?;
            unsafe {
                device.cmd_reset_query_pool(cmd, pool, 0, crate::context::TS_CAPACITY);
                device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, pool, 0);
            }
            state.ts_count = 1;
        }
        Ok(cmd)
    }

    /// Profiling enabled and queue supports timestamps.
    fn profiling_on(&self) -> bool {
        crate::stats::enabled() && self.timestamp_period > 0.0
    }

    fn ensure_query_pool(&self) -> Result<vk::QueryPool> {
        let mut guard = self.query_pool.lock().unwrap();
        if let Some(pool) = *guard {
            return Ok(pool);
        }
        let pool = unsafe {
            self.device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(crate::context::TS_CAPACITY),
                None,
            )?
        };
        *guard = Some(pool);
        Ok(pool)
    }

    /// Writes timestamp after dispatch and records label.
    fn write_timestamp(&self, cmd: vk::CommandBuffer, state: &mut StreamState) {
        if !self.profiling_on() || state.ts_count >= crate::context::TS_CAPACITY {
            return;
        }
        let pool = match *self.query_pool.lock().unwrap() {
            Some(p) => p,
            None => return,
        };
        unsafe {
            self.device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                pool,
                state.ts_count,
            );
        }
        state.ts_labels.push(crate::stats::current_op());
        state.ts_count += 1;
    }

    /// Reads timestamps after flush and attributes GPU time per op.
    fn collect_timestamps(&self, state: &mut StreamState) {
        if state.ts_count < 2 {
            state.ts_labels.clear();
            state.ts_count = 0;
            return;
        }
        let pool = match *self.query_pool.lock().unwrap() {
            Some(p) => p,
            None => return,
        };
        let mut ts = vec![0u64; state.ts_count as usize];
        let ok = unsafe {
            self.device
                .get_query_pool_results(pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64)
                .is_ok()
        };
        if ok {
            for (i, label) in state.ts_labels.iter().enumerate() {
                let ticks = ts[i + 1].saturating_sub(ts[i]);
                let ns = (ticks as f64 * self.timestamp_period as f64) as u64;
                crate::stats::record_gpu(label, ns);
            }
        }
        state.ts_labels.clear();
        state.ts_count = 0;
    }

    /// Single grouped barrier: previous writes (transfer or compute)
    /// made visible to subsequent reads/writes.
    fn stream_barrier(
        &self,
        cmd: vk::CommandBuffer,
        dst_stage: vk::PipelineStageFlags,
        dst_access: vk::AccessFlags,
    ) {
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(dst_access);
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// Upload enqueued into the stream (no submit).
    pub fn stream_upload(&self, dst: &GpuBuffer, data: &[u8]) -> Result<()> {
        self.stream_upload_at(dst, 0, data)
    }

    /// Upload into a target buffer **region**.
    pub fn stream_upload_at(&self, dst: &GpuBuffer, dst_offset: u64, data: &[u8]) -> Result<()> {
        anyhow::ensure!(
            dst_offset + data.len() as u64 <= dst.size,
            "upload out of bounds: {dst_offset}+{} on {}",
            data.len(),
            dst.size
        );
        if data.is_empty() {
            return Ok(());
        }
        crate::stats::record_up(data.len() as u64);
        let mut staging = self.acquire_staging_upload(data.len() as u64)?;
        staging.write_mapped(data)?;
        let mut state = self.stream.lock().unwrap();
        let cmd = self.stream_cmd(&mut state)?;
        // The destination may be a buffer an earlier command already touched —
        // a pooled buffer handed out again, or a tensor filled by several
        // uploads. Without this the only barriers in the stream are the ones
        // dispatch records, and those do not cover a transfer write: two copies
        // into the same buffer may then overlap and land in either order.
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            let region = vk::BufferCopy::default()
                .dst_offset(dst_offset)
                .size(data.len() as u64);
            self.device
                .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
        }
        state.staging.push(staging);
        Ok(())
    }

    /// Dispatch enqueued into the stream (no submit), whole buffers bound.
    pub fn stream_dispatch(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&GpuBuffer],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let slices: Vec<BufferSlice<'_>> = buffers.iter().map(|b| (*b).into()).collect();
        self.stream_dispatch_slices(pipeline, &slices, push_constants, groups)
    }

    /// Like [`Self::stream_dispatch`], but each binding starts at an offset
    /// inside its buffer (tensors that are slices of an ORT allocation).
    pub fn stream_dispatch_slices(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[BufferSlice<'_>],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<()> {
        let mut state = self.stream.lock().unwrap();
        let cmd = self.stream_cmd(&mut state)?;
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        self.record_dispatch(cmd, pipeline, buffers, push_constants, groups)?;
        self.write_timestamp(cmd, &mut state);
        Ok(())
    }

    /// GPU→GPU copy enqueued into the stream.
    pub fn stream_copy(&self, src: &GpuBuffer, dst: &GpuBuffer, bytes: u64) -> Result<()> {
        self.stream_copy_range(src, 0, dst, 0, bytes)
    }

    /// GPU→GPU copy of a **region**: used when a tensor occupies only a
    /// portion of a larger buffer (ORT, with the memory pattern enabled,
    /// allocates a single block and assigns slices to individual tensors).
    pub fn stream_copy_range(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        dst: &GpuBuffer,
        dst_offset: u64,
        bytes: u64,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            src_offset + bytes <= src.size && dst_offset + bytes <= dst.size,
            "copy out of bounds: src {src_offset}+{bytes}/{}, dst {dst_offset}+{bytes}/{}",
            src.size,
            dst.size
        );
        let mut state = self.stream.lock().unwrap();
        let cmd = self.stream_cmd(&mut state)?;
        // both sides: the copy reads the source and writes the destination, and
        // either can be a buffer a previous command in the stream touched
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            let region = vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(bytes);
            self.device
                .cmd_copy_buffer(cmd, src.buffer, dst.buffer, &[region]);
        }
        Ok(())
    }

    /// Destroys the buffer when safe: immediately if the stream is empty,
    /// otherwise after flush.
    pub fn defer_destroy(&self, buffer: GpuBuffer) {
        let mut state = self.stream.lock().unwrap();
        if state.cmd.is_some() {
            state.graveyard.push(buffer);
        } else {
            drop(state);
            self.destroy_buffer(buffer);
        }
    }

    /// Submit + fence wait + release of deferred resources.
    pub fn flush(&self) -> Result<()> {
        let mut state = self.stream.lock().unwrap();
        let Some(cmd) = state.cmd.take() else {
            return Ok(()); // empty stream
        };
        let device = &self.device;
        let _guard = self.submit_lock.lock().unwrap();
        let wall = std::time::Instant::now();
        let result = unsafe {
            device.end_command_buffer(cmd)?;
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let r = device
                .queue_submit(self.queue, &[submit], fence)
                .and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &cmds);
            r
        };
        if self.profiling_on() {
            crate::stats::record_flush(wall.elapsed().as_nanos() as u64);
            self.collect_timestamps(&mut state);
        }
        let staging: Vec<_> = state.staging.drain(..).collect();
        let graveyard: Vec<_> = state.graveyard.drain(..).collect();
        drop(state); // destroy_buffer re-acquires the allocator lock
        // GPU idle after the fence: descriptor sets are free to reuse
        self.reset_descriptors();
        // staging buffers return to the pool: the fence has passed, the GPU no longer reads them
        for b in staging {
            self.release_staging(b, true);
        }
        for b in graveyard {
            self.destroy_buffer(b);
        }
        result.context("flush stream Vulkan")?;
        Ok(())
    }

    /// Readback: stream flush + synchronous copy.
    pub fn stream_download(&self, src: &GpuBuffer, bytes: usize) -> Result<Vec<u8>> {
        self.stream_download_at(src, 0, bytes)
    }

    /// Download from a **region** of the source buffer.
    pub fn stream_download_at(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        anyhow::ensure!(
            src_offset + bytes as u64 <= src.size,
            "download out of bounds: {src_offset}+{bytes} on {}",
            src.size
        );
        crate::stats::record_down(bytes as u64);
        // barrier + copy to staging recorded into the stream, then flush
        let staging = self.acquire_staging_download(bytes as u64)?;
        {
            let mut state = self.stream.lock().unwrap();
            let cmd = self.stream_cmd(&mut state)?;
            self.stream_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            );
            unsafe {
                let region = vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .size(bytes as u64);
                self.device
                    .cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[region]);
            }
        }
        self.flush()?;
        let data = staging.read_mapped(bytes)?;
        self.release_staging(staging, false);
        Ok(data)
    }
}
