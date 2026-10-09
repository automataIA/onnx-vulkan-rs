//! Deferred command stream.
//!
//! Dispatches and copies are recorded into a command buffer; the host waits
//! (on fences) only on `flush()` — typically when a result must return to
//! CPU. Drastically reduces `vkQueueSubmit` calls and per-op waits
//! (NVIDIA/Khronos best practice).
//!
//! A long stream is submitted in chunks while it is being recorded, without
//! waiting: otherwise the GPU idles for the whole host walk of the graph
//! (SAM 3's detector: ~9 ms of recording before a single 34 ms submit). The
//! chunks run in submission order and every command opens with a barrier on
//! everything before it, which by the spec reaches back into earlier
//! submissions on the queue; what a chunk keeps alive (staging, descriptor
//! sets, buffers to destroy) is released only at `flush()`, after every fence.

use crate::buffer::GpuBuffer;
use crate::context::VkContext;
use crate::pipeline::{BufferSlice, ComputePipeline};
use anyhow::{Context as _, Result};
use ash::vk;

/// Dispatches recorded before the first chunk is submitted: few, so the GPU
/// starts early. Each later chunk doubles, up to [`MAX_CHUNK`]. Swept on the
/// SAM 3 chain (later prompt, RTX 4070): 8/64 and 8/128 at 40.1 ms, 32/512
/// at 42.5, 64/1024 at 42.4, one buffer at 49.7: wherever the host records
/// slower than the GPU runs, a smaller chunk is less work the GPU waits for.
const FIRST_CHUNK: u32 = 8;
const MAX_CHUNK: u32 = 128;

/// Timestamps the `VULKAN_GPU_SPAN` probe keeps per flush: a begin/end pair
/// per command buffer.
const SPAN_CAPACITY: u32 = 256;

/// `VULKAN_GPU_SPAN=1`: two timestamps per submitted command buffer, chunked
/// submission left on (unlike `VULKAN_EP_STATS`), and at each flush the GPU
/// busy time (union of the command buffers' intervals) against their span.
/// The difference is time the GPU waited for the host to record and submit.
fn span_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VULKAN_GPU_SPAN").is_some())
}

#[derive(Default)]
pub(crate) struct StreamState {
    /// Command buffer being recorded (None = nothing being recorded).
    cmd: Option<vk::CommandBuffer>,
    /// Chunks submitted since the last flush, each with its fence.
    in_flight: Vec<(vk::CommandBuffer, vk::Fence)>,
    /// Unsignalled fences free for the next chunk.
    free_fences: Vec<vk::Fence>,
    /// Dispatches recorded into `cmd`, and how many submit it (0 = first chunk).
    recorded: u32,
    chunk: u32,
    /// Staging buffers kept alive until flush.
    staging: Vec<GpuBuffer>,
    /// Buffers to destroy after flush.
    graveyard: Vec<GpuBuffer>,
    /// Profiling: op for each timed dispatch (index = timestamp i→i+1).
    ts_labels: Vec<&'static str>,
    /// Timestamps written into the current command buffer (0 = no baseline).
    ts_count: u32,
    /// `VULKAN_GPU_SPAN`: timestamps written since the last flush (odd = the
    /// current command buffer still has to write its end).
    span_count: u32,
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
        if self.span_on() && state.span_count + 2 <= SPAN_CAPACITY {
            let pool = self.ensure_query_pool()?;
            unsafe {
                if state.span_count == 0 {
                    device.cmd_reset_query_pool(cmd, pool, 0, SPAN_CAPACITY);
                }
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    pool,
                    state.span_count,
                );
            }
            state.span_count += 1;
        }
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

    /// The `VULKAN_GPU_SPAN` probe: never with profiling, which owns the pool.
    fn span_on(&self) -> bool {
        span_probe() && !self.profiling_on() && self.timestamp_period > 0.0
    }

    /// Ends the probe's interval for the command buffer about to be closed.
    fn close_span(&self, cmd: vk::CommandBuffer, state: &mut StreamState) {
        if state.span_count.is_multiple_of(2) {
            return;
        }
        if let Some(pool) = *self.query_pool.lock().unwrap() {
            unsafe {
                self.device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    pool,
                    state.span_count,
                );
            }
            state.span_count += 1;
        }
    }

    /// Logs the probe's busy time and span for the flush just waited on.
    fn report_span(&self, state: &mut StreamState, host_wait: std::time::Duration) {
        let count = std::mem::take(&mut state.span_count);
        let Some(pool) = *self.query_pool.lock().unwrap() else {
            return;
        };
        if count < 2 {
            return;
        }
        let mut ts = vec![0u64; count as usize];
        let read = unsafe {
            self.device
                .get_query_pool_results(pool, 0, &mut ts, vk::QueryResultFlags::TYPE_64)
        };
        if read.is_err() {
            return;
        }
        // in ms from the first begin, which wrapping counters only allow as deltas
        let ms = |t: u64| {
            let ticks = crate::pipeline::timestamp_delta_ticks(ts[0], t, self.timestamp_valid_bits);
            ticks as f64 * f64::from(self.timestamp_period) / 1e6
        };
        let mut intervals: Vec<(f64, f64)> =
            ts.chunks_exact(2).map(|p| (ms(p[0]), ms(p[1]))).collect();
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (mut busy, mut end) = (0.0, f64::MIN);
        for (i, &(a, b)) in intervals.iter().enumerate() {
            if i > 0 && a > end {
                log::debug!("gpu span: idle {:.3} ms before command buffer {i}", a - end);
            }
            let start = a.max(end);
            if b > start {
                busy += b - start;
            }
            end = end.max(b);
        }
        log::info!(
            "gpu span: {} command buffers, busy {busy:.3} ms of {end:.3} ms, host waited {:.3} ms",
            intervals.len(),
            host_wait.as_secs_f64() * 1e3
        );
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
                let ticks = crate::pipeline::timestamp_delta_ticks(
                    ts[i],
                    ts[i + 1],
                    self.timestamp_valid_bits,
                );
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
        self.stream_upload_at(dst, 0, data, None)
    }

    /// Upload of a value that has a name, which a capture keeps so a replay can
    /// refresh the payload from the same value at the next step.
    pub fn stream_upload_labeled(&self, dst: &GpuBuffer, data: &[u8], label: &str) -> Result<()> {
        self.stream_upload_at(dst, 0, data, Some(label))
    }

    /// Upload into a target buffer **region**.
    pub fn stream_upload_at(
        &self,
        dst: &GpuBuffer,
        dst_offset: u64,
        data: &[u8],
        label: Option<&str>,
    ) -> Result<()> {
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
        let span = crate::stats::span("upload staging");
        let mut staging = self.acquire_staging_upload(data.len() as u64)?;
        staging.write_mapped(data)?;
        drop(span);
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
        drop(state);
        self.capture_op(|| {
            crate::StreamOp::Upload(crate::UploadOp::new(dst.buffer, dst_offset, data, label))
        });
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
        crate::stats::record_dispatch_count();
        let started = crate::stats::enabled().then(std::time::Instant::now);
        let mut state = self.stream.lock().unwrap();
        let cmd = self.stream_cmd(&mut state)?;
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
        self.record_dispatch(cmd, pipeline, buffers, push_constants, groups)?;
        self.write_timestamp(cmd, &mut state);
        state.recorded += 1;
        if state.recorded >= state.chunk.max(FIRST_CHUNK) && !self.profiling_on() {
            // profiling keeps one command buffer: its timestamps share a
            // query pool reset at the start of the buffer and read at flush
            self.submit_chunk(&mut state)?;
        }
        drop(state);
        self.capture_op(|| {
            let bindings = buffers.iter().map(|b| self.binding_info(b)).collect();
            crate::StreamOp::Dispatch(crate::DispatchOp::new(
                pipeline,
                bindings,
                push_constants,
                groups,
            ))
        });
        if let Some(started) = started {
            crate::stats::record_recording(started.elapsed().as_nanos() as u64);
        }
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
        drop(state);
        self.capture_op(|| {
            crate::StreamOp::Copy(crate::CopyOp {
                src: src.buffer,
                src_offset,
                dst: dst.buffer,
                dst_offset,
                bytes,
            })
        });
        Ok(())
    }

    /// Sets `bytes` of `dst` from `dst_offset` to the repeated word `value`,
    /// enqueued into the stream: a cleared accumulator without a dispatch or
    /// an upload. `dst_offset` and `bytes` are multiples of 4.
    pub fn stream_fill(
        &self,
        dst: &GpuBuffer,
        dst_offset: u64,
        bytes: u64,
        value: u32,
    ) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        anyhow::ensure!(
            dst_offset.is_multiple_of(4)
                && bytes.is_multiple_of(4)
                && dst_offset + bytes <= dst.size,
            "fill out of bounds or unaligned: {dst_offset}+{bytes} on {}",
            dst.size
        );
        let mut state = self.stream.lock().unwrap();
        let cmd = self.stream_cmd(&mut state)?;
        self.record_fill(cmd, dst.buffer, dst_offset, bytes, value);
        drop(state);
        self.capture_op(|| {
            crate::StreamOp::Fill(crate::FillOp {
                dst: dst.buffer,
                dst_offset,
                bytes,
                value,
            })
        });
        Ok(())
    }

    fn record_fill(
        &self,
        cmd: vk::CommandBuffer,
        dst: vk::Buffer,
        offset: u64,
        bytes: u64,
        value: u32,
    ) {
        // the destination may be a pooled buffer an earlier command read or wrote
        self.stream_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            self.device.cmd_fill_buffer(cmd, dst, offset, bytes, value);
        }
    }

    /// Records the ops of the next commands instead of only issuing them.
    ///
    /// A capture already in progress is replaced: the caller that opened it is
    /// the one that closes it, and nesting two would make the inner one steal
    /// the outer one's commands.
    pub fn begin_capture(&self) {
        *self.capture.lock().unwrap() = Some(Vec::new());
    }

    /// Closes the capture and returns what was recorded, `None` if none was open.
    pub fn end_capture(&self) -> Option<Vec<crate::StreamOp>> {
        self.capture.lock().unwrap().take()
    }

    /// How many commands the open capture holds, `None` if none is open.
    ///
    /// A caller that walks a graph uses it to record which ops each node
    /// produced, which is what lets a plan re-run one node and replay another.
    pub fn capture_len(&self) -> Option<usize> {
        self.capture.lock().unwrap().as_ref().map(Vec::len)
    }

    fn capture_op(&self, build: impl FnOnce() -> crate::StreamOp) {
        let mut guard = self.capture.lock().unwrap();
        if let Some(ops) = guard.as_mut() {
            ops.push(build());
        }
    }

    /// Issues captured commands again, in order.
    ///
    /// The barriers are the ones the original calls placed, because they are a
    /// property of the command and not of how it was decided: every dispatch is
    /// preceded by a full memory barrier, every transfer by a transfer one.
    ///
    /// # Safety of the handles
    ///
    /// The ops hold `VkBuffer` and `VkPipeline` handles, and nothing here can
    /// tell whether they still name live objects. See `capture.rs`: keeping
    /// them alive is the caller's contract.
    pub fn replay(&self, ops: &[crate::StreamOp]) -> Result<()> {
        let mut state = self.stream.lock().unwrap();
        let cmd = self.stream_cmd(&mut state)?;
        for op in ops {
            match op {
                crate::StreamOp::Dispatch(dispatch) => {
                    crate::stats::record_dispatch_count();
                    self.stream_barrier(
                        cmd,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                    );
                    self.record_dispatch_op(cmd, dispatch)?;
                    self.write_timestamp(cmd, &mut state);
                }
                crate::StreamOp::Upload(upload) => {
                    if upload.bytes.is_empty() {
                        continue;
                    }
                    crate::stats::record_up(upload.bytes.len() as u64);
                    let mut staging = self.acquire_staging_upload(upload.bytes.len() as u64)?;
                    staging.write_mapped(&upload.bytes)?;
                    self.stream_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                    );
                    unsafe {
                        let region = vk::BufferCopy::default()
                            .dst_offset(upload.dst_offset)
                            .size(upload.bytes.len() as u64);
                        self.device
                            .cmd_copy_buffer(cmd, staging.buffer, upload.dst, &[region]);
                    }
                    state.staging.push(staging);
                }
                crate::StreamOp::Copy(copy) => {
                    self.stream_barrier(
                        cmd,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
                    );
                    unsafe {
                        let region = vk::BufferCopy::default()
                            .src_offset(copy.src_offset)
                            .dst_offset(copy.dst_offset)
                            .size(copy.bytes);
                        self.device
                            .cmd_copy_buffer(cmd, copy.src, copy.dst, &[region]);
                    }
                }
                crate::StreamOp::Fill(fill) => {
                    self.record_fill(cmd, fill.dst, fill.dst_offset, fill.bytes, fill.value);
                }
            }
        }
        Ok(())
    }

    /// Submits what is recorded without waiting, and starts the next chunks
    /// again from [`FIRST_CHUNK`] dispatches. Called at the end of a run whose
    /// outputs may stay on the device: without it the next run inherits
    /// [`MAX_CHUNK`]-dispatch chunks, and the previous run's tail waits for
    /// them to fill: on SAM 3's resident chain (text encoder then detector,
    /// one flush) the later prompt went 39.0 → 38.5 ms. A no-op while
    /// profiling, which keeps one buffer.
    pub fn stream_break(&self) -> Result<()> {
        if self.profiling_on() {
            return Ok(());
        }
        let mut state = self.stream.lock().unwrap();
        self.submit_chunk(&mut state)?;
        state.chunk = 0;
        Ok(())
    }

    /// Submits the command buffer being recorded without waiting for it.
    fn submit_chunk(&self, state: &mut StreamState) -> Result<()> {
        let Some(cmd) = state.cmd else {
            return Ok(());
        };
        let device = &self.device;
        let _guard = self.submit_lock.lock().unwrap();
        let fence = match state.free_fences.pop() {
            Some(fence) => fence,
            None => unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? },
        };
        self.close_span(cmd, state);
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        let result = unsafe {
            device
                .end_command_buffer(cmd)
                .and_then(|()| device.queue_submit(self.queue, &[submit], fence))
        };
        if let Err(error) = result {
            // `cmd` stays for `flush` to free (and report again)
            state.free_fences.push(fence);
            return Err(error).context("submit of a stream chunk");
        }
        crate::stats::record_submit();
        state.cmd = None;
        state.in_flight.push((cmd, fence));
        state.recorded = 0;
        state.chunk = (state.chunk.max(FIRST_CHUNK) * 2).min(MAX_CHUNK);
        Ok(())
    }

    /// Destroys the chunk fences (called on context destruction, device idle).
    pub(crate) fn destroy_stream_fences(&self) {
        let mut state = self.stream.lock().unwrap();
        let in_flight: Vec<_> = state.in_flight.drain(..).map(|(_, f)| f).collect();
        for fence in state.free_fences.drain(..).chain(in_flight) {
            unsafe { self.device.destroy_fence(fence, None) };
        }
    }

    /// Destroys the buffer when safe: immediately if the stream is empty,
    /// otherwise after flush.
    pub fn defer_destroy(&self, buffer: GpuBuffer) {
        let mut state = self.stream.lock().unwrap();
        if state.cmd.is_some() || !state.in_flight.is_empty() {
            state.graveyard.push(buffer);
        } else {
            drop(state);
            self.destroy_buffer(buffer);
        }
    }

    /// Submit + fence wait + release of deferred resources.
    pub fn flush(&self) -> Result<()> {
        let mut state = self.stream.lock().unwrap();
        let tail = state.cmd.take();
        if tail.is_none() && state.in_flight.is_empty() {
            return Ok(()); // empty stream
        }
        let device = &self.device;
        let _guard = self.submit_lock.lock().unwrap();
        let wall = std::time::Instant::now();
        let fence = self.flush_fence;
        if let Some(cmd) = tail {
            self.close_span(cmd, &mut state);
        }
        let in_flight: Vec<_> = state.in_flight.drain(..).collect();
        let chunk_fences: Vec<vk::Fence> = in_flight.iter().map(|&(_, f)| f).collect();
        let result = unsafe {
            let mut fences = chunk_fences.clone();
            let mut r = Ok(());
            if let Some(cmd) = tail {
                let cmds = [cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds);
                // reset before submitting, not after waiting: the previous flush
                // left it signalled, and a fence that is never reset is waited on
                // once and then satisfied instantly forever
                r = device
                    .end_command_buffer(cmd)
                    .and_then(|()| device.reset_fences(&[fence]))
                    .and_then(|()| device.queue_submit(self.queue, &[submit], fence));
                if r.is_ok() {
                    crate::stats::record_submit();
                    fences.push(fence);
                }
            }
            // waited on even after a failed submit: earlier chunks may still run
            if !fences.is_empty() {
                r = r.and(device.wait_for_fences(&fences, true, u64::MAX));
            }
            let mut cmds: Vec<_> = in_flight.iter().map(|&(c, _)| c).collect();
            cmds.extend(tail);
            device.free_command_buffers(self.command_pool, &cmds);
            if !chunk_fences.is_empty() && device.reset_fences(&chunk_fences).is_ok() {
                state.free_fences.extend(chunk_fences);
            }
            r
        };
        state.recorded = 0;
        state.chunk = 0;
        if self.profiling_on() {
            crate::stats::record_flush(wall.elapsed().as_nanos() as u64);
            self.collect_timestamps(&mut state);
        } else if state.span_count > 0 {
            self.report_span(&mut state, wall.elapsed());
        }
        let _retire = crate::stats::span("fence retire");
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

    /// Reads several device buffers with one queue submission and one fence.
    ///
    /// A model may expose dozens of outputs (a decoder has one key/value pair
    /// per layer). Calling [`Self::stream_download`] for each one serializes a
    /// submit and fence per output. This records every copy first, flushes the
    /// shared stream once, then reads all staging buffers after that fence.
    pub fn stream_download_many(&self, sources: &[(&GpuBuffer, usize)]) -> Result<Vec<Vec<u8>>> {
        let staging = self.stage_downloads(sources)?;
        let mut outputs = Vec::with_capacity(staging.len());
        let mut first_error = None;
        for (buffer, &(_, bytes)) in staging.into_iter().zip(sources) {
            let Some(buffer) = buffer else {
                outputs.push(Vec::new());
                continue;
            };
            let data = {
                let _span = crate::stats::span("readback copy");
                buffer.read_mapped(bytes)
            };
            self.release_staging(buffer, false);
            match data {
                Ok(data) => outputs.push(data),
                Err(error) => {
                    first_error.get_or_insert(error);
                    outputs.push(Vec::new());
                }
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(outputs),
        }
    }

    /// Like [`Self::stream_download_many`], into memory the caller already
    /// owns: each source is read into its slice, `len()` bytes from offset 0.
    ///
    /// A fresh `Vec` per output pays first-touch page faults on every run —
    /// the copy of SAM 3 detector's 66.7 MB took 42 ms, 11 ms into memory already
    /// touched. A caller that keeps its output buffers across runs pays them
    /// once.
    pub fn stream_download_many_into(&self, sources: &mut [(&GpuBuffer, &mut [u8])]) -> Result<()> {
        let sizes: Vec<(&GpuBuffer, usize)> =
            sources.iter().map(|(src, out)| (*src, out.len())).collect();
        let staging = self.stage_downloads(&sizes)?;
        let mut first_error = None;
        for (buffer, (_, out)) in staging.into_iter().zip(sources.iter_mut()) {
            let Some(buffer) = buffer else {
                continue;
            };
            let read = {
                let _span = crate::stats::span("readback copy");
                buffer.read_mapped_into(out)
            };
            self.release_staging(buffer, false);
            if let Err(error) = read {
                first_error.get_or_insert(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Records a copy of each source's first `bytes` into its own staging
    /// buffer and flushes the stream once. A zero-byte source gets `None`.
    /// The caller reads each staging buffer and returns it to the pool.
    fn stage_downloads(&self, sources: &[(&GpuBuffer, usize)]) -> Result<Vec<Option<GpuBuffer>>> {
        if sources.is_empty() {
            return Ok(Vec::new());
        }
        for &(src, bytes) in sources {
            anyhow::ensure!(
                bytes as u64 <= src.size,
                "download out of bounds: {bytes} on {}",
                src.size
            );
        }

        let mut staging = Vec::with_capacity(sources.len());
        for &(src, bytes) in sources {
            crate::stats::record_down(bytes as u64);
            if bytes == 0 {
                staging.push(None);
                continue;
            }
            let staged = {
                let _span = crate::stats::span("readback staging");
                self.acquire_staging_download(bytes as u64)
            };
            let buffer = match staged {
                Ok(buffer) => buffer,
                Err(error) => {
                    // Only recycle staging after the fence proves that no
                    // queued copy still references it. On the compound error
                    // path, leaking a buffer is safer than reusing live memory.
                    if self.flush().is_ok() {
                        for buffer in staging.drain(..).flatten() {
                            self.release_staging(buffer, false);
                        }
                    }
                    return Err(error);
                }
            };
            if let Err(error) = self.stream_copy_range(src, 0, &buffer, 0, bytes as u64) {
                // A recorded copy may already reference an earlier staging
                // buffer. Flush it before returning those buffers to the pool.
                if self.flush().is_ok() {
                    self.release_staging(buffer, false);
                    for buffer in staging.drain(..).flatten() {
                        self.release_staging(buffer, false);
                    }
                }
                return Err(error);
            }
            staging.push(Some(buffer));
        }

        self.flush()?;
        Ok(staging)
    }

    /// Download from a **region** of the source buffer.
    pub fn stream_download_at(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        bytes: usize,
    ) -> Result<Vec<u8>> {
        let mut data = vec![0u8; bytes];
        self.stream_download_into(src, src_offset, &mut data)?;
        Ok(data)
    }

    /// Like `stream_download_at`, into memory the caller already owns.
    ///
    /// A host adapter that has its own destination — ORT's output tensor —
    /// would otherwise pay for a fresh `Vec` and a second copy. The `Vec` is
    /// the expensive part: a 67 MB one is a fresh `mmap` whose page faults cost
    /// ~34 ms per run (SAM 3's `pred_masks`), three times the copy itself.
    pub fn stream_download_into(
        &self,
        src: &GpuBuffer,
        src_offset: u64,
        out: &mut [u8],
    ) -> Result<()> {
        let bytes = out.len();
        anyhow::ensure!(
            src_offset + bytes as u64 <= src.size,
            "download out of bounds: {src_offset}+{bytes} on {}",
            src.size
        );
        crate::stats::record_down(bytes as u64);
        // barrier + copy to staging recorded into the stream, then flush
        let staging = {
            let _span = crate::stats::span("readback staging");
            self.acquire_staging_download(bytes as u64)?
        };
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
        let read = {
            let _span = crate::stats::span("readback copy");
            staging.read_mapped_into(out)
        };
        self.release_staging(staging, false);
        read
    }
}
