//! Device-local buffer storage with staging upload/readback.

use crate::context::VkContext;
use anyhow::{Context as _, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

/// Free staging buffers divided by direction: buffer usage flag is not
/// interchangeable between upload and download.
#[derive(Default)]
pub(crate) struct StagingPool {
    pub(crate) upload: Vec<GpuBuffer>,
    pub(crate) download: Vec<GpuBuffer>,
}

/// Free device-local storage buffers, indexed by **exact** size.
///
/// Exact and not best-fit on purpose: these buffers are whole tensors, and a
/// best-fit would hand a 98 MB allocation to a request for a few hundred bytes.
/// Repeated shapes — every layer of a transformer or a CNN produces the same
/// sizes as the previous one — make the exact match hit anyway.
/// **First in, first out**, and that is a correctness property for a decode
/// loop, not a preference. A step asks for the same sizes in the same order as
/// the step before it; taking from the back hands the requests their buffers in
/// reverse each time, so a given tensor alternates between two buffers with
/// period two and a descriptor set recorded on one step points at the wrong
/// memory on the next. Taking from the front reproduces the previous step's
/// assignment exactly.
#[derive(Default)]
pub(crate) struct StoragePool {
    free: std::collections::HashMap<u64, std::collections::VecDeque<GpuBuffer>>,
}

impl StoragePool {
    fn take(&mut self, size: u64) -> Option<GpuBuffer> {
        self.free.get_mut(&size)?.pop_front()
    }

    fn put(&mut self, buffer: GpuBuffer) {
        self.free.entry(buffer.size).or_default().push_back(buffer);
    }

    fn canonicalize(&mut self) {
        for list in self.free.values_mut() {
            list.make_contiguous().sort_by_key(|buffer| buffer.id);
        }
    }

    fn drain(&mut self) -> Vec<GpuBuffer> {
        self.free.drain().flat_map(|(_, list)| list).collect()
    }
}

pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub size: u64,
    /// Identity, assigned once when the buffer is created and never reused.
    ///
    /// A decode step must be served the same buffers, in the same order, as the
    /// step before it, or a recorded binding points at the wrong memory. FIFO
    /// alone does not give that: a step acquires and releases in different
    /// orders — a kernel frees its scratch before the graph frees the kernel's
    /// input — so the queue comes out of a step permuted, and two live tensors
    /// of the same size swap buffers with period two. The identity is what
    /// `reset_pool_order` sorts by, so that every step starts from the same
    /// pool and the same sequence of requests gets the same answers.
    id: u64,
    allocation: Option<Allocation>,
    /// Device-local tensor buffer, and so counted in the VRAM statistics —
    /// staging buffers live in host memory and are not.
    storage: bool,
}

impl GpuBuffer {
    /// This buffer's identity, see the field.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Writes to mapped bytes (host-visible buffers only).
    pub(crate) fn write_mapped(&mut self, data: &[u8]) -> Result<()> {
        self.allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .context("buffer non mappabile")?[..data.len()]
            .copy_from_slice(data);
        Ok(())
    }

    /// Reads mapped bytes (host-visible buffers only).
    pub(crate) fn read_mapped(&self, len: usize) -> Result<Vec<u8>> {
        Ok(self
            .allocation
            .as_ref()
            .unwrap()
            .mapped_slice()
            .context("buffer non mappabile")?[..len]
            .to_vec())
    }
}

impl VkContext {
    fn create_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<GpuBuffer> {
        let device = &self.device;
        let buffer = unsafe {
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        }?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mut guard = self.allocator.lock().unwrap();
        let allocator = guard.as_mut().context("allocator distrutto")?;
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "vk-compute buffer",
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;
        Ok(GpuBuffer {
            buffer,
            size,
            id: self.next_id(),
            allocation: Some(allocation),
            storage: false,
        })
    }

    /// Device-local storage buffer (for GPU tensors), taken from the pool of
    /// buffers already freed when one of the same size is there.
    pub fn create_storage_buffer(&self, size: u64) -> Result<GpuBuffer> {
        if let Some(buffer) = self.storage_pool.lock().unwrap().take(size) {
            crate::stats::record_storage_pool_hit();
            return Ok(buffer);
        }
        let mut buffer = self.create_buffer(
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        buffer.storage = true;
        crate::stats::record_storage_alloc(size);
        Ok(buffer)
    }

    fn next_id(&self) -> u64 {
        self.buffer_serial
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Puts every free buffer back in a canonical order.
    ///
    /// Called at the start of a step that has to be served exactly like the one
    /// before it. Within a step the pool is a queue and the order of requests
    /// decides everything; between steps the queue is left permuted by the
    /// interleaving of acquisitions and releases, and this undoes that.
    pub fn reset_pool_order(&self) {
        let mut pool = self.storage_pool.lock().unwrap();
        pool.canonicalize();
        if std::env::var_os("POOL_PROBE").is_some() {
            for (size, list) in pool.free.iter() {
                if *size == 4608 {
                    let ids: Vec<u64> = list.iter().map(|b| b.id).collect();
                    eprintln!("POOL size {size}: {} free, ids {ids:?}", list.len());
                }
            }
        }
    }

    /// Returns a storage buffer to the pool instead of destroying it.
    ///
    /// Safe while the command buffer is open: every dispatch and every copy in
    /// the stream is preceded by a full memory barrier, so a dispatch recorded
    /// later can neither read nor write the buffer before the ones recorded
    /// earlier are done with it.
    ///
    /// The caller must hold the only reference — anything else keeps pointing
    /// at a `VkBuffer` that the next allocation is free to overwrite.
    pub fn recycle_storage_buffer(&self, buffer: GpuBuffer) {
        self.storage_pool.lock().unwrap().put(buffer);
    }

    /// Hands the pooled buffers to `defer_destroy`: the stream may still hold
    /// dispatches that read them, so destroying them here would be a use after
    /// free on the GPU side.
    pub fn drain_storage_pool(&self) {
        let buffers = self.storage_pool.lock().unwrap().drain();
        for buffer in buffers {
            self.defer_destroy(buffer);
        }
    }

    /// Hands the free buffers to the caller, leaving the pool empty.
    ///
    /// Unlike `drain_storage_pool` these are not destroyed: the caller becomes
    /// their owner, which is how a replay plan keeps the buffers its bindings
    /// name out of everyone else's reach.
    pub fn take_storage_pool(&self) -> Vec<GpuBuffer> {
        self.storage_pool.lock().unwrap().drain()
    }

    /// Empties the pool (called on context destruction).
    pub(crate) fn destroy_storage_pool(&self) {
        let buffers = self.storage_pool.lock().unwrap().drain();
        for buffer in buffers {
            self.destroy_buffer(buffer);
        }
    }

    /// Host-visible staging buffer for uploads, acquired from pool when possible.
    ///
    /// Allocating a host-visible buffer for every transfer costs a
    /// `vkAllocateMemory` (or suballocation) and `vkCreateBuffer` per
    /// run: sizes repeat across runs, so buffers are reused. Sizes are
    /// rounded to powers of two to prevent pool fragmentation.
    pub(crate) fn acquire_staging_upload(&self, size: u64) -> Result<GpuBuffer> {
        self.acquire_staging(size, true)
    }

    /// Host-visible staging buffer for downloads, acquired from pool when possible.
    pub(crate) fn acquire_staging_download(&self, size: u64) -> Result<GpuBuffer> {
        self.acquire_staging(size, false)
    }

    fn acquire_staging(&self, size: u64, upload: bool) -> Result<GpuBuffer> {
        {
            let mut pool = self.staging_pool.lock().unwrap();
            let list = if upload {
                &mut pool.upload
            } else {
                &mut pool.download
            };
            if let Some(index) = list.iter().position(|b| b.size >= size) {
                return Ok(list.swap_remove(index));
            }
        }
        // minimum 64 KiB: below that threshold the pool would fill up with
        // many different sizes, all hard to reuse
        let capacity = size.max(64 * 1024).next_power_of_two();
        if upload {
            self.create_buffer(
                capacity,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
            )
        } else {
            self.create_buffer(
                capacity,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
            )
        }
    }

    /// Returns to the pool a staging buffer no longer in use by the GPU.
    pub(crate) fn release_staging(&self, buffer: GpuBuffer, upload: bool) {
        let mut pool = self.staging_pool.lock().unwrap();
        if upload {
            pool.upload.push(buffer);
        } else {
            pool.download.push(buffer);
        }
    }

    /// Empties the pool (called on context destruction).
    pub(crate) fn destroy_staging_pool(&self) {
        let buffers: Vec<GpuBuffer> = {
            let mut pool = self.staging_pool.lock().unwrap();
            let mut buffers: Vec<GpuBuffer> = pool.upload.drain(..).collect();
            buffers.append(&mut pool.download);
            buffers
        };
        for buffer in buffers {
            self.destroy_buffer(buffer);
        }
    }

    /// Synchronous upload: host-visible staging + copy.
    pub fn upload(&self, dst: &GpuBuffer, data: &[u8]) -> Result<()> {
        assert!(data.len() as u64 <= dst.size);
        let mut staging = self.create_buffer(
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging
            .allocation
            .as_mut()
            .unwrap()
            .mapped_slice_mut()
            .context("staging non mappabile")?[..data.len()]
            .copy_from_slice(data);
        self.run_commands(|cmd| unsafe {
            let region = vk::BufferCopy::default().size(data.len() as u64);
            self.device
                .cmd_copy_buffer(cmd, staging.buffer, dst.buffer, &[region]);
            Ok(())
        })?;
        self.destroy_buffer(staging);
        Ok(())
    }

    /// Synchronous readback of the buffer contents.
    pub fn download(&self, src: &GpuBuffer) -> Result<Vec<u8>> {
        let staging = self.create_buffer(
            src.size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
        )?;
        self.run_commands(|cmd| unsafe {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
            let region = vk::BufferCopy::default().size(src.size);
            self.device
                .cmd_copy_buffer(cmd, src.buffer, staging.buffer, &[region]);
            Ok(())
        })?;
        let data = staging
            .allocation
            .as_ref()
            .unwrap()
            .mapped_slice()
            .context("staging non mappabile")?[..src.size as usize]
            .to_vec();
        self.destroy_buffer(staging);
        Ok(data)
    }

    pub fn destroy_buffer(&self, mut buffer: GpuBuffer) {
        if buffer.storage {
            crate::stats::record_storage_free(buffer.size);
        }
        if let Some(allocation) = buffer.allocation.take()
            && let Ok(mut guard) = self.allocator.lock()
            && let Some(allocator) = guard.as_mut()
        {
            let _ = allocator.free(allocation);
        }
        unsafe { self.device.destroy_buffer(buffer.buffer, None) };
    }
}
