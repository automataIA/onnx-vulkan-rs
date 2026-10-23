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
///
/// Beside them, one **scratch** buffer (see `take_scratch_buffer`): a
/// temporary that has a different size at every node — an `im2col` — would
/// otherwise leave one buffer per distinct size in the exact-size lists.
#[derive(Default)]
pub(crate) struct StoragePool {
    free: std::collections::HashMap<u64, std::collections::VecDeque<GpuBuffer>>,
    scratch: Option<GpuBuffer>,
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
        let scratch = self.scratch.take();
        self.free
            .drain()
            .flat_map(|(_, list)| list)
            .chain(scratch)
            .collect()
    }

    /// Keeps the smallest buffers whose sizes add up to at most `budget` and
    /// returns the others.
    ///
    /// Smallest first because what the pool saves is a call per allocation,
    /// and a byte of budget kept as small buffers saves more of them. The
    /// buffers kept are queued by identity, as `canonicalize` would leave them.
    /// The scratch is kept first, when it fits, and its size comes out of the
    /// budget.
    fn trim(&mut self, mut budget: u64) -> Vec<GpuBuffer> {
        let scratch = self.scratch.take();
        let mut buffers = self.drain();
        match scratch {
            Some(scratch) if scratch.size <= budget => {
                budget -= scratch.size;
                self.scratch = Some(scratch);
            }
            scratch => buffers.extend(scratch),
        }
        buffers.sort_by_key(|buffer| (buffer.size, buffer.id));
        let mut kept = 0u64;
        let keep = buffers
            .iter()
            .take_while(|buffer| {
                kept += buffer.size;
                kept <= budget
            })
            .count();
        let evicted = buffers.split_off(keep);
        for buffer in buffers {
            self.put(buffer);
        }
        evicted
    }

    /// Takes the scratch when it holds at least `size` bytes; a smaller one is
    /// returned as `Err` for the caller to destroy.
    fn take_scratch(&mut self, size: u64) -> Result<Option<GpuBuffer>, GpuBuffer> {
        match self.scratch.take() {
            Some(scratch) if scratch.size >= size => Ok(Some(scratch)),
            Some(smaller) => Err(smaller),
            None => Ok(None),
        }
    }

    /// Keeps the larger of `buffer` and the current scratch, returning the
    /// other.
    fn put_scratch(&mut self, buffer: GpuBuffer) -> Option<GpuBuffer> {
        match self.scratch.take() {
            Some(current) if current.size >= buffer.size => {
                self.scratch = Some(current);
                Some(buffer)
            }
            current => {
                self.scratch = Some(buffer);
                current
            }
        }
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

    /// Reads the first `out.len()` mapped bytes into `out` (host-visible
    /// buffers only).
    pub(crate) fn read_mapped_into(&self, out: &mut [u8]) -> Result<()> {
        out.copy_from_slice(
            &self
                .allocation
                .as_ref()
                .unwrap()
                .mapped_slice()
                .context("buffer non mappabile")?[..out.len()],
        );
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
        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name: "vk-compute buffer",
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(allocation) => allocation,
            Err(error) => {
                // the handle has no memory yet: without this every failed
                // allocation, and so every retry, would leak one
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error.into());
            }
        };
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
    ///
    /// When the device is out of memory and the pool holds buffers of other
    /// sizes, they are destroyed and the allocation tried once more: kept
    /// buffers must never be what makes a model fail on a card it fits on.
    /// They may still be bound by commands not yet submitted, hence the flush
    /// before their memory counts as free.
    pub fn create_storage_buffer(&self, size: u64) -> Result<GpuBuffer> {
        if let Some(buffer) = self.storage_pool.lock().unwrap().take(size) {
            crate::stats::record_storage_pool_hit();
            return Ok(buffer);
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let mut buffer = match self.create_buffer(size, usage, MemoryLocation::GpuOnly) {
            Err(error) if out_of_device_memory(&error) => {
                let pooled = self.storage_pool.lock().unwrap().drain();
                if pooled.is_empty() {
                    return Err(error);
                }
                log::warn!(
                    "out of device memory for {size} bytes: releasing {} pooled bytes and retrying",
                    pooled.iter().map(|buffer| buffer.size).sum::<u64>()
                );
                for buffer in pooled {
                    self.defer_destroy(buffer);
                }
                self.flush()?;
                self.create_buffer(size, usage, MemoryLocation::GpuOnly)?
            }
            result => result?,
        };
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

    /// A storage buffer of **at least** `size` bytes, for a temporary written
    /// and consumed by the same kernel sequence: the pool's scratch when it is
    /// large enough, otherwise a new buffer of `size` bytes, the smaller
    /// scratch destroyed after a flush. Return it with
    /// `recycle_scratch_buffer`.
    ///
    /// The kernels that use it must take their extent from push constants, not
    /// from the buffer's length. The scratch is part of the pool for
    /// `take_storage_pool`, `trim_storage_pool` and the out-of-memory retry, so
    /// a replay plan holds it like any pooled buffer it binds.
    pub fn take_scratch_buffer(&self, size: u64) -> Result<GpuBuffer> {
        let taken = self.storage_pool.lock().unwrap().take_scratch(size);
        match taken {
            Ok(Some(scratch)) => {
                crate::stats::record_storage_pool_hit();
                Ok(scratch)
            }
            Ok(None) => self.create_storage_buffer(size),
            Err(smaller) => {
                // the smaller one may still be bound by recorded commands; left
                // to the next flush it stays allocated next to its successor,
                // and a run that grows the scratch node after node without a
                // flush stacks them all (70 MB on Piper's first run). Growth
                // stops once the largest request has been seen.
                let flushed = self.flush();
                self.defer_destroy(smaller);
                flushed?;
                self.create_storage_buffer(size)
            }
        }
    }

    /// Returns a buffer from `take_scratch_buffer`. Same safety contract as
    /// `recycle_storage_buffer`.
    pub fn recycle_scratch_buffer(&self, buffer: GpuBuffer) {
        let displaced = self.storage_pool.lock().unwrap().put_scratch(buffer);
        if let Some(buffer) = displaced {
            self.defer_destroy(buffer);
        }
    }

    /// Called at the end of a run: keeps the smallest pooled buffers within
    /// the context's budget for the next run (see `storage_pool_budget`) and
    /// hands the rest to `defer_destroy` — the stream may still hold dispatches
    /// that read them, so destroying them here would be a use after free on
    /// the GPU side.
    ///
    /// Without it, every run allocates its working set again and frees it at
    /// the end (SAM 3 detector int8, standalone: 373 allocations per run
    /// without the budget, 187 with it).
    pub fn trim_storage_pool(&self) {
        let evicted = self
            .storage_pool
            .lock()
            .unwrap()
            .trim(self.storage_pool_budget);
        for buffer in evicted {
            self.defer_destroy(buffer);
        }
    }

    /// Hands the free buffers to the caller, leaving the pool empty.
    ///
    /// Unlike `trim_storage_pool` these are not destroyed: the caller becomes
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

/// An allocation that failed for lack of device memory, the one failure that
/// freeing pooled buffers can cure.
fn out_of_device_memory(error: &anyhow::Error) -> bool {
    matches!(
        error.downcast_ref::<gpu_allocator::AllocationError>(),
        Some(gpu_allocator::AllocationError::OutOfMemory)
    ) || error.downcast_ref::<vk::Result>() == Some(&vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake(size: u64, id: u64) -> GpuBuffer {
        GpuBuffer {
            buffer: vk::Buffer::null(),
            size,
            id,
            allocation: None,
            storage: true,
        }
    }

    #[test]
    fn trim_keeps_the_smallest_buffers_within_the_budget() {
        let mut pool = StoragePool::default();
        for (size, id) in [(64, 3), (1024, 0), (64, 1), (256, 2), (4096, 4)] {
            pool.put(fake(size, id));
        }
        let mut evicted: Vec<(u64, u64)> = pool
            .trim(400)
            .iter()
            .map(|buffer| (buffer.size, buffer.id))
            .collect();
        evicted.sort_unstable();
        assert_eq!(evicted, [(1024, 0), (4096, 4)]);
        // kept in identity order within a size, whatever order they came back in
        assert_eq!(pool.take(64).map(|buffer| buffer.id), Some(1));
        assert_eq!(pool.take(64).map(|buffer| buffer.id), Some(3));
        assert_eq!(pool.take(256).map(|buffer| buffer.id), Some(2));
        assert!(pool.take(1024).is_none());

        pool.put(fake(64, 5));
        assert_eq!(pool.trim(0).len(), 1, "a zero budget keeps nothing");
    }

    #[test]
    fn the_scratch_grows_to_the_largest_request_and_belongs_to_the_pool() {
        let mut pool = StoragePool::default();
        assert!(matches!(pool.take_scratch(100), Ok(None)));
        assert!(pool.put_scratch(fake(100, 0)).is_none());
        // a larger request hands back the smaller scratch for destruction
        assert_eq!(pool.take_scratch(200).map_err(|b| b.id).err(), Some(0));
        assert!(pool.put_scratch(fake(200, 1)).is_none());
        // a smaller request is served by the larger buffer
        let scratch = pool.take_scratch(50).ok().flatten().expect("scratch");
        assert_eq!(scratch.id, 1);
        // of two scratches the larger stays
        assert!(pool.put_scratch(fake(64, 2)).is_none());
        assert_eq!(pool.put_scratch(scratch).map(|b| b.id), Some(2));
        assert!(pool.take(200).is_none(), "not an exact-size buffer");

        // the scratch counts against the trim budget first
        pool.put(fake(64, 3));
        assert_eq!(pool.trim(220).iter().map(|b| b.id).collect::<Vec<_>>(), [3]);
        assert_eq!(pool.trim(100).iter().map(|b| b.id).collect::<Vec<_>>(), [1]);
        assert!(matches!(pool.take_scratch(1), Ok(None)));

        // and drains with the rest, for plans and the out-of-memory retry
        assert!(pool.put_scratch(fake(32, 4)).is_none());
        pool.put(fake(16, 5));
        let mut drained: Vec<u64> = pool.drain().iter().map(|b| b.id).collect();
        drained.sort_unstable();
        assert_eq!(drained, [4, 5]);
    }
}
