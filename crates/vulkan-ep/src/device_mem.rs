//! Device memory for ORT: `OrtAllocator` allocating `GpuBuffer` and returning
//! opaque handles (`BufferEntry` pointers). ORT never dereferences these
//! pointers: it passes them to our kernels and our DataTransfer.

use crate::vk;
use anyhow::{Result, bail};
use ort_ep_sys as sys;
use std::collections::BTreeMap;
use std::ffi::c_void;
use vk_compute::GpuBuffer;

const MAGIC: u64 = 0x564b_4246_5f45_4e54; // "VKBF_ENT"

/// Handle for tensor resident in VRAM.
#[repr(C)]
pub struct BufferEntry {
    magic: u64,
    /// Logical size requested by ORT (bytes).
    pub size: usize,
    pub buf: GpuBuffer,
}

/// Reserves `size` bytes of address space and returns base pointer.
///
/// A `Box` is not enough: ORT treats the handle as the base of a region
/// `size` bytes wide and does arithmetic on top of it (see [`LIVE`]). If the
/// handle were a normal heap allocation, other allocations would fall **inside**
/// the range ORT considers ours, and an internal pointer would no longer be
/// uniquely resolvable. By actually reserving the region, intervals stay
/// disjoint. Pages beyond the first are never touched, so the physical memory
/// cost is one page per allocation.
fn reserve(size: usize) -> Option<*mut u8> {
    #[cfg(unix)]
    {
        const PROT_READ_WRITE: i32 = 0x1 | 0x2;
        const MAP_PRIVATE_ANON: i32 = 0x02 | 0x20;
        unsafe extern "C" {
            fn mmap(
                addr: *mut c_void,
                len: usize,
                prot: i32,
                flags: i32,
                fd: i32,
                offset: i64,
            ) -> *mut c_void;
        }
        let p = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ_WRITE,
                MAP_PRIVATE_ANON,
                -1,
                0,
            )
        };
        (p as isize != -1).then(|| p.cast::<u8>())
    }
    #[cfg(windows)]
    {
        const MEM_RESERVE_COMMIT: u32 = 0x2000 | 0x1000;
        const PAGE_READWRITE: u32 = 0x04;
        unsafe extern "system" {
            fn VirtualAlloc(
                addr: *mut c_void,
                size: usize,
                allocation_type: u32,
                protect: u32,
            ) -> *mut c_void;
        }
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null_mut(),
                size,
                MEM_RESERVE_COMMIT,
                PAGE_READWRITE,
            )
        };
        (!p.is_null()).then(|| p.cast::<u8>())
    }
}

/// Returns to the system a region obtained from [`reserve`].
fn unreserve(p: *mut u8, size: usize) {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn munmap(addr: *mut c_void, len: usize) -> i32;
        }
        unsafe { munmap(p.cast(), size) };
    }
    #[cfg(windows)]
    {
        const MEM_RELEASE: u32 = 0x8000;
        unsafe extern "system" {
            fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        }
        let _ = size;
        unsafe { VirtualFree(p.cast(), 0, MEM_RELEASE) };
    }
}

/// Live allocations, indexed by handle address.
///
/// ORT does not treat our pointers as opaque as the documentation suggests:
/// with the **memory pattern** active it allocates a single block and assigns
/// each tensor `handle + offset`. An internal pointer is not a `BufferEntry`,
/// so it must be resolved by walking back to the allocation that contains it.
/// Thanks to [`reserve`] intervals are disjoint, so the walk-back is unique.
static LIVE: std::sync::Mutex<Option<BTreeMap<usize, usize>>> = std::sync::Mutex::new(None);

fn live() -> std::sync::MutexGuard<'static, Option<BTreeMap<usize, usize>>> {
    let mut guard = LIVE.lock().expect("registro allocazioni avvelenato");
    guard.get_or_insert_with(BTreeMap::new);
    guard
}

/// A device tensor: the allocation that contains it and its offset within it
/// (0 in the normal case, non-zero with ORT's memory pattern).
pub struct DeviceRegion<'a> {
    pub entry: &'a BufferEntry,
    pub offset: u64,
}

impl DeviceRegion<'_> {
    pub fn buffer(&self) -> &GpuBuffer {
        &self.entry.buf
    }

    /// The tensor coincides with the whole allocation: it can be bound
    /// directly to a descriptor, with no copies.
    pub fn is_whole(&self) -> bool {
        self.offset == 0
    }

    /// Binding for a dispatch: the descriptor starts from the tensor's offset,
    /// so the shader indexes from 0 as if the tensor were standalone.
    pub fn slice(&self) -> vk_compute::BufferSlice<'_> {
        vk_compute::BufferSlice {
            buf: &self.entry.buf,
            offset: self.offset,
        }
    }
}

/// Resolves an ORT data pointer to the allocation that contains it.
///
/// # Safety
/// `p` must come from [`alloc_entry`], optionally shifted by an offset
/// internal to the allocation (fail loud otherwise).
pub unsafe fn region_from_ptr<'a>(p: *const c_void) -> Result<DeviceRegion<'a>> {
    if p.is_null() {
        bail!("null device pointer");
    }
    let addr = p as usize;
    // The registry lookup happens **before** any dereference: this function
    // also receives pointers to host memory (used to tell the two cases apart)
    // and already-freed addresses, and freed regions are returned to the
    // system, so reading them would access unmapped memory. Intervals are
    // disjoint: at most one contains `addr`.
    let guard = live();
    let map = guard.as_ref().expect("registro inizializzato");
    if let Some((&start, &span)) = map.range(..=addr).next_back()
        && addr < start + span
    {
        let entry = unsafe { &*(start as *const BufferEntry) };
        if entry.magic == MAGIC {
            return Ok(DeviceRegion {
                entry,
                offset: (addr - start) as u64,
            });
        }
    }
    bail!("pointer {addr:#x} does not belong to any VulkanEP allocation")
}

/// Allocates a device buffer and returns the opaque handle for ORT.
pub fn alloc_entry(size: usize) -> Result<*mut c_void> {
    let ctx = vk::context()?;
    // round up to 4: shaders write whole u32 words
    let vk_size = size.next_multiple_of(4).max(4) as u64;
    let buf = ctx.create_storage_buffer(vk_size)?;
    // the region must hold the header and cover everything ORT considers
    // its own, so internal pointers stay uniquely resolvable
    let span = size.max(size_of::<BufferEntry>()).next_multiple_of(4096);
    let Some(base) = reserve(span) else {
        ctx.defer_destroy(buf);
        bail!("riserva di {span} byte di spazio di indirizzi fallita");
    };
    let ptr = base.cast::<BufferEntry>();
    unsafe {
        ptr.write(BufferEntry {
            magic: MAGIC,
            size,
            buf,
        })
    };
    live()
        .as_mut()
        .expect("registro inizializzato")
        .insert(ptr as usize, span);
    Ok(ptr.cast::<c_void>())
}

/// Frees a handle created by [`alloc_entry`].
///
/// # Safety
/// `p` must come from [`alloc_entry`] and must not have already been freed.
pub unsafe fn free_entry(p: *mut c_void) {
    if p.is_null() {
        return;
    }
    let ptr = p.cast::<BufferEntry>();
    if unsafe { (*ptr).magic } != MAGIC {
        log::error!("free_entry: magic errato, leak intenzionale per sicurezza");
        return;
    }
    let span = live()
        .as_mut()
        .expect("registro inizializzato")
        .remove(&(p as usize));
    let entry = unsafe { ptr.read() };
    if let Ok(ctx) = vk::context() {
        ctx.defer_destroy(entry.buf);
    }
    if let Some(span) = span {
        unreserve(p.cast::<u8>(), span);
    }
}

/// `OrtAllocator` for the VulkanEP device memory.
#[repr(C)]
pub struct VulkanOrtAllocator {
    base: sys::OrtAllocator,
    memory_info: *const sys::OrtMemoryInfo,
}

impl VulkanOrtAllocator {
    pub fn new(memory_info: *const sys::OrtMemoryInfo) -> Self {
        let mut base: sys::OrtAllocator = unsafe { std::mem::zeroed() };
        base.version = sys::ORT_API_VERSION;
        base.Alloc = Some(alloc_impl);
        base.Free = Some(free_impl);
        base.Info = Some(info_impl);
        base.Reserve = Some(alloc_impl);
        Self { base, memory_info }
    }
}

unsafe extern "C" fn alloc_impl(_this: *mut sys::OrtAllocator, size: usize) -> *mut c_void {
    match alloc_entry(size) {
        Ok(p) => p,
        Err(e) => {
            log::error!("VulkanEP Alloc({size}) fallita: {e:#}");
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn free_impl(_this: *mut sys::OrtAllocator, p: *mut c_void) {
    unsafe { free_entry(p) };
}

unsafe extern "C" fn info_impl(this: *const sys::OrtAllocator) -> *const sys::OrtMemoryInfo {
    let allocator = unsafe { &*this.cast::<VulkanOrtAllocator>() };
    allocator.memory_info
}
