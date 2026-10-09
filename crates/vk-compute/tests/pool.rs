//! The storage pool under memory pressure, on the device the context picks.

use ash::vk;
use vk_compute::VkContext;

/// Buffers kept pooled for a later run must not be what makes the device run
/// out: an allocation that fails for lack of memory releases them before it
/// gives up.
#[test]
fn an_allocation_out_of_memory_releases_the_pool() {
    let context = VkContext::new().expect("Vulkan context");
    let memory = unsafe {
        context
            .instance
            .get_physical_device_memory_properties(context.physical_device)
    };
    let heap = memory
        .memory_heaps_as_slice()
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .expect("a device-local heap");
    let mut limits = vk::PhysicalDeviceMaintenance4Properties::default();
    let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut limits);
    unsafe {
        context
            .instance
            .get_physical_device_properties2(context.physical_device, &mut properties)
    };
    let too_big = heap + (1 << 20);
    if too_big > limits.max_buffer_size {
        // asking for it would be invalid usage, not an out-of-memory error
        eprintln!("skipped: maxBufferSize below the device-local heap");
        return;
    }

    let kept = context.create_storage_buffer(1 << 20).expect("1 MB");
    context.recycle_storage_buffer(kept);
    assert!(context.create_storage_buffer(too_big).is_err());
    assert!(
        context.take_storage_pool().is_empty(),
        "the pooled buffer was released for the retry"
    );
    let after = context
        .create_storage_buffer(1 << 20)
        .expect("the device still allocates after the failure");
    context.recycle_storage_buffer(after);
}
