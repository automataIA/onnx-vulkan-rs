//! Instance, physical device, logical device, and compute queue.

use anyhow::{Context as _, Result, bail};
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::CStr;
use std::sync::Mutex;

/// A 16×16×K `u8 × u8 → 32 bit` cooperative matrix configuration the device
/// advertises. Only this shape family is listed: it is the one the integer
/// matmul kernel is written for. Everything else the driver reports (f16, bf16,
/// fp8, non-square shapes) is ignored here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopMatU8 {
    /// K of one cooperative multiply. The matrix K must be a multiple of it.
    pub k_tile: u32,
    /// The accumulator is `int32` rather than `uint32`. The two are not
    /// interchangeable: the shader's component type must match the driver's.
    pub acc_signed: bool,
}

pub struct VkContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub command_pool: vk::CommandPool,
    /// `VK_KHR_shader_integer_dot_product` enabled on the device.
    pub has_integer_dot_product: bool,
    /// `u8 × u8` cooperative matrix configurations usable on this device, empty
    /// if `VK_KHR_cooperative_matrix` (or one of the features the kernel needs)
    /// is missing.
    pub coop_u8: Vec<CoopMatU8>,
    /// Subgroup width, which is also the workgroup size of the cooperative
    /// matrix shaders (one subgroup computes one 16×16 tile).
    pub subgroup_size: u32,
    pub device_name: String,
    pub vendor_id: u32,
    pub(crate) allocator: Mutex<Option<Allocator>>,
    /// Serializes command pool + queue submit (not thread-safe in Vulkan).
    pub(crate) submit_lock: Mutex<()>,
    /// Deferred command stream (see `stream.rs`).
    pub(crate) stream: Mutex<crate::stream::StreamState>,
    /// Arena of reusable descriptor pools (see `descriptor.rs`).
    pub(crate) descriptors: Mutex<crate::descriptor::DescriptorArena>,
    /// ns per GPU timestamp tick (0 = unsupported).
    pub(crate) timestamp_period: f32,
    /// Alignment required for storage buffer offset in a
    /// descriptor (`minStorageBufferOffsetAlignment`).
    pub storage_offset_alignment: u64,
    /// Profiling query pool (lazy, reused).
    pub(crate) query_pool: Mutex<Option<vk::QueryPool>>,
    /// Host-visible staging buffers reused across flushes (see `buffer.rs`).
    pub(crate) staging_pool: Mutex<crate::buffer::StagingPool>,
    /// Device-local storage buffers freed by their owner and reusable by the
    /// next allocation of the same size (see `buffer.rs`).
    pub(crate) storage_pool: Mutex<crate::buffer::StoragePool>,
}

/// Timestamp slot in profiling query pool.
pub(crate) const TS_CAPACITY: u32 = 16384;

// Vulkan handles can be used from multiple threads; the non-thread-safe
// sections (command pool, queue) are serialized by `submit_lock`, the allocator by a Mutex.
unsafe impl Send for VkContext {}
unsafe impl Sync for VkContext {}

impl VkContext {
    pub fn new() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan loader")?;

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"onnx-vulkan-rs")
            .api_version(vk::API_VERSION_1_2);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance =
            unsafe { entry.create_instance(&create_info, None) }.context("creazione VkInstance")?;

        let (physical_device, queue_family_index) = Self::pick_device(&instance)?;
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        // available device extensions
        let ext_props = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
        let has_ext = |name: &CStr| {
            ext_props.iter().any(|e| {
                let ext = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                ext == name
            })
        };
        let dot_ext = vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME;
        let mut enabled_exts: Vec<*const i8> = Vec::new();
        let has_dot_ext = has_ext(dot_ext);

        let mut dot_features = vk::PhysicalDeviceShaderIntegerDotProductFeaturesKHR::default();
        let mut has_integer_dot_product = false;
        if has_dot_ext {
            let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut dot_features);
            unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
            has_integer_dot_product = dot_features.shader_integer_dot_product == vk::TRUE;
            if has_integer_dot_product {
                enabled_exts.push(dot_ext.as_ptr());
            }
        }

        // Cooperative matrix. Three things have to line up: the extension, the
        // 8-bit storage / int8 / memory-model features the SPIR-V declares, and
        // a (M, N, K, types, scope) combination we have a shader for. The
        // combinations are driver-dependent, so they are queried, never assumed.
        let mut coop_features = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default();
        let mut subgroup_props = vk::PhysicalDeviceSubgroupProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_props);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let subgroup_size = subgroup_props.subgroup_size;

        let mut coop_u8 = Vec::new();
        if has_ext(vk::KHR_COOPERATIVE_MATRIX_NAME) {
            let mut features2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut coop_features)
                .push_next(&mut vk12_features);
            unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
            let usable = coop_features.cooperative_matrix == vk::TRUE
                && vk12_features.vulkan_memory_model == vk::TRUE
                && vk12_features.storage_buffer8_bit_access == vk::TRUE
                && vk12_features.shader_int8 == vk::TRUE;
            if usable {
                let coop = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);
                let combos = unsafe {
                    coop.get_physical_device_cooperative_matrix_properties(physical_device)
                }?;
                for c in combos {
                    let acc_signed = match (c.c_type, c.result_type) {
                        (vk::ComponentTypeKHR::UINT32, vk::ComponentTypeKHR::UINT32) => false,
                        (vk::ComponentTypeKHR::SINT32, vk::ComponentTypeKHR::SINT32) => true,
                        _ => continue,
                    };
                    if c.scope == vk::ScopeKHR::SUBGROUP
                        && c.m_size == 16
                        && c.n_size == 16
                        && c.a_type == vk::ComponentTypeKHR::UINT8
                        && c.b_type == vk::ComponentTypeKHR::UINT8
                        && c.saturating_accumulation == vk::FALSE
                    {
                        coop_u8.push(CoopMatU8 {
                            k_tile: c.k_size,
                            acc_signed,
                        });
                    }
                }
            }
            if !coop_u8.is_empty() {
                enabled_exts.push(vk::KHR_COOPERATIVE_MATRIX_NAME.as_ptr());
                // Only the features the kernel's SPIR-V declares.
                vk12_features = vk::PhysicalDeviceVulkan12Features::default()
                    .vulkan_memory_model(true)
                    .storage_buffer8_bit_access(true)
                    .shader_int8(true);
            }
        }

        let queue_priorities = [1.0f32];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let queue_infos = [queue_info];
        let mut device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_exts);
        if has_integer_dot_product {
            device_info = device_info.push_next(&mut dot_features);
        }
        if !coop_u8.is_empty() {
            device_info = device_info
                .push_next(&mut coop_features)
                .push_next(&mut vk12_features);
        }
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("creazione VkDevice")?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(queue_family_index),
                None,
            )
        }?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .context("creazione gpu-allocator")?;

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let timestamp_valid_bits = queue_families[queue_family_index as usize].timestamp_valid_bits;

        log::info!(
            "Vulkan device: {device_name} (vendor 0x{:04x}), integer_dot_product={has_integer_dot_product}, \
             subgroup={subgroup_size}, coop_u8={coop_u8:?}",
            props.vendor_id
        );

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue,
            queue_family_index,
            command_pool,
            has_integer_dot_product,
            coop_u8,
            subgroup_size,
            device_name,
            vendor_id: props.vendor_id,
            allocator: Mutex::new(Some(allocator)),
            submit_lock: Mutex::new(()),
            stream: Mutex::new(Default::default()),
            descriptors: Mutex::new(Default::default()),
            // timestampValidBits==0 → timestamps not reliable on this queue
            timestamp_period: if timestamp_valid_bits > 0 {
                props.limits.timestamp_period
            } else {
                0.0
            },
            query_pool: Mutex::new(None),
            staging_pool: Mutex::new(Default::default()),
            storage_pool: Mutex::new(Default::default()),
            storage_offset_alignment: props.limits.min_storage_buffer_offset_alignment.max(1),
        })
    }

    /// Enumerates physical devices with compute queues: (name, vendor_id, type).
    pub fn enumerate_devices() -> Result<Vec<(String, u32, vk::PhysicalDeviceType)>> {
        let entry = unsafe { ash::Entry::load() }.context("loading the Vulkan loader")?;
        let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }?;
        let mut out = Vec::new();
        for pd in unsafe { instance.enumerate_physical_devices() }? {
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            if !families
                .iter()
                .any(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            {
                continue;
            }
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            out.push((name, props.vendor_id, props.device_type));
        }
        unsafe { instance.destroy_instance(None) };
        Ok(out)
    }

    fn pick_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let mut best: Option<(vk::PhysicalDevice, u32, i32)> = None;
        for pd in devices {
            let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(qfi) = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            else {
                continue;
            };
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
                _ => 0, // CPU (lavapipe) as last choice
            };
            if best.is_none_or(|(_, _, s)| score > s) {
                best = Some((pd, qfi as u32, score));
            }
        }
        match best {
            Some((pd, qfi, _)) => Ok((pd, qfi)),
            None => bail!("no Vulkan device with a compute queue"),
        }
    }

    /// One-shot command buffer: records, submits, waits for completion.
    pub(crate) fn run_commands(&self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let device = &self.device;
        let _guard = self.submit_lock.lock().unwrap();
        unsafe {
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
            record(cmd);
            device.end_command_buffer(cmd)?;

            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let result = device
                .queue_submit(self.queue, &[submit], fence)
                .and_then(|()| device.wait_for_fences(&[fence], true, u64::MAX));
            device.destroy_fence(fence, None);
            device.free_command_buffers(self.command_pool, &cmds);
            result?;
        }
        Ok(())
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(pool) = self.query_pool.lock().unwrap().take() {
                self.device.destroy_query_pool(pool, None);
            }
            self.destroy_descriptors();
            self.destroy_staging_pool();
            self.destroy_storage_pool();
            // the allocator must be destroyed before the device
            drop(self.allocator.lock().unwrap().take());
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
