//! Probe `VK_KHR_cooperative_matrix` on every Vulkan device of the machine.
//!
//! The (M, N, K, types, scope) combinations are driver-dependent and must never
//! be assumed: this dumps exactly what `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`
//! reports, plus the pieces needed to decide whether a naga-generated shader can
//! use them (subgroup size, supported stages, device API version).
//!
//! Run: `cargo run -p vk-compute --example coopmat_probe`

use ash::vk;
use std::ffi::CStr;

fn comp_type(t: vk::ComponentTypeKHR) -> &'static str {
    match t {
        vk::ComponentTypeKHR::FLOAT16 => "f16",
        vk::ComponentTypeKHR::FLOAT32 => "f32",
        vk::ComponentTypeKHR::FLOAT64 => "f64",
        vk::ComponentTypeKHR::SINT8 => "i8",
        vk::ComponentTypeKHR::SINT16 => "i16",
        vk::ComponentTypeKHR::SINT32 => "i32",
        vk::ComponentTypeKHR::SINT64 => "i64",
        vk::ComponentTypeKHR::UINT8 => "u8",
        vk::ComponentTypeKHR::UINT16 => "u16",
        vk::ComponentTypeKHR::UINT32 => "u32",
        vk::ComponentTypeKHR::UINT64 => "u64",
        other => Box::leak(format!("{other:?}").into_boxed_str()),
    }
}

fn scope(s: vk::ScopeKHR) -> &'static str {
    match s {
        vk::ScopeKHR::DEVICE => "Device",
        vk::ScopeKHR::WORKGROUP => "Workgroup",
        vk::ScopeKHR::SUBGROUP => "Subgroup",
        vk::ScopeKHR::QUEUE_FAMILY => "QueueFamily",
        other => Box::leak(format!("{other:?}").into_boxed_str()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let entry = unsafe { ash::Entry::load() }?;
    let app_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let exts = [vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr()];
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_extension_names(&exts),
            None,
        )
    }?;
    let coop = ash::khr::cooperative_matrix::Instance::new(&entry, &instance);

    for pd in unsafe { instance.enumerate_physical_devices() }? {
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut coop_props = vk::PhysicalDeviceCooperativeMatrixPropertiesKHR::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup)
            .push_next(&mut coop_props);
        unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
        let p = props2.properties;
        let name = unsafe { CStr::from_ptr(p.device_name.as_ptr()) }.to_string_lossy();
        let (maj, min, patch) = (
            vk::api_version_major(p.api_version),
            vk::api_version_minor(p.api_version),
            vk::api_version_patch(p.api_version),
        );
        println!("=== {name} (vendor 0x{:x}) ===", p.vendor_id);
        println!("  apiVersion            {maj}.{min}.{patch}");
        println!("  subgroupSize          {}", subgroup.subgroup_size);

        let ext_props = unsafe { instance.enumerate_device_extension_properties(pd) }?;
        let has = |n: &CStr| {
            ext_props
                .iter()
                .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == n)
        };
        let has_coop = has(vk::KHR_COOPERATIVE_MATRIX_NAME);
        println!("  VK_KHR_cooperative_matrix       {has_coop}");
        println!(
            "  VK_KHR_shader_integer_dot_product {}",
            has(vk::KHR_SHADER_INTEGER_DOT_PRODUCT_NAME)
        );
        println!(
            "  VK_KHR_vulkan_memory_model        {}",
            has(vk::KHR_VULKAN_MEMORY_MODEL_NAME)
        );
        println!("  VK_KHR_16bit_storage              {}", has(vk::KHR_16BIT_STORAGE_NAME));
        println!(
            "  VK_KHR_shader_float16_int8        {}",
            has(vk::KHR_SHADER_FLOAT16_INT8_NAME)
        );
        println!("  VK_KHR_8bit_storage               {}", has(vk::KHR_8BIT_STORAGE_NAME));
        if !has_coop {
            println!("  -> no cooperative matrix on this device\n");
            continue;
        }

        let mut feat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
        let mut memmodel = vk::PhysicalDeviceVulkanMemoryModelFeatures::default();
        let mut feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut feat)
            .push_next(&mut f16i8)
            .push_next(&mut memmodel);
        unsafe { instance.get_physical_device_features2(pd, &mut feats2) };
        println!("  cooperativeMatrix     {}", feat.cooperative_matrix == vk::TRUE);
        println!(
            "  ...RobustBufferAccess {}",
            feat.cooperative_matrix_robust_buffer_access == vk::TRUE
        );
        println!(
            "  supportedStages       {:?}",
            coop_props.cooperative_matrix_supported_stages
        );
        println!("  shaderFloat16         {}", f16i8.shader_float16 == vk::TRUE);
        println!("  shaderInt8            {}", f16i8.shader_int8 == vk::TRUE);
        println!(
            "  vulkanMemoryModel     {}",
            memmodel.vulkan_memory_model == vk::TRUE
        );

        let combos = unsafe { coop.get_physical_device_cooperative_matrix_properties(pd) }?;
        println!("  {} combinations:", combos.len());
        println!("    {:>3} {:>3} {:>3}  {:>4} {:>4} {:>4} {:>4}  {:<9} sat", "M", "N", "K", "A", "B", "C", "R", "scope");
        for c in &combos {
            println!(
                "    {:>3} {:>3} {:>3}  {:>4} {:>4} {:>4} {:>4}  {:<9} {}",
                c.m_size,
                c.n_size,
                c.k_size,
                comp_type(c.a_type),
                comp_type(c.b_type),
                comp_type(c.c_type),
                comp_type(c.result_type),
                scope(c.scope),
                c.saturating_accumulation == vk::TRUE,
            );
        }
        println!();
    }

    unsafe { instance.destroy_instance(None) };
    Ok(())
}
