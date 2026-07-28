//! Arena of descriptor pools reused across dispatches.
//!
//! Per-dispatch pools (create+free each time) are the primary anti-pattern
//! for workloads with thousands of dispatches. Here pools are persistent:
//! sets are allocated until depleted, then moves to next pool;
//! on `flush()` all pools are **reset** (not destroyed) and reused.

use crate::context::VkContext;
use anyhow::{Context as _, Result};
use ash::vk;

/// Sets per pool and STORAGE_BUFFER descriptors per set (max kernel bindings).
const SETS_PER_POOL: u32 = 1024;
const MAX_BINDINGS_PER_SET: u32 = 8;

#[derive(Default)]
pub(crate) struct DescriptorArena {
    pools: Vec<vk::DescriptorPool>,
    /// Index of pool currently in use.
    current: usize,
    /// Sets already allocated from the current pool.
    used_in_current: u32,
}

impl VkContext {
    fn new_descriptor_pool(&self) -> Result<vk::DescriptorPool> {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(SETS_PER_POOL * MAX_BINDINGS_PER_SET)];
        let pool = unsafe {
            self.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(SETS_PER_POOL)
                    .pool_sizes(&pool_sizes),
                None,
            )?
        };
        Ok(pool)
    }

    /// Allocates descriptor set for `layout` from reusable pool.
    pub(crate) fn acquire_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let mut arena = self.descriptors.lock().unwrap();
        if arena.pools.is_empty() {
            let pool = self.new_descriptor_pool()?;
            arena.pools.push(pool);
        }
        if arena.used_in_current >= SETS_PER_POOL {
            arena.current += 1;
            arena.used_in_current = 0;
            if arena.current >= arena.pools.len() {
                let pool = self.new_descriptor_pool()?;
                arena.pools.push(pool);
            }
        }
        let pool = arena.pools[arena.current];
        let layouts = [layout];
        let set = unsafe {
            self.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .context("allocazione descriptor set dall'arena")?[0];
        arena.used_in_current += 1;
        Ok(set)
    }

    /// Resets all pools (sets become free) without destroying them.
    /// Safe only at GPU idle — called after flush fence.
    pub(crate) fn reset_descriptors(&self) {
        let mut arena = self.descriptors.lock().unwrap();
        for &pool in &arena.pools {
            unsafe {
                let _ = self
                    .device
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty());
            }
        }
        arena.current = 0;
        arena.used_in_current = 0;
    }

    pub(crate) fn destroy_descriptors(&self) {
        let mut arena = self.descriptors.lock().unwrap();
        for pool in arena.pools.drain(..) {
            unsafe { self.device.destroy_descriptor_pool(pool, None) };
        }
    }
}
