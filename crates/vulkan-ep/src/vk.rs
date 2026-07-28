//! Vulkan context shared by the plugin (one per process, lazy).

use std::sync::OnceLock;
use vk_compute::VkContext;

static CTX: OnceLock<Option<VkContext>> = OnceLock::new();

pub fn context() -> anyhow::Result<&'static VkContext> {
    CTX.get_or_init(|| match VkContext::new() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            log::error!("VulkanEP: inizializzazione Vulkan fallita: {e:#}");
            None
        }
    })
    .as_ref()
    .ok_or_else(|| anyhow::anyhow!("contesto Vulkan non disponibile"))
}
