//! Tensori residenti sul backend Vulkan.

use crate::HostTensor;
use vk_compute::GpuBuffer;

/// Vulkan storage owned by runtime or borrowed from host.
pub enum DeviceBuffer<'a> {
    Owned(GpuBuffer),
    Borrowed(&'a GpuBuffer),
}

impl DeviceBuffer<'_> {
    pub fn get(&self) -> &GpuBuffer {
        match self {
            Self::Owned(buffer) => buffer,
            Self::Borrowed(buffer) => buffer,
        }
    }
}

/// Tensore residente in VRAM.
pub struct DeviceTensor<'a> {
    pub dtype: i32,
    pub shape: Vec<i64>,
    pub elem_count: usize,
    pub buf: DeviceBuffer<'a>,
}

impl DeviceTensor<'_> {
    pub fn buffer(&self) -> &GpuBuffer {
        self.buf.get()
    }
}

/// Execution value, resident on host or device.
pub enum Tensor<'a> {
    Device(DeviceTensor<'a>),
    Host(HostTensor),
}
