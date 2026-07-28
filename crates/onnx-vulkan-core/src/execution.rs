//! Shared execution environment between standalone frontend and host adapters.

use crate::KernelCache;
use crate::host_ops::{FLOAT, HostTensor};
use crate::{
    DeviceBuffer, DeviceTensor, Error, InitializerIr, Result, Tensor, elem_size, storage_len,
};
use std::cell::RefCell;
use std::collections::HashMap;
use vk_compute::VkContext;

const HOST_MEMO_MAX_BYTES: usize = 4096;

/// Name→tensor table with management of host/device residency.
pub struct ExecutionEnv<'context, 'values> {
    context: &'context VkContext,
    cache: &'context KernelCache<'context>,
    values: HashMap<String, Tensor<'values>>,
    initializers: &'values HashMap<String, InitializerIr>,
    host_cache: RefCell<HashMap<String, HostTensor>>,
}

impl<'context, 'values> ExecutionEnv<'context, 'values> {
    /// Cache must belong to the same `VkContext`: pipelines and buffers are not
    /// transferable across devices.
    pub fn new(
        cache: &'context KernelCache<'context>,
        initializers: &'values HashMap<String, InitializerIr>,
    ) -> Self {
        Self {
            context: cache.context(),
            cache,
            values: HashMap::new(),
            initializers,
            host_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn context(&self) -> &'context VkContext {
        self.context
    }

    /// Resources reused across executions (pipelines, packed weights).
    pub fn cache(&self) -> &'context KernelCache<'context> {
        self.cache
    }

    pub fn value(&self, name: &str) -> Option<&Tensor<'values>> {
        self.values.get(name)
    }

    /// Value is a graph constant, so anything derived from it can be cached
    /// across runs under its name.
    pub fn is_initializer(&self, name: &str) -> bool {
        self.initializers.contains_key(name)
    }

    pub fn on_device(&self, name: &str) -> bool {
        matches!(self.values.get(name), Some(Tensor::Device(_)))
    }

    /// Value is available on host **without** reading from device, and is
    /// small enough to justify CPU computation.
    ///
    /// Used to keep shape-math (anchors, strides, indices) off the GPU:
    /// computing on device causes a download for every value a
    /// `Reshape` or `Slice` reads, where each download is a submit+fence.
    pub fn host_resident_small(&self, name: &str, max_bytes: usize) -> bool {
        let byte_len = match self.values.get(name) {
            Some(Tensor::Host(tensor)) => tensor.data.len(),
            Some(Tensor::Device(_)) => return false,
            None => match self.initializers.get(name) {
                Some(initializer) => initializer.data.len(),
                None => return false,
            },
        };
        byte_len <= max_bytes
    }

    /// Threshold above which a host tensor is still computed on GPU.
    pub const SMALL_HOST_BYTES: usize = HOST_MEMO_MAX_BYTES;

    pub fn device(&self, name: &str) -> Result<&DeviceTensor<'_>> {
        match self.values.get(name) {
            Some(Tensor::Device(tensor)) => Ok(tensor),
            _ => Err(Error::InvalidTensor(format!(
                "device value '{name}' not present"
            ))),
        }
    }

    /// Returns a host copy, downloading and memoizing small tensors
    /// used by shape-math.
    pub fn host(&self, name: &str) -> Result<HostTensor> {
        // an initializer is immutable: the host copy already exists, downloading
        // the device copy would be a submit+fence for data we already have in RAM
        if let Some(initializer) = self.initializers.get(name) {
            return Ok(HostTensor::new(
                initializer.dtype,
                initializer.shape.clone(),
                initializer.data.clone(),
            ));
        }
        match self.values.get(name) {
            Some(Tensor::Host(tensor)) => Ok(tensor.clone()),
            Some(Tensor::Device(tensor)) => {
                if let Some(host) = self.host_cache.borrow().get(name) {
                    return Ok(host.clone());
                }
                let byte_len = storage_len(tensor.dtype, tensor.elem_count).ok_or_else(|| {
                    Error::InvalidTensor(format!(
                        "dtype {} has no fixed-size storage",
                        tensor.dtype
                    ))
                })?;
                let data = if byte_len == 0 {
                    Vec::new()
                } else {
                    self.context
                        .stream_download(tensor.buffer(), byte_len)
                        .map_err(backend_error)?
                };
                let host = HostTensor::new(tensor.dtype, tensor.shape.clone(), data);
                if byte_len <= HOST_MEMO_MAX_BYTES {
                    self.host_cache
                        .borrow_mut()
                        .insert(name.to_owned(), host.clone());
                }
                Ok(host)
            }
            None => {
                let initializer = self.initializers.get(name).ok_or_else(|| {
                    Error::InvalidTensor(format!("host value '{name}' not present"))
                })?;
                Ok(HostTensor::new(
                    initializer.dtype,
                    initializer.shape.clone(),
                    initializer.data.clone(),
                ))
            }
        }
    }

    pub fn dtype_of(&self, name: &str) -> Result<i32> {
        match self.values.get(name) {
            Some(Tensor::Device(tensor)) => Ok(tensor.dtype),
            Some(Tensor::Host(tensor)) => Ok(tensor.dtype),
            None => self
                .initializers
                .get(name)
                .map(|initializer| initializer.dtype)
                .ok_or_else(|| Error::InvalidTensor(format!("dtype of '{name}' not available"))),
        }
    }

    pub fn shape_of(&self, name: &str) -> Result<Vec<i64>> {
        match self.values.get(name) {
            Some(Tensor::Device(tensor)) => Ok(tensor.shape.clone()),
            Some(Tensor::Host(tensor)) => Ok(tensor.shape.clone()),
            None => self
                .initializers
                .get(name)
                .map(|initializer| initializer.shape.clone())
                .ok_or_else(|| Error::InvalidTensor(format!("shape of '{name}' not available"))),
        }
    }

    /// Uploads to VRAM keeping the original ONNX dtype.
    pub fn ensure_device_dtype(&mut self, name: &str) -> Result<()> {
        if self.on_device(name) {
            return Ok(());
        }
        // before `host()`: on a warm cache that call would clone the weight for
        // nothing
        if let Some(tensor) = self.cached_initializer(name)? {
            self.set(name, Tensor::Device(tensor));
            return Ok(());
        }
        let host = self.host(name)?;
        host.validate()?;
        let buffer = self
            .context
            .create_storage_buffer(device_storage_bytes(host.dtype, host.elem_count())?)
            .map_err(backend_error)?;
        if !host.data.is_empty() {
            self.context
                .stream_upload(&buffer, &host.data)
                .map_err(backend_error)?;
        }
        self.set(
            name,
            Tensor::Device(DeviceTensor {
                dtype: host.dtype,
                shape: host.shape.clone(),
                elem_count: host.elem_count(),
                buf: DeviceBuffer::Owned(buffer),
            }),
        );
        Ok(())
    }

    /// If `name` is an initializer — a constant across runs — returns a tensor
    /// pointing at the copy already in VRAM, uploading it on first request.
    ///
    /// This is what makes weights **resident**: without it, every run
    /// reallocates and re-uploads every constant in the graph (measured:
    /// 10.4 MB per run on YOLOv8n, 25 MB on MobileNetV2).
    /// The key and metadata are read **from the initializer itself**, never from
    /// a materialized `HostTensor`. That distinction is the whole point:
    /// `host()` clones an initializer's bytes, so asking for the host copy in
    /// order to discover that the device copy is already cached costs a full
    /// copy of the weight on *every* run. Measured on resnet50-qdq once constant
    /// folding turned 25.5 MB of int8 weights into 102.1 MB of fp32: ~13 ms per
    /// run of pure host memcpy, more than the 8 ms the whole graph spends on the
    /// GPU. The bytes are touched only on a cache miss, inside the closure.
    fn cached_initializer(&self, name: &str) -> Result<Option<DeviceTensor<'values>>> {
        // only values that come from initializers: anything else changes per
        // run and must never be memoized by name
        let Some(initializer) = self.initializers.get(name) else {
            return Ok(None);
        };
        if self.values.contains_key(name) {
            return Ok(None);
        }
        let elem_count = crate::element_count(&initializer.shape)?;
        let bytes = device_storage_bytes(initializer.dtype, elem_count)?;
        let context = self.context;
        let data = &initializer.data;
        let buffer = self
            .cache
            .initializer((name.to_owned(), initializer.dtype, data.len()), || {
                let buffer = context.create_storage_buffer(bytes)?;
                if !data.is_empty() {
                    context.stream_upload(&buffer, data)?;
                }
                Ok(buffer)
            })
            .map_err(backend_error)?;
        // SAFETY: the cache lives as long as the session and never removes an
        // entry, so the buffer stays valid for the whole execution (see `cache.rs`).
        Ok(Some(DeviceTensor {
            dtype: initializer.dtype,
            shape: initializer.shape.clone(),
            elem_count,
            buf: DeviceBuffer::Borrowed(unsafe { &*buffer }),
        }))
    }

    /// Uploads an f32 tensor to VRAM.
    pub fn ensure_device(&mut self, name: &str) -> Result<()> {
        if self.on_device(name) {
            return Ok(());
        }
        // the dtype is read off the initializer's metadata rather than a host
        // copy, so a warm cache never clones the weight (see `cached_initializer`)
        if let Some(initializer) = self.initializers.get(name) {
            if initializer.dtype != FLOAT {
                return Err(Error::InvalidTensor(format!(
                    "ensure_device '{name}': dtype {}, expected f32",
                    initializer.dtype
                )));
            }
            if let Some(tensor) = self.cached_initializer(name)? {
                self.set(name, Tensor::Device(tensor));
                return Ok(());
            }
        }
        let host = self.host(name)?;
        host.validate()?;
        if host.dtype != FLOAT {
            return Err(Error::InvalidTensor(format!(
                "ensure_device '{name}': dtype {}, expected f32",
                host.dtype
            )));
        }
        let tensor = upload_float(self.context, &host.shape, &host.data)?;
        self.set(name, Tensor::Device(tensor));
        Ok(())
    }

    pub fn set(&mut self, name: &str, tensor: Tensor<'values>) {
        self.values.insert(name.to_owned(), tensor);
        self.host_cache.borrow_mut().remove(name);
    }

    /// Drops a value the graph will not read again, returning its VRAM to the
    /// pool the next allocation draws from.
    ///
    /// This is what bounds a block's peak memory to its **live set** instead of
    /// the sum of every intermediate it produces. Without it a 4066-node block
    /// keeps all 4066 tensors alive to the last node, because `finish` is the
    /// only point that frees anything.
    ///
    /// Recycling rather than destroying is the part that matters: `defer_destroy`
    /// only queues the free until the next flush, and inside a block there is no
    /// flush — the memory would come back after the run that needed it.
    pub fn release(&mut self, name: &str) {
        self.host_cache.borrow_mut().remove(name);
        if let Some(Tensor::Device(DeviceTensor {
            buf: DeviceBuffer::Owned(buffer),
            ..
        })) = self.values.remove(name)
        {
            self.context.recycle_storage_buffer(buffer);
        }
    }

    /// Releases owned intermediate buffers; borrowed buffers stay with the
    /// host that provided them.
    pub fn finish(mut self) {
        self.release_owned();
    }

    fn release_owned(&mut self) {
        for (_, tensor) in self.values.drain() {
            if let Tensor::Device(DeviceTensor {
                buf: DeviceBuffer::Owned(buffer),
                ..
            }) = tensor
            {
                self.context.defer_destroy(buffer);
            }
        }
        // buffers recycled by `release` belong to this execution: keeping them
        // pooled would hold the whole live set of the block until the context dies
        self.context.drain_storage_pool();
    }
}

impl Drop for ExecutionEnv<'_, '_> {
    fn drop(&mut self) {
        self.release_owned();
    }
}

/// Device bytes, aligned to a `u32` word with a 4-byte minimum allocation.
pub fn device_storage_bytes(dtype: i32, element_count: usize) -> Result<u64> {
    let byte_len = storage_len(dtype, element_count)
        .ok_or_else(|| Error::InvalidTensor(format!("dtype {dtype} has no fixed-size storage")))?;
    let aligned = byte_len
        .checked_add(3)
        .ok_or_else(|| Error::InvalidTensor("buffer size overflow".into()))?
        / 4
        * 4;
    u64::try_from(aligned.max(4))
        .map_err(|_| Error::InvalidTensor("buffer size not representable".into()))
}

fn upload_float(context: &VkContext, shape: &[i64], bytes: &[u8]) -> Result<DeviceTensor<'static>> {
    let element_count = bytes.len() / elem_size(FLOAT);
    let buffer = context
        .create_storage_buffer(device_storage_bytes(FLOAT, element_count)?)
        .map_err(backend_error)?;
    if !bytes.is_empty() {
        context
            .stream_upload(&buffer, bytes)
            .map_err(backend_error)?;
    }
    Ok(DeviceTensor {
        dtype: FLOAT,
        shape: shape.to_vec(),
        elem_count: element_count,
        buf: DeviceBuffer::Owned(buffer),
    })
}

fn backend_error(error: impl std::fmt::Display) -> Error {
    Error::Backend(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;

    #[test]
    fn device_storage_is_word_aligned_and_supports_packed_types() {
        assert_eq!(
            device_storage_bytes(ElementType::Float32 as i32, 3).expect("f32"),
            12
        );
        assert_eq!(
            device_storage_bytes(ElementType::Int4 as i32, 3).expect("int4 packed"),
            4
        );
        assert_eq!(
            device_storage_bytes(ElementType::Uint8 as i32, 0).expect("empty tensor"),
            4
        );
    }

    #[test]
    fn device_storage_rejects_unknown_types() {
        assert!(matches!(
            device_storage_bytes(999, 1),
            Err(Error::InvalidTensor(_))
        ));
    }
}
