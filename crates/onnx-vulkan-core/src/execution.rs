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
    bound_outputs: HashMap<String, (&'values vk_compute::GpuBuffer, usize)>,
    /// Values whose declared shape is the logical tensor while the buffer holds
    /// it with a longer row stride — see `mark_row_padded`.
    row_padded: HashMap<String, usize>,
    /// Shape and dtype of values whose memory has gone back to the pool.
    ///
    /// `Shape` asks for the dimensions of a tensor, not for its bytes, and a
    /// graph is free to ask after the last node that read the data — qwen2.5-VL
    /// does exactly that in its mRoPE path. Freeing the buffer at the last
    /// *reader* is what bounds a step's memory, so the answer is not to keep
    /// the memory but to stop conflating it with what is known about it.
    released: HashMap<String, (i32, Vec<i64>)>,
    /// Hand this run's buffers back to the pool instead of freeing them.
    retain: bool,
    /// The node being executed is the last reader of its first input, see
    /// `set_first_input_dies`.
    first_input_dies: bool,
    /// Zeroed slots for `DynamicQuantizeLinear` ranges, see `range_slot`.
    ranges: Vec<vk_compute::GpuBuffer>,
    /// Bytes of the last of `ranges` already handed out.
    range_used: u64,
    /// Values whose range a producer folded into a slot while writing them,
    /// see `set_range`.
    computed_ranges: HashMap<String, (usize, u64)>,
}

/// Slots per buffer of `ExecutionEnv::ranges`: SAM 3's detector quantizes
/// 181 tensors per run.
const RANGE_SLOTS: u64 = 256;

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
            bound_outputs: HashMap::new(),
            row_padded: HashMap::new(),
            released: HashMap::new(),
            retain: false,
            first_input_dies: false,
            ranges: Vec::new(),
            range_used: 0,
            computed_ranges: HashMap::new(),
        }
    }

    /// A fresh 8-byte slot of zeros, as the buffer index in `ranges` and the
    /// offset in it, for the range `DynamicQuantizeLinear` accumulates with
    /// atomics. The slots come from buffers of [`RANGE_SLOTS`], each cleared
    /// by one fill when it is created: a fill per quantization costs about
    /// what the reduction pass it replaced did.
    pub fn range_slot(&mut self) -> Result<(usize, u64)> {
        let stride = self.context.storage_offset_alignment.max(8);
        if self.ranges.is_empty() || self.range_used + stride > RANGE_SLOTS * stride {
            let bytes = RANGE_SLOTS * stride;
            let buffer = self
                .context
                .create_storage_buffer(bytes)
                .map_err(backend_error)?;
            self.context
                .stream_fill(&buffer, 0, bytes, 0)
                .map_err(backend_error)?;
            self.ranges.push(buffer);
            self.range_used = 0;
        }
        let offset = self.range_used;
        self.range_used += stride;
        Ok((self.ranges.len() - 1, offset))
    }

    /// The buffer of a slot `range_slot` handed out.
    pub fn range_buffer(&self, index: usize) -> &vk_compute::GpuBuffer {
        &self.ranges[index]
    }

    /// Records that the kernel producing `name` also folded its range into
    /// `slot`, from `range_slot`, so a `DynamicQuantizeLinear` of `name` can
    /// skip its own min/max pass. Valid until the end of the run.
    pub fn set_range(&mut self, name: &str, slot: (usize, u64)) {
        self.computed_ranges.insert(name.to_owned(), slot);
    }

    /// The slot `set_range` recorded for `name`, if any.
    pub fn computed_range(&self, name: &str) -> Option<(usize, u64)> {
        self.computed_ranges.get(name).copied()
    }

    /// Records that `name` sits in its buffer with a row stride of `stride`
    /// time steps while its shape declares fewer.
    ///
    /// A KV cache bound for `max_seq_len` tokens and holding more than one
    /// `(batch, kv head)` row is the case: the kernels address it through that
    /// stride, so on the device it is exact, but the leading elements of the
    /// buffer are no longer the logical tensor. [`Self::host`] refuses those
    /// values instead of downloading a prefix that interleaves real tokens with
    /// whatever the padding holds.
    pub fn mark_row_padded(&mut self, name: &str, stride: usize) {
        self.row_padded.insert(name.to_owned(), stride);
    }

    /// The row stride of a value whose buffer is padded, if it is one.
    pub fn row_padding(&self, name: &str) -> Option<usize> {
        self.row_padded.get(name).copied()
    }

    /// Hands a kernel a buffer to write one of its outputs into, instead of
    /// letting it allocate one.
    ///
    /// `capacity` is in elements and may exceed what this run produces: that is
    /// the point. A KV cache is allocated once for the longest sequence it will
    /// ever hold and written a token at a time, so the buffer outlives the run
    /// and its physical extent is not the run's logical one. The buffer is
    /// borrowed, which is also what keeps it out of `release`/`release_owned` —
    /// both only ever touch `DeviceBuffer::Owned`.
    ///
    /// Only kernels that look for it honour it; everything else allocates as
    /// before.
    pub fn bind_output(
        &mut self,
        name: &str,
        buffer: &'values vk_compute::GpuBuffer,
        capacity: usize,
    ) {
        self.bound_outputs
            .insert(name.to_owned(), (buffer, capacity));
    }

    /// The buffer bound for `name`, with its capacity in elements.
    pub fn bound_output(&self, name: &str) -> Option<(&'values vk_compute::GpuBuffer, usize)> {
        self.bound_outputs.get(name).copied()
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
        if let Some(stride) = self.row_padding(name) {
            return Err(Error::InvalidTensor(format!(
                "'{name}' lives in a buffer padded to {stride} time steps per row, so its leading \
                 elements are not the tensor its shape declares: read it row by row with that \
                 stride, or run against a cache whose length is the sequence"
            )));
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

    /// Returns several host copies with at most one device synchronization.
    ///
    /// Host-resident values and cached downloads are copied immediately. All
    /// remaining device values are copied to staging buffers together and the
    /// stream is flushed once, irrespective of the number of outputs.
    pub fn host_many(&self, names: &[&str]) -> Result<Vec<HostTensor>> {
        let mut outputs = vec![HostTensor::new(0, Vec::new(), Vec::new()); names.len()];
        self.host_many_into(names, &mut outputs)?;
        Ok(outputs)
    }

    /// [`Self::host_many`] into tensors the caller keeps across runs: each
    /// one's dtype, shape and bytes are overwritten, reusing its allocations.
    /// On error their contents are unspecified.
    pub fn host_many_into(&self, names: &[&str], outputs: &mut [HostTensor]) -> Result<()> {
        if outputs.len() != names.len() {
            return Err(Error::InvalidTensor(format!(
                "{} destinations for {} values",
                outputs.len(),
                names.len()
            )));
        }
        // the device values still to download, by destination
        let mut device = vec![None; names.len()];

        for ((&name, out), device) in names.iter().zip(outputs.iter_mut()).zip(&mut device) {
            if let Some(initializer) = self.initializers.get(name) {
                copy_into(
                    out,
                    initializer.dtype,
                    &initializer.shape,
                    &initializer.data,
                );
                continue;
            }
            if let Some(stride) = self.row_padding(name) {
                return Err(Error::InvalidTensor(format!(
                    "'{name}' lives in a buffer padded to {stride} time steps per row, so its leading \
                     elements are not the tensor its shape declares: read it row by row with that \
                     stride, or run against a cache whose length is the sequence"
                )));
            }
            match self.values.get(name) {
                Some(Tensor::Host(tensor)) => {
                    copy_into(out, tensor.dtype, &tensor.shape, &tensor.data);
                }
                Some(Tensor::Device(tensor)) => {
                    if let Some(host) = self.host_cache.borrow().get(name) {
                        copy_into(out, host.dtype, &host.shape, &host.data);
                        continue;
                    }
                    let byte_len =
                        storage_len(tensor.dtype, tensor.elem_count).ok_or_else(|| {
                            Error::InvalidTensor(format!(
                                "dtype {} has no fixed-size storage",
                                tensor.dtype
                            ))
                        })?;
                    out.dtype = tensor.dtype;
                    out.shape.clone_from(&tensor.shape);
                    resize_for_overwrite(&mut out.data, byte_len);
                    if byte_len > 0 {
                        *device = Some(tensor);
                    }
                }
                None => {
                    return Err(Error::InvalidTensor(format!(
                        "host value '{name}' not present"
                    )));
                }
            }
        }

        let mut sources: Vec<_> = outputs
            .iter_mut()
            .zip(&device)
            .filter_map(|(out, tensor)| tensor.map(|t| (t.buffer(), out.data.as_mut_slice())))
            .collect();
        self.context
            .stream_download_many_into(&mut sources)
            .map_err(backend_error)?;
        for ((&name, out), tensor) in names.iter().zip(outputs.iter()).zip(&device) {
            if tensor.is_some() && out.data.len() <= HOST_MEMO_MAX_BYTES {
                self.host_cache
                    .borrow_mut()
                    .insert(name.to_owned(), out.clone());
            }
        }
        Ok(())
    }

    pub fn dtype_of(&self, name: &str) -> Result<i32> {
        match self.values.get(name) {
            Some(Tensor::Device(tensor)) => Ok(tensor.dtype),
            Some(Tensor::Host(tensor)) => Ok(tensor.dtype),
            None => self
                .initializers
                .get(name)
                .map(|initializer| initializer.dtype)
                .or_else(|| self.released.get(name).map(|(dtype, _)| *dtype))
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
                .or_else(|| self.released.get(name).map(|(_, shape)| shape.clone()))
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
                .stream_upload_labeled(&buffer, &host.data, name)
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
        // a host value is uploaded from where it is: `host()` would clone it
        // first (21 MB of SAM 3 image embeddings per run)
        let owned;
        let host = match self.values.get(name) {
            Some(Tensor::Host(tensor)) => tensor,
            _ => {
                owned = self.host(name)?;
                &owned
            }
        };
        host.validate()?;
        if host.dtype != FLOAT {
            return Err(Error::InvalidTensor(format!(
                "ensure_device '{name}': dtype {}, expected f32",
                host.dtype
            )));
        }
        let tensor = upload_float(self.context, &host.shape, &host.data, name)?;
        self.set(name, Tensor::Device(tensor));
        Ok(())
    }

    /// Uploads a host f32 value into a buffer laid out for `capacity` elements.
    ///
    /// Same as `ensure_device` except the buffer does not follow the tensor's
    /// length. A decoder's attention bias is `[b, 1, s, total]` and grows by one
    /// element per token, so a buffer sized to its shape is a new allocation —
    /// and a new `VkBuffer` — at every step, which is the one thing a recorded
    /// descriptor set cannot survive. Sized to the cache's capacity it is the
    /// same buffer for the whole generation.
    ///
    /// The value keeps its own layout inside the buffer — the padding is all at
    /// the end — and the tail is left as the pool handed it over: the caller is
    /// the one that knows it is never read, and says so by asking for the
    /// padding. A value already on device is left alone, at whatever size
    /// produced it.
    pub fn ensure_device_padded(&mut self, name: &str, capacity: usize) -> Result<()> {
        if self.on_device(name) {
            return Ok(());
        }
        let host = self.host(name)?;
        host.validate()?;
        if host.dtype != FLOAT {
            return Err(Error::InvalidTensor(format!(
                "ensure_device_padded '{name}': dtype {}, expected f32",
                host.dtype
            )));
        }
        if host.elem_count() > capacity {
            return Err(Error::InvalidTensor(format!(
                "ensure_device_padded '{name}': {} elements do not fit a capacity of {capacity}",
                host.elem_count()
            )));
        }
        let buffer = self
            .context
            .create_storage_buffer(device_storage_bytes(FLOAT, capacity.max(1))?)
            .map_err(backend_error)?;
        if !host.data.is_empty() {
            self.context
                .stream_upload_labeled(&buffer, &host.data, name)
                .map_err(backend_error)?;
        }
        self.set(
            name,
            Tensor::Device(DeviceTensor {
                dtype: FLOAT,
                shape: host.shape.clone(),
                elem_count: host.elem_count(),
                buf: DeviceBuffer::Owned(buffer),
            }),
        );
        Ok(())
    }

    /// Forgets the host copies memoized by `host`.
    ///
    /// The memo is keyed by name and assumes a value is written once, which
    /// holds inside a run. A replayed step writes new data into the very same
    /// buffers under the very same names, so at a step boundary every memo is
    /// a copy of the previous token's answer.
    pub fn forget_host_cache(&mut self) {
        self.host_cache.borrow_mut().clear();
    }

    pub fn set(&mut self, name: &str, tensor: Tensor<'values>) {
        self.values.insert(name.to_owned(), tensor);
        self.host_cache.borrow_mut().remove(name);
        self.released.remove(name);
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
        let Some(tensor) = self.values.remove(name) else {
            return;
        };
        // What the value *was* outlives where it lived: `Shape` and the other
        // metadata readers need the dimensions, and a graph may ask for them
        // after the last node that read the bytes.
        let (dtype, shape) = match &tensor {
            Tensor::Device(tensor) => (tensor.dtype, tensor.shape.clone()),
            Tensor::Host(tensor) => (tensor.dtype, tensor.shape.clone()),
        };
        self.released.insert(name.to_owned(), (dtype, shape));
        if let Tensor::Device(DeviceTensor {
            buf: DeviceBuffer::Owned(buffer),
            ..
        }) = tensor
        {
            self.context.recycle_storage_buffer(buffer);
        }
    }

    /// Removes `name` from the environment as a device tensor the caller owns,
    /// uploading it first if it is on the host.
    ///
    /// This is how a value leaves a run without crossing the bus: the caller
    /// binds it to the next run, of this graph or another on the same context,
    /// and returns the buffer with `VkContext::recycle_storage_buffer` when done.
    /// Storage the environment only borrows (an initializer, a caller's buffer)
    /// and row-padded values are refused rather than copied.
    pub fn take_owned_device(&mut self, name: &str) -> Result<DeviceTensor<'static>> {
        if self.row_padded.contains_key(name) {
            return Err(Error::InvalidTensor(format!(
                "'{name}' is row-padded in its buffer, it cannot leave the run"
            )));
        }
        self.ensure_device_dtype(name)?;
        let Some(Tensor::Device(DeviceTensor {
            buf: DeviceBuffer::Owned(_),
            ..
        })) = self.values.get(name)
        else {
            return Err(Error::InvalidTensor(format!(
                "'{name}' lives in borrowed storage, it cannot leave the run"
            )));
        };
        let Some(Tensor::Device(DeviceTensor {
            dtype,
            shape,
            elem_count,
            buf: DeviceBuffer::Owned(buffer),
        })) = self.values.remove(name)
        else {
            unreachable!("checked above");
        };
        self.host_cache.borrow_mut().remove(name);
        self.released
            .insert(name.to_owned(), (dtype, shape.clone()));
        Ok(DeviceTensor {
            dtype,
            shape,
            elem_count,
            buf: DeviceBuffer::Owned(buffer),
        })
    }

    /// Tells the node about to run whether it is the last reader of its first
    /// input, which only the walk over the whole graph knows.
    ///
    /// A shape-only op (`Reshape`, `Squeeze`, …) uses it to take the input's
    /// buffer instead of copying it: the bytes are the same, and once the
    /// input is dead nobody else can see them change owner.
    pub fn set_first_input_dies(&mut self, dies: bool) {
        self.first_input_dies = dies;
    }

    /// Takes `name`'s buffer out of the environment, as `release` would, when
    /// the current node is its last reader and the runtime owns it.
    ///
    /// `None` leaves the value where it is: borrowed storage belongs to the
    /// host, a row-padded buffer does not hold the tensor as its shape says,
    /// and a value read later must keep its own buffer.
    pub fn take_dying_device(&mut self, name: &str) -> Option<DeviceTensor<'values>> {
        if !self.first_input_dies || self.row_padded.contains_key(name) {
            return None;
        }
        if !matches!(
            self.values.get(name),
            Some(Tensor::Device(DeviceTensor {
                buf: DeviceBuffer::Owned(_),
                ..
            }))
        ) {
            return None;
        }
        let Some(Tensor::Device(tensor)) = self.values.remove(name) else {
            return None;
        };
        self.host_cache.borrow_mut().remove(name);
        self.released
            .insert(name.to_owned(), (tensor.dtype, tensor.shape.clone()));
        Some(tensor)
    }

    /// Keeps this run's buffers pooled for the next one instead of freeing them.
    ///
    /// A decode step allocates the same sizes in the same order as the step
    /// before it, so a pool that survives the run hands the **same** buffers back
    /// in the same order — which is what makes a recorded descriptor set still
    /// valid a token later. The cost is that the whole working set stays
    /// resident between steps, which for a generation loop is the point and for
    /// a one-shot run would hold memory without bound; hence opt-in, set by the
    /// cached path. A one-shot run keeps only what fits the context's budget.
    pub fn retain_buffers(&mut self) {
        self.retain = true;
    }

    /// Releases owned intermediate buffers; borrowed buffers stay with the
    /// host that provided them.
    pub fn finish(mut self) {
        self.release_owned();
    }

    fn release_owned(&mut self) {
        // By buffer identity, not in `HashMap` order: iteration order is
        // seeded per instance, so draining directly would return the
        // buffers differently on every step. Which order in particular does
        // not matter — `reset_pool_order` canonicalizes the pool at the
        // start of the next step — only that it is the same one every time.
        let mut owned: Vec<vk_compute::GpuBuffer> = self
            .values
            .drain()
            .filter_map(|(_, tensor)| match tensor {
                Tensor::Device(DeviceTensor {
                    buf: DeviceBuffer::Owned(buffer),
                    ..
                }) => Some(buffer),
                _ => None,
            })
            .collect();
        owned.append(&mut self.ranges);
        self.range_used = 0;
        self.computed_ranges.clear();
        owned.sort_by_key(|buffer| buffer.id());
        for buffer in owned {
            self.context.recycle_storage_buffer(buffer);
        }
        if !self.retain {
            // the next run of the same graph asks for the same sizes: what
            // the budget allows stays pooled for it, the rest is freed
            self.context.trim_storage_pool();
        }
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

fn upload_float(
    context: &VkContext,
    shape: &[i64],
    bytes: &[u8],
    label: &str,
) -> Result<DeviceTensor<'static>> {
    let element_count = bytes.len() / elem_size(FLOAT);
    let buffer = context
        .create_storage_buffer(device_storage_bytes(FLOAT, element_count)?)
        .map_err(backend_error)?;
    if !bytes.is_empty() {
        context
            .stream_upload_labeled(&buffer, bytes, label)
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

/// Overwrites `out` with a host value, reusing its allocations.
fn copy_into(out: &mut HostTensor, dtype: i32, shape: &[i64], data: &[u8]) {
    out.dtype = dtype;
    out.shape.clear();
    out.shape.extend_from_slice(shape);
    out.data.clear();
    out.data.extend_from_slice(data);
}

/// Makes `data` exactly `len` bytes, to be overwritten by a download. Its
/// allocation is reused when large enough; a new one comes zeroed from the
/// allocator, so the download is its first touch, as with an owned `Vec`.
fn resize_for_overwrite(data: &mut Vec<u8>, len: usize) {
    if data.capacity() < len {
        *data = vec![0; len];
    } else {
        data.resize(len, 0);
    }
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
