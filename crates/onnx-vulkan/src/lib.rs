//! Runs an ONNX model on a Vulkan GPU, in pure Rust, with no ONNX Runtime.
//!
//! ```no_run
//! let session = onnx_vulkan::Session::load("model.onnx")?;
//! for input in session.inputs() {
//!     println!("{}: {input}", input.name);
//! }
//! let run = session.run([("data", onnx_vulkan::HostTensor::from_f32(vec![1, 3, 224, 224], &[0.0; 150528]))])?;
//! let output = run.get("output")?;
//! run.finish();
//! # Ok::<(), onnx_vulkan::Error>(())
//! ```
//!
//! Three properties are the point of this crate, and each is a contract rather
//! than a default:
//!
//! - **Nothing native ships with it.** The only shared library the process opens
//!   is the system Vulkan loader, the way a program opens `libc`. `ldd` on a
//!   binary built against this crate lists no ONNX Runtime.
//! - **All or nothing.** [`Session::load`] refuses a model containing any node
//!   the engine cannot run, and the error names every one of them. There is no
//!   per-op fallback to the CPU, silent or otherwise: a session that exists runs
//!   entirely on the GPU.
//! - **The model stays on the device.** Weights are uploaded and pipelines
//!   compiled on the first [`Session::run`] and reused by every later one, for
//!   as long as the session lives. Loading is the expensive call; running is not.
//!
//! The Vulkan device is created once per process, on first use.

use onnx_vulkan_core::{DeviceBuffer, DeviceTensor, Executor, Tensor};
use std::cell::Cell;
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;
use vk_compute::{GpuBuffer, VkContext};

pub use onnx_vulkan_core::graph::ElementType;
pub use onnx_vulkan_core::host_ops::HostTensor;
pub use onnx_vulkan_frontend::Dim;

/// Why a model could not be loaded or run.
#[derive(Debug)]
pub enum Error {
    /// The file is not a readable ONNX model.
    Load(onnx_vulkan_frontend::Error),
    /// The graph contains nodes this engine does not implement. All-or-nothing:
    /// the message lists them, because a partial answer is not one.
    Unsupported(String),
    /// Vulkan is unavailable, or a device operation failed.
    Device(String),
    /// A value the caller asked for is not one this graph produces.
    NoSuchValue(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(e) => write!(f, "loading the model: {e}"),
            Self::Unsupported(m) => write!(f, "model not fully supported: {m}"),
            Self::Device(m) => write!(f, "Vulkan: {m}"),
            Self::NoSuchValue(name) => write!(f, "'{name}' is not a value of this graph"),
        }
    }
}

impl std::error::Error for Error {}

impl From<onnx_vulkan_frontend::Error> for Error {
    fn from(e: onnx_vulkan_frontend::Error) -> Self {
        Self::Load(e)
    }
}

impl From<onnx_vulkan_core::Error> for Error {
    fn from(e: onnx_vulkan_core::Error) -> Self {
        match e {
            onnx_vulkan_core::Error::Unsupported(m) => Self::Unsupported(m),
            other => Self::Device(other.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Name, element type and shape of a graph input or output.
///
/// The shape can carry symbols: an exported model states that its first
/// dimension is `batch`, not that it is 1. Whoever builds the tensor is who
/// knows the value, so the symbol is reported rather than guessed at.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: Option<ElementType>,
    pub shape: Option<Vec<Dim>>,
}

impl fmt::Display for TensorInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.dtype {
            Some(d) => write!(f, "{d:?}")?,
            None => write!(f, "?")?,
        }
        match &self.shape {
            None => write!(f, "[?]"),
            Some(dims) => {
                write!(f, "[")?;
                for (i, dim) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    match dim {
                        Dim::Fixed(n) => write!(f, "{n}")?,
                        Dim::Symbol(s) => write!(f, "{s}")?,
                        Dim::Unknown => write!(f, "?")?,
                    }
                }
                write!(f, "]")
            }
        }
    }
}

/// The Vulkan device, created once per process on first use.
///
/// One device per process is the same choice the ORT plugin makes. It is what
/// lets a `Session` be a plain owned value instead of borrowing a context the
/// caller has to keep alive alongside it.
fn context() -> Result<&'static VkContext> {
    static CTX: OnceLock<std::result::Result<VkContext, String>> = OnceLock::new();
    CTX.get_or_init(|| VkContext::new().map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| Error::Device(e.clone()))
}

/// A model loaded onto the GPU, ready to run any number of times.
pub struct Session {
    executor: Executor<'static>,
    inputs: Vec<TensorInfo>,
    outputs: Vec<TensorInfo>,
}

impl Session {
    /// Loads an `.onnx` file, rewrites it, and prepares it on the GPU.
    ///
    /// External weights are resolved relative to the model's own directory, as
    /// the ONNX specification prescribes.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_model(onnx_vulkan_frontend::load(path)?)
    }

    /// Same as [`Session::load`], for a model already in memory. `base_dir` is
    /// where external weights are looked up; `None` rejects a model that uses
    /// them rather than guessing where they live.
    pub fn load_from_bytes(bytes: &[u8], base_dir: Option<&Path>) -> Result<Self> {
        Self::from_model(onnx_vulkan_frontend::load_from_bytes(bytes, base_dir)?)
    }

    fn from_model(model: onnx_vulkan_frontend::Model) -> Result<Self> {
        for conflict in &model.conflicts {
            log::warn!("shape inference: {conflict}");
        }
        let describe = |names: &[String]| -> Vec<TensorInfo> {
            names
                .iter()
                .map(|name| {
                    let declared = model.types.get(name);
                    TensorInfo {
                        name: name.clone(),
                        dtype: declared
                            .and_then(|t| t.dtype)
                            .and_then(|d| ElementType::try_from(d).ok()),
                        shape: declared.and_then(|t| t.shape.clone()),
                    }
                })
                .collect()
        };
        let inputs = describe(&model.graph.inputs);
        let outputs = describe(&model.graph.outputs);
        Ok(Self {
            executor: Executor::new(context()?, model.graph)?,
            inputs,
            outputs,
        })
    }

    /// What the caller must supply, in the graph's own order.
    pub fn inputs(&self) -> &[TensorInfo] {
        &self.inputs
    }

    /// What a run produces, in the graph's own order.
    pub fn outputs(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// How many nodes the graph runs after the load-time rewrites.
    pub fn node_count(&self) -> usize {
        self.executor.graph().nodes.len()
    }

    /// Runs the graph once.
    ///
    /// The returned [`Run`] borrows the session, so its outputs stay readable
    /// until it is dropped; the session's weights and pipelines survive it and
    /// serve the next run.
    pub fn run<'a, N>(
        &'a self,
        inputs: impl IntoIterator<Item = (N, HostTensor)>,
    ) -> Result<Run<'a>>
    where
        N: AsRef<str>,
    {
        let supplied: Vec<(String, HostTensor)> = inputs
            .into_iter()
            .map(|(name, tensor)| (name.as_ref().to_string(), tensor))
            .collect();
        for (name, _) in &supplied {
            if !self.inputs.iter().any(|i| &i.name == name) {
                return Err(Error::NoSuchValue(name.clone()));
            }
        }
        for input in &self.inputs {
            if !supplied.iter().any(|(name, _)| name == &input.name) {
                return Err(Error::NoSuchValue(format!(
                    "{} (a required input was not supplied)",
                    input.name
                )));
            }
        }
        let bound: Vec<(&str, Tensor<'a>)> = supplied
            .iter()
            .map(|(name, tensor)| (name.as_str(), Tensor::Host(tensor.clone())))
            .collect();
        Ok(Run {
            outputs: self.executor.run(bound)?,
        })
    }

    /// Runs one step of a sequence against a resident cache.
    ///
    /// The caller supplies only the inputs that are not cache — the tokens,
    /// the mask, the positions. The `past_key_values.*` are the cache itself,
    /// bound as views over the buffers the previous step wrote, and the
    /// `present.*` are bound as outputs onto that same memory. Nothing about
    /// the cache crosses the PCIe bus.
    ///
    /// The cache's length advances by the sequence length this step produced,
    /// so a prefill of `n` tokens and a decode of one are the same call.
    pub fn run_cached<'a, N>(
        &'a self,
        inputs: impl IntoIterator<Item = (N, HostTensor)>,
        cache: &'a KvCache,
    ) -> Result<Run<'a>>
    where
        N: AsRef<str>,
    {
        let supplied: Vec<(String, HostTensor)> = inputs
            .into_iter()
            .map(|(name, tensor)| (name.as_ref().to_string(), tensor))
            .collect();
        for (name, _) in &supplied {
            if !self.inputs.iter().any(|i| &i.name == name) {
                return Err(Error::NoSuchValue(name.clone()));
            }
        }
        for input in &self.inputs {
            let supplied = supplied.iter().any(|(name, _)| name == &input.name);
            let cached = cache.entries.iter().any(|e| e.past == input.name);
            if !supplied && !cached {
                return Err(Error::NoSuchValue(format!(
                    "{} (a required input was neither supplied nor in the cache)",
                    input.name
                )));
            }
        }
        let past = cache.len.get();
        let mut bound: Vec<(&str, Tensor<'a>)> = supplied
            .iter()
            .map(|(name, tensor)| (name.as_str(), Tensor::Host(tensor.clone())))
            .collect();
        for entry in &cache.entries {
            bound.push((
                entry.past.as_str(),
                Tensor::Device(DeviceTensor {
                    dtype: onnx_vulkan_core::host_ops::FLOAT,
                    shape: entry.cache_shape(past),
                    elem_count: entry.row() * past,
                    buf: DeviceBuffer::Borrowed(&entry.buffer),
                }),
            ));
        }
        let outputs = self.executor.run_with_outputs(
            bound,
            cache
                .entries
                .iter()
                .map(|e| {
                    (
                        e.present.as_str(),
                        &e.buffer,
                        e.row() * cache.max_seq_len,
                    )
                })
                .collect(),
        )?;
        // the graph decides how many tokens went in; reading it back from the
        // cache it wrote is one source of truth instead of two
        let total = outputs
            .shape_of(&cache.entries[0].present)?
            .get(2)
            .copied()
            .unwrap_or(past as i64)
            .max(0) as usize;
        if total > cache.max_seq_len {
            return Err(Error::Device(format!(
                "the sequence reached {total} tokens, past the cache's {} \
                 (allocate a longer one)",
                cache.max_seq_len
            )));
        }
        cache.len.set(total);
        Ok(Run { outputs })
    }
}

impl CacheEntry {
    /// `[batch, kv_heads, tokens, head_size]` is the declared shape, but the
    /// engine only reads the time axis and the total element count, and the
    /// rest of the layout is folded into `row`. At `batch · kv_heads == 1` this
    /// is also the written prefix of the buffer; with more rows it is the
    /// logical cache and the rows sit `max_seq_len` apart, which the kernels
    /// address through the stride and a host download does not.
    fn cache_shape(&self, tokens: usize) -> Vec<i64> {
        let [batch, kv_heads, head_size] = self.dims;
        vec![batch as i64, kv_heads as i64, tokens as i64, head_size as i64]
    }

    /// Elements per token.
    fn row(&self) -> usize {
        self.dims.iter().product()
    }
}

/// A key/value cache that lives in VRAM across runs.
///
/// One buffer per `present.*` output, laid out for `max_seq_len` tokens and
/// written a step at a time. What it removes is not one copy but three: the
/// download of `present`, the upload of `past`, and the `GQA_past` dispatch
/// that rebuilt the cache inside the graph. Source and destination of a decode
/// step are the same memory, so the step writes only its own token.
///
/// The cache is the caller's, not the session's: a session serves any number
/// of independent sequences, and each has its own cache and its own length.
/// The buffers are handed to a run as borrows, which is also what keeps them
/// out of the run's teardown — `ExecutionEnv` only ever reclaims what it owns.
pub struct KvCache {
    entries: Vec<CacheEntry>,
    max_seq_len: usize,
    len: Cell<usize>,
}

struct CacheEntry {
    /// Graph output written by a run, and the input it comes back as.
    present: String,
    past: String,
    buffer: GpuBuffer,
    /// `batch`, `kv_heads` and `head_size` of `[batch, kv_heads, T, head_size]`
    /// — everything but the time axis, which is what varies.
    dims: [usize; 3],
}

impl KvCache {
    /// Allocates a cache for every `present.*`/`past_key_values.*` pair the
    /// model declares, sized for `max_seq_len` tokens.
    ///
    /// The pairing is by name, which is the convention every exported decoder
    /// follows; a model with no such pair is an error rather than an empty
    /// cache, because running it through here would silently do nothing.
    pub fn new(session: &Session, max_seq_len: usize) -> Result<Self> {
        let context = context()?;
        let inputs: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
        let mut entries = Vec::new();
        for output in &session.outputs {
            let Some(suffix) = output.name.strip_prefix("present") else {
                continue;
            };
            let past = format!("past_key_values{suffix}");
            if !inputs.contains(&past.as_str()) {
                continue;
            }
            // `[batch, kv_heads, total, head_size]`: every dimension but the
            // time axis has to be a number, because it is what the layout is
            // computed from. The time axis is the symbol that varies.
            let shape = output.shape.as_ref().ok_or_else(|| {
                Error::Unsupported(format!("{} has no declared shape", output.name))
            })?;
            if shape.len() != 4 {
                return Err(Error::Unsupported(format!(
                    "{} is not a [batch, kv_heads, total, head_size] cache",
                    output.name
                )));
            }
            let mut dims = [0usize; 3];
            for (slot, axis) in [0usize, 1, 3].into_iter().enumerate() {
                match shape[axis] {
                    Dim::Fixed(n) if n > 0 => dims[slot] = n as usize,
                    // a decoder exports its batch as a symbol, and here it can
                    // only be 1: the cache is laid out for one sequence, and a
                    // run that then asks for a wider batch fails against this
                    // buffer's size rather than reading the wrong tokens.
                    // `kv_heads` is a different matter — it is fixed in the
                    // file, and grouped attention (`1 < kv_heads < num_heads`,
                    // which is what Llama 3 and Qwen export) is supported: the
                    // rows sit `max_seq_len` apart and the kernels stride over
                    // them.
                    Dim::Symbol(_) => dims[slot] = 1,
                    _ => {
                        return Err(Error::Unsupported(format!(
                            "{}: axis {axis} is neither fixed nor symbolic, so the cache layout \
                             is unknown",
                            output.name
                        )));
                    }
                }
            }
            let row: usize = dims.iter().product();
            let bytes = (row * max_seq_len).max(1) as u64 * 4;
            let buffer = context
                .create_storage_buffer(bytes)
                .map_err(|e| Error::Device(format!("{e:#}")))?;
            entries.push(CacheEntry {
                present: output.name.clone(),
                past,
                buffer,
                dims,
            });
        }
        if entries.is_empty() {
            return Err(Error::Unsupported(
                "no present.*/past_key_values.* pair: this model has no KV cache".into(),
            ));
        }
        Ok(Self {
            entries,
            max_seq_len,
            len: Cell::new(0),
        })
    }

    /// Tokens currently held.
    pub fn len(&self) -> usize {
        self.len.get()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens it was allocated for.
    pub fn capacity(&self) -> usize {
        self.max_seq_len
    }

    /// Forgets the sequence, keeping the memory for the next one.
    pub fn clear(&self) {
        self.len.set(0);
    }
}

/// The values one run produced, readable until dropped.
pub struct Run<'a> {
    outputs: onnx_vulkan_core::Outputs<'a>,
}

impl Run<'_> {
    /// Reads an output on the host, downloading it from VRAM. This is the
    /// synchronization point: it waits for the GPU.
    pub fn get(&self, name: &str) -> Result<HostTensor> {
        Ok(self.outputs.host(name)?)
    }

    /// Index of the largest element along the last axis of an output, computed
    /// on the GPU.
    ///
    /// A decoder emits logits and no `argmax`, so greedy sampling would
    /// otherwise mean downloading a row per token — 151936 floats on gemma3-1b
    /// — to keep one index. This enqueues the reduction into the command buffer
    /// the run is still building, so what crosses the bus is one index per row
    /// and the step keeps a single flush.
    pub fn argmax(&mut self, name: &str) -> Result<Vec<i64>> {
        Ok(self.outputs.argmax(name)?)
    }

    /// Releases the run's device buffers.
    ///
    /// Consuming rather than `Drop` because freeing device memory can fail, and
    /// swallowing that in a destructor would hide a leak.
    pub fn finish(self) {
        self.outputs.finish();
    }
}
