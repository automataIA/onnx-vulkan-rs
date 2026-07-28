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

use onnx_vulkan_core::{Executor, Tensor};
use std::fmt;
use std::path::Path;
use std::sync::OnceLock;
use vk_compute::VkContext;

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

    /// Releases the run's device buffers.
    ///
    /// Consuming rather than `Drop` because freeing device memory can fail, and
    /// swallowing that in a destructor would hide a leak.
    pub fn finish(self) {
        self.outputs.finish();
    }
}
