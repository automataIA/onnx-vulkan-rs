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
        let tuning = onnx_vulkan_tune::runtime_resolver_from_env()
            .map_err(|error| Error::Device(format!("tuning configuration: {error}")))?;
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
        let context = context()?;
        let executor = Executor::with_tuning(context, model.graph, tuning)?;
        Ok(Self {
            executor,
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
                .map(|e| (e.present.as_str(), &e.buffer, e.row() * cache.max_seq_len))
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

    /// A decode loop that stops walking the graph once it has seen enough of it.
    ///
    /// The first steps run through the interpreter and are recorded; from then
    /// on each token issues the recorded commands with this token's scalars in
    /// them. See `onnx_vulkan_core::plan` for what makes that sound and what
    /// makes it refuse.
    pub fn decoder<'a>(&'a self, cache: &'a KvCache) -> Decoder<'a> {
        Decoder {
            session: self,
            cache,
            traces: std::collections::VecDeque::new(),
            plan: None,
            run: None,
            step: 0,
        }
    }

    /// One captured step: `run_cached`, recording what it issues.
    fn run_traced_cached<'a>(
        &'a self,
        supplied: &[(String, HostTensor)],
        cache: &'a KvCache,
    ) -> Result<(Run<'a>, onnx_vulkan_core::StepTrace)> {
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
        let (outputs, trace) = self.executor.run_traced(
            bound,
            cache
                .entries
                .iter()
                .map(|e| (e.present.as_str(), &e.buffer, e.row() * cache.max_seq_len))
                .collect(),
        )?;
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
        Ok((Run { outputs }, trace))
    }
}

/// Steps a plan is inferred from: two to see what moves between tokens, one to
/// check the answer on a step neither of them saw.
const PLAN_STEPS: usize = 3;

/// How long to keep trying before deciding this decoder has no plan.
///
/// The first tokens of a sequence are not like the ones after them — the buffer
/// pool is still filling, so a request is served a different buffer than it
/// will be later — and how many it takes to settle is a property of the graph,
/// not a constant worth guessing. So the loop simply tries again on the last
/// three steps until they agree, and gives up here.
const PLAN_ATTEMPTS: usize = 12;

/// A decode loop that replaces the interpreter with a recording of itself.
///
/// Each `step` takes the inputs that are not cache and advances the sequence by
/// one token. The first `CAPTURED_STEPS` run normally and are recorded; if they
/// agree on a plan, every later step issues that plan instead of walking the
/// graph. If they do not agree, `planned` stays false and the loop keeps
/// interpreting — the outputs are the same either way.
pub struct Decoder<'a> {
    session: &'a Session,
    cache: &'a KvCache,
    traces: std::collections::VecDeque<onnx_vulkan_core::StepTrace>,
    plan: Option<onnx_vulkan_core::StepPlan>,
    /// The run whose environment names the buffers a replayed step writes into,
    /// which is also how its outputs are read.
    run: Option<Run<'a>>,
    step: i64,
}

impl<'a> Decoder<'a> {
    /// Advances one step, from the inputs that are not the cache.
    pub fn step<N: AsRef<str>>(
        &mut self,
        inputs: impl IntoIterator<Item = (N, HostTensor)>,
    ) -> Result<()> {
        let supplied: Vec<(String, HostTensor)> = inputs
            .into_iter()
            .map(|(name, tensor)| (name.as_ref().to_string(), tensor))
            .collect();
        if let (Some(plan), Some(run)) = (self.plan.as_ref(), self.run.as_mut()) {
            // `Tensor::Host` owns its bytes, so these are free of the
            // environment's lifetime even though `supplied` is local.
            let refs: Vec<(&str, Tensor<'a>)> = supplied
                .iter()
                .map(|(name, tensor)| (name.as_str(), Tensor::Host(tensor.clone())))
                .collect();
            self.session
                .executor
                .replay_step(plan, self.step, run.outputs.env_mut(), refs)?;
            self.cache.len.set(self.cache.len.get() + 1);
            self.step += 1;
            return Ok(());
        }

        if let Some(previous) = self.run.take() {
            previous.finish();
        }
        // Step zero already has to warm the retained buffer pool and cannot
        // participate in a stable plan. Run it without capture as the tactic
        // resolution boundary too: every decode-stable concrete signature is
        // selected before a later trace can store pipeline identity.
        if self.step == 0 {
            self.run = Some(
                self.session.run_cached(
                    supplied
                        .iter()
                        .map(|(name, tensor)| (name.as_str(), tensor.clone())),
                    self.cache,
                )?,
            );
            self.step += 1;
            return Ok(());
        }
        let (run, trace) = self.session.run_traced_cached(&supplied, self.cache)?;
        self.run = Some(run);
        self.traces.push_back(trace);
        if self.traces.len() > PLAN_STEPS {
            self.traces.pop_front();
        }
        self.step += 1;
        if self.traces.len() == PLAN_STEPS && (self.step as usize) <= PLAN_ATTEMPTS {
            let recent: Vec<&onnx_vulkan_core::StepTrace> = self.traces.iter().collect();
            let origin = self.step - PLAN_STEPS as i64;
            match onnx_vulkan_core::StepPlan::build_from(
                self.session.executor.graph(),
                &recent,
                origin,
            ) {
                Ok(mut plan) => {
                    // The plan against the very step it was built to reproduce,
                    // payloads included: `StepPlan::build` checks the scalars it
                    // inferred, this checks that re-running the host nodes
                    // rebuilds the same mask and positions.
                    let last = self.step - 1;
                    let verified = {
                        let run = self.run.as_mut().expect("the step that was just captured");
                        let ir = self.session.executor.graph();
                        let env = run.outputs.env_mut();
                        // the step's own inputs, which its execution consumed
                        // and released: the host nodes read them
                        for (name, tensor) in &supplied {
                            env.set(name, Tensor::Host(tensor.clone()));
                        }
                        onnx_vulkan_core::execute_host_nodes(ir, env, plan.host_nodes())
                            .map_err(Error::from)
                            .and_then(|()| {
                                plan.verify_against(
                                    self.traces.back().expect("a trace"),
                                    last,
                                    &|name| {
                                        env.value(name)
                                            .and_then(onnx_vulkan_core::host_bytes)
                                            .ok_or_else(|| {
                                                onnx_vulkan_core::Error::InvalidTensor(
                                                    name.to_string(),
                                                )
                                            })
                                    },
                                )
                                .map_err(Error::from)
                            })
                    };
                    if let Err(error) = verified {
                        log::info!("plan rejected by its own step: {error}");
                        return Ok(());
                    }
                    log::info!(
                        "decode plan: {} dispatches, {} host nodes, from steps {origin}..{}",
                        plan.dispatches(),
                        plan.host_node_count(),
                        self.step - 1
                    );
                    plan.hold_pool(context()?);
                    self.plan = Some(plan);
                }
                Err(error) => {
                    if self.step as usize == PLAN_ATTEMPTS {
                        log::info!("no plan for this decoder, interpreting: {error}");
                    } else {
                        log::debug!("step {}: no plan yet: {error}", self.step - 1);
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether the loop is issuing a plan rather than walking the graph.
    pub fn planned(&self) -> bool {
        self.plan.is_some()
    }

    /// Structural/resource metadata of the verified live plan, if capture has
    /// converged. No Vulkan identity or command payload is exposed.
    pub fn plan_stats(&self) -> Option<onnx_vulkan_core::StepPlanStats> {
        self.plan.as_ref().map(onnx_vulkan_core::StepPlan::stats)
    }

    /// Index of the largest element along the last axis of an output.
    pub fn argmax(&mut self, name: &str) -> Result<Vec<i64>> {
        self.run
            .as_mut()
            .ok_or_else(|| Error::NoSuchValue(name.to_string()))?
            .argmax(name)
    }

    /// Reads an output on the host.
    pub fn get(&self, name: &str) -> Result<HostTensor> {
        self.run
            .as_ref()
            .ok_or_else(|| Error::NoSuchValue(name.to_string()))?
            .get(name)
    }

    /// Releases the loop's buffers.
    pub fn finish(self) -> Result<()> {
        if let Some(run) = self.run {
            run.finish();
        }
        if let Some(plan) = self.plan {
            plan.release(context()?);
        }
        Ok(())
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
        vec![
            batch as i64,
            kv_heads as i64,
            tokens as i64,
            head_size as i64,
        ]
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

    /// Reads several outputs with one GPU synchronization.
    ///
    /// Prefer this over repeated [`Self::get`] calls for models with several
    /// outputs, such as decoders exposing one key/value cache per layer.
    pub fn get_many(&self, names: &[&str]) -> Result<Vec<HostTensor>> {
        Ok(self.outputs.host_many(names)?)
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
