//! Runs any ONNX model twice — once on CPU EP only, once with the
//! Vulkan EP registered — and compares the outputs.
//!
//! Used to validate op coverage on real models without writing an app for
//! each architecture: inputs are generated from session metadata, and the
//! reference is ORT itself on the same graph. Whether the model is quantized
//! well or poorly does not matter: two EPs are compared, not two models.
//!
//! ```text
//! model-runner <model.onnx> [--dim NAME=N] [--fill NAME=V] [--iters N] [--tol F] [--seed S]
//! ```
//!
//! Dynamic dimensions default to 1 if not specified with `--dim`.

use anyhow::{Context, Result, bail};
use ep_registry::PluginEp;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, Value, ValueType};
use serde::Serialize;
use std::collections::HashMap;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    model: PathBuf,
    dims: HashMap<String, i64>,
    iters: usize,
    /// Absolute tolerance (`atol`): floor below which differences do not
    /// matter, for outputs near zero.
    tol: f64,
    /// Relative tolerance (`rtol`): an element diverges if
    /// `|Δ| > atol + rtol·|reference|`, like `allclose`. Without this, the
    /// threshold depends on the scale of the outputs, which varies by orders
    /// of magnitude across models.
    rtol: f64,
    /// How many divergent elements to print per output (`--dump N`).
    dump: usize,
    /// `--self-check`: instead of the Vulkan EP, uses a second **CPU** session
    /// with graph optimizations disabled. Measures how sensitive the model is
    /// to any numerical perturbation: in dynamic-quantization graphs scales
    /// depend on min/max, so a minimal difference can change the bucket of
    /// everything that follows.
    self_check: bool,
    /// `--no-opt`: disables graph optimizations on **both** sessions. This is
    /// needed for per-node comparison: with different optimizations the two
    /// sides run different graphs, and the fusion difference is confounded
    /// with that of the backend.
    no_opt: bool,
    seed: u64,
    /// `--no-mem-pattern` disables the ORT memory pattern planner, which from
    /// the second run onward reuses a single block and hands out **offsets**
    /// within it: the plugin's device allocator does not recognize them.
    mem_pattern: bool,
    /// `--reference DIR`: an ONNX model zoo `test_data_set_*` directory. Inputs
    /// come from `input_*.pb` instead of the generator, and both sessions are
    /// checked against `output_*.pb`.
    ///
    /// It answers a question random inputs cannot: not "do the two backends
    /// agree" but "is the answer the right one". The expected values come from
    /// the model's authors.
    reference: Option<PathBuf>,
    /// `--fill NAME=V`: fill that input with the constant `V` instead of random
    /// values. Some inputs are not free variables: an `attention_mask` of random
    /// integers makes ONNX Runtime's own `GroupQueryAttention` reject the run
    /// (`seqlens_k` is derived from its sum), so the model cannot be validated
    /// at all until the mask is a mask.
    fill: HashMap<String, i64>,
    /// `--decode N`: run `N` autoregressive steps instead of one, feeding each
    /// step's `present.*` back as the next step's `past_key_values.*` and
    /// growing `past_sequence_length` by the step's `sequence_length`.
    ///
    /// A single-shot run says nothing about a generation loop: the KV cache
    /// changes shape at every token, so buffer sizes never repeat and the
    /// allocator sees a fresh request per tensor per step. That cost is
    /// invisible at `--iters N`, which replays the *same* shapes.
    ///
    /// The two backends each feed back **their own** outputs, so any divergence
    /// compounds the way it would in a real loop rather than being reset every
    /// step.
    decode: usize,
    /// `--standalone`: also run the graph through the `onnx-vulkan` facade —
    /// same kernels, same rewrites, but the IR comes from our own parser
    /// instead of `OrtGraph`, and nothing is dispatched by ORT.
    ///
    /// The point is the **same inputs in the same process**. Comparing a
    /// standalone run against an EP run from two separate commands compares two
    /// different random inputs, which on a dynamically-quantized graph is not a
    /// comparison at all (`plan.md` Phase 1.5).
    standalone: bool,
    /// `--webgpu`: also run the graph through Microsoft's standalone WebGPU
    /// plugin EP, in the same process and on the same inputs.
    ///
    /// It is a **reference backend, not a gated one**: on Linux it runs on
    /// Dawn → Vulkan, i.e. the same device our EP uses, which makes it the
    /// sharpest external number our kernels can be read against. Unlike our
    /// all-or-nothing contract it partitions and falls back to the CPU EP for
    /// ops it lacks, so its wall is not necessarily a fully-GPU wall — read it
    /// with that caveat, and at the EP's defaults (no graph capture, no layout
    /// override).
    webgpu: bool,
    /// `--webgpu-lib PATH`: where `libonnxruntime_providers_webgpu.so` lives.
    /// Default: `third_party/webgpu-ep/`, where `scripts/fetch-webgpu-ep.sh`
    /// puts it.
    webgpu_lib: Option<PathBuf>,
    /// Writes per-output dtype-aware CPU-vs-second-backend metrics as JSON.
    report_json: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let model = PathBuf::from(
        args.next()
            .context("uso: model-runner <model.onnx> [--dim NAME=N] [--iters N] [--tol F]")?,
    );
    let mut out = Args {
        model,
        dims: HashMap::new(),
        iters: 1,
        tol: 1e-5,
        rtol: 1e-3,
        dump: 0,
        self_check: false,
        no_opt: false,
        seed: 42,
        mem_pattern: true,
        reference: None,
        fill: HashMap::new(),
        decode: 0,
        standalone: false,
        webgpu: false,
        webgpu_lib: None,
        report_json: None,
    };
    while let Some(flag) = args.next() {
        if flag == "--no-mem-pattern" {
            out.mem_pattern = false;
            continue;
        }
        if flag == "--self-check" {
            out.self_check = true;
            continue;
        }
        if flag == "--no-opt" {
            out.no_opt = true;
            continue;
        }
        if flag == "--standalone" {
            out.standalone = true;
            continue;
        }
        if flag == "--webgpu" {
            out.webgpu = true;
            continue;
        }
        let value = args
            .next()
            .with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--dim" => {
                let (name, n) = value.split_once('=').context("--dim expects NAME=N")?;
                out.dims.insert(name.to_string(), n.parse()?);
            }
            "--fill" => {
                let (name, constant) = value.split_once('=').context("--fill expects NAME=V")?;
                out.fill.insert(name.to_string(), constant.parse()?);
            }
            "--iters" => out.iters = value.parse::<usize>()?.max(1),
            "--decode" => out.decode = value.parse()?,
            "--tol" => out.tol = value.parse()?,
            "--rtol" => out.rtol = value.parse()?,
            "--dump" => out.dump = value.parse()?,
            "--seed" => out.seed = value.parse()?,
            "--reference" => out.reference = Some(PathBuf::from(value)),
            "--webgpu-lib" => out.webgpu_lib = Some(PathBuf::from(value)),
            "--report-json" => out.report_json = Some(PathBuf::from(value)),
            other => bail!("flag sconosciuto: {other}"),
        }
    }
    Ok(out)
}

fn default_ort_dylib() -> PathBuf {
    let name = if cfg!(windows) {
        "win-x64/lib/onnxruntime.dll"
    } else {
        "linux-x64/lib/libonnxruntime.so"
    };
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/onnxruntime")
        .join(name)
}

fn plugin_path() -> PathBuf {
    std::env::var("VULKAN_EP_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let lib = if cfg!(windows) {
                "onnxruntime_ep_vulkan.dll"
            } else {
                "libonnxruntime_ep_vulkan.so"
            };
            std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.join(lib)))
                .unwrap_or_else(|| PathBuf::from(lib))
        })
}

/// Where `scripts/fetch-webgpu-ep.sh` leaves Microsoft's WebGPU plugin. It is
/// a third-party artifact, not a build output, so it is not next to our binary.
fn webgpu_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../third_party/webgpu-ep/libonnxruntime_providers_webgpu.so")
}

/// Deterministic generator: the same inputs for the two sessions, without
/// depending on an RNG crate.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.0
    }

    /// Uniform f32 in `[-1, 1)`.
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 33) as f32 / (1u64 << 30) as f32) - 1.0
    }

    fn next_below(&mut self, limit: u64) -> u64 {
        (self.next_u64() >> 33) % limit
    }
}

/// An input generated from metadata: concrete shape (dynamics resolved) and
/// plausible values for the dtype. Integers stay small: they often serve as
/// indices (token id, position) and large values would make the model fail for
/// reasons unrelated to the EP.
fn make_input(
    ty: &ValueType,
    dims: &HashMap<String, i64>,
    constant: Option<i64>,
    rng: &mut Rng,
) -> Result<Value> {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols,
    } = ty
    else {
        bail!("non-tensor input not supported");
    };

    let shape: Vec<usize> = shape
        .iter()
        .zip(dimension_symbols.iter())
        .map(|(&d, symbol)| {
            if d >= 0 {
                return d as usize;
            }
            dims.get(symbol.as_str()).copied().unwrap_or(1) as usize
        })
        .collect();
    let n: usize = shape.iter().product();

    Ok(match ty {
        TensorElementType::Float32 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| match constant {
                    Some(c) => c as f32,
                    None => rng.next_f32(),
                })
                .collect::<Vec<f32>>(),
        ))?
        .into_dyn(),
        TensorElementType::Int64 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| constant.unwrap_or_else(|| rng.next_below(64) as i64))
                .collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        TensorElementType::Int32 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| constant.unwrap_or_else(|| rng.next_below(64) as i64) as i32)
                .collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        TensorElementType::Uint8 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| rng.next_below(256) as u8)
                .collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        TensorElementType::Int8 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| rng.next_below(256) as i64 as i8)
                .collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        TensorElementType::Bool => Tensor::from_array((shape, vec![true; n]))?.into_dyn(),
        other => bail!("input dtype {other:?} not handled by the runner"),
    })
}

/// `ort::Value` → `HostTensor`, so the standalone engine is fed **the exact
/// bytes** ORT was fed rather than a second generation from the same seed.
fn host_tensor_from(value: &Value) -> Result<onnx_vulkan::HostTensor> {
    use onnx_vulkan_core::host_ops::{HostTensor, INT32};
    let ValueType::Tensor { ty, shape, .. } = value.dtype() else {
        bail!("non-tensor input not supported");
    };
    let shape: Vec<i64> = shape.to_vec();
    Ok(match ty {
        TensorElementType::Float32 => {
            let (_, data) = value.try_extract_tensor::<f32>()?;
            HostTensor::from_f32(shape, data)
        }
        TensorElementType::Int64 => {
            let (_, data) = value.try_extract_tensor::<i64>()?;
            HostTensor::from_i64(shape, data)
        }
        TensorElementType::Int32 => {
            let (_, data) = value.try_extract_tensor::<i32>()?;
            let bytes = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            HostTensor::new(INT32, shape, bytes)
        }
        other => bail!("input dtype {other:?} not convertible to a host tensor"),
    })
}

#[derive(Clone, Debug)]
struct OutputTensor {
    dtype: &'static str,
    values: Vec<f64>,
    exact: Option<Vec<i64>>,
}

impl Deref for OutputTensor {
    type Target = [f64];

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<'a> IntoIterator for &'a OutputTensor {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

fn floating(dtype: &'static str, values: impl IntoIterator<Item = f64>) -> OutputTensor {
    OutputTensor {
        dtype,
        values: values.into_iter().collect(),
        exact: None,
    }
}

fn exact(dtype: &'static str, values: impl IntoIterator<Item = i64>) -> OutputTensor {
    let exact = values.into_iter().collect::<Vec<_>>();
    OutputTensor {
        dtype,
        values: exact.iter().map(|&value| value as f64).collect(),
        exact: Some(exact),
    }
}

/// Output values preserve their dtype and exact integer representation.
fn extract(value: &Value) -> Result<OutputTensor> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        bail!("non-tensor output");
    };
    Ok(match ty {
        TensorElementType::Float32 => floating(
            "float32",
            value
                .try_extract_tensor::<f32>()?
                .1
                .iter()
                .map(|&v| f64::from(v)),
        ),
        TensorElementType::Int64 => exact(
            "int64",
            value.try_extract_tensor::<i64>()?.1.iter().copied(),
        ),
        TensorElementType::Int32 => exact(
            "int32",
            value
                .try_extract_tensor::<i32>()?
                .1
                .iter()
                .map(|&v| i64::from(v)),
        ),
        TensorElementType::Bool => exact(
            "bool",
            value
                .try_extract_tensor::<bool>()?
                .1
                .iter()
                .map(|&v| i64::from(v)),
        ),
        // Quantized dtypes: most intermediates in int8 graphs.
        TensorElementType::Uint8 => exact(
            "uint8",
            value
                .try_extract_tensor::<u8>()?
                .1
                .iter()
                .map(|&v| i64::from(v)),
        ),
        TensorElementType::Int8 => exact(
            "int8",
            value
                .try_extract_tensor::<i8>()?
                .1
                .iter()
                .map(|&v| i64::from(v)),
        ),
        other => bail!("output dtype {other:?} not handled by the runner"),
    })
}

fn summarize(outputs: &SessionOutputs) -> Result<Outputs> {
    outputs
        .iter()
        .map(|(name, value)| Ok((name.to_string(), extract(&value)?)))
        .collect()
}

fn extract_host(value: &onnx_vulkan_core::HostTensor) -> Result<OutputTensor> {
    use onnx_vulkan_core::host_ops::{BOOL, FLOAT, INT8, INT32, INT64, UINT8};
    Ok(match value.dtype {
        FLOAT => floating(
            "float32",
            value
                .to_f32()
                .map_err(anyhow::Error::msg)?
                .into_iter()
                .map(f64::from),
        ),
        INT64 => exact("int64", value.to_i64().map_err(anyhow::Error::msg)?),
        INT32 => exact("int32", value.to_i64().map_err(anyhow::Error::msg)?),
        INT8 => exact("int8", value.to_i64().map_err(anyhow::Error::msg)?),
        UINT8 => exact("uint8", value.to_i64().map_err(anyhow::Error::msg)?),
        BOOL => exact("bool", value.to_i64().map_err(anyhow::Error::msg)?),
        dtype => bail!("output dtype {dtype} not handled by the runner"),
    })
}

/// Output of a run: value name and dtype-preserving contents.
type Outputs = Vec<(String, OutputTensor)>;

fn compare_tensor(
    reference: &OutputTensor,
    candidate: &OutputTensor,
    tol: f64,
    rtol: f64,
) -> onnx_vulkan_core::comparison::ComparisonReport {
    use onnx_vulkan_core::comparison::{
        ComparisonReport, FloatTolerance, compare_f64, compare_i64,
    };
    if reference.dtype != candidate.dtype {
        return ComparisonReport {
            dtype: reference.dtype,
            reference_len: reference.len(),
            candidate_len: candidate.len(),
            mismatches: reference.len().max(candidate.len()),
            max_abs: 0.0,
            max_relative: 0.0,
            matching_nan: 0,
            nan_mismatches: 0,
            matching_infinity: 0,
            infinity_mismatches: 0,
        };
    }
    match (&reference.exact, &candidate.exact) {
        (Some(reference), Some(candidate)) => compare_i64(reference, candidate),
        (None, None) => compare_f64(
            &reference.values,
            &candidate.values,
            FloatTolerance::new(tol, rtol),
        ),
        _ => ComparisonReport {
            dtype: reference.dtype,
            reference_len: reference.len(),
            candidate_len: candidate.len(),
            mismatches: reference.len().max(candidate.len()),
            max_abs: 0.0,
            max_relative: 0.0,
            matching_nan: 0,
            nan_mismatches: 0,
            matching_infinity: 0,
            infinity_mismatches: 0,
        },
    }
}

#[derive(Serialize)]
struct ComparisonJson<'a> {
    name: &'a str,
    dtype: &'static str,
    candidate_dtype: &'static str,
    passed: bool,
    reference_len: usize,
    candidate_len: usize,
    mismatches: usize,
    max_abs: f64,
    max_relative: f64,
    matching_nan: usize,
    nan_mismatches: usize,
    matching_infinity: usize,
    infinity_mismatches: usize,
}

fn comparison_json<'a>(
    name: &'a str,
    reference: &'a OutputTensor,
    candidate: &'a OutputTensor,
    tol: f64,
    rtol: f64,
) -> ComparisonJson<'a> {
    let report = compare_tensor(reference, candidate, tol, rtol);
    ComparisonJson {
        name,
        dtype: reference.dtype,
        candidate_dtype: candidate.dtype,
        passed: report.passed() && reference.dtype == candidate.dtype,
        reference_len: report.reference_len,
        candidate_len: report.candidate_len,
        mismatches: report.mismatches,
        max_abs: report.max_abs,
        max_relative: report.max_relative,
        matching_nan: report.matching_nan,
        nan_mismatches: report.nan_mismatches,
        matching_infinity: report.matching_infinity,
        infinity_mismatches: report.infinity_mismatches,
    }
}

/// The `input_*.pb` / `output_*.pb` pair of an ONNX model zoo
/// `test_data_set_*` directory.
struct Reference {
    inputs: Vec<(String, Value)>,
    outputs: Outputs,
}

/// Reads a `test_data_set_*` directory.
///
/// The files are serialized `TensorProto`s and are numbered by binding
/// position; the name inside the tensor is authoritative when present, since
/// nothing guarantees that the file order matches the session's.
fn load_reference(dir: &Path, session: &Session) -> Result<Reference> {
    let read = |prefix: &str, index: usize| -> Result<Option<(String, Vec<u8>)>> {
        let path = dir.join(format!("{prefix}_{index}.pb"));
        if !path.exists() {
            return Ok(None);
        }
        let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Some((path.display().to_string(), bytes)))
    };

    let mut inputs = Vec::new();
    for index in 0.. {
        let Some((path, bytes)) = read("input", index)? else {
            break;
        };
        let (name, tensor) = onnx_vulkan_frontend::read_tensor_proto(&bytes)
            .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        let name = if name.is_empty() {
            session
                .inputs
                .get(index)
                .map(|i| i.name.clone())
                .with_context(|| format!("{path}: no input at position {index}"))?
        } else {
            name
        };
        inputs.push((name, value_from(&tensor)?));
    }
    anyhow::ensure!(!inputs.is_empty(), "{}: no input_*.pb", dir.display());

    let mut outputs = Vec::new();
    for index in 0.. {
        let Some((path, bytes)) = read("output", index)? else {
            break;
        };
        let (name, tensor) = onnx_vulkan_frontend::read_tensor_proto(&bytes)
            .map_err(|e| anyhow::anyhow!("{path}: {e}"))?;
        let name = if name.is_empty() {
            session
                .outputs
                .get(index)
                .map(|o| o.name.clone())
                .with_context(|| format!("{path}: no output at position {index}"))?
        } else {
            name
        };
        let host = onnx_vulkan_core::HostTensor::new(tensor.dtype, tensor.shape, tensor.data);
        outputs.push((
            name,
            extract_host(&host).with_context(|| format!("{path}: extract output"))?,
        ));
    }
    anyhow::ensure!(!outputs.is_empty(), "{}: no output_*.pb", dir.display());

    Ok(Reference { inputs, outputs })
}

fn value_from(tensor: &onnx_vulkan_core::InitializerIr) -> Result<Value> {
    use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT32, INT64};
    let shape: Vec<usize> = tensor.shape.iter().map(|d| *d as usize).collect();
    let host = HostTensor::new(tensor.dtype, tensor.shape.clone(), tensor.data.clone());
    Ok(match tensor.dtype {
        FLOAT => {
            Tensor::from_array((shape, host.to_f32().map_err(anyhow::Error::msg)?))?.into_dyn()
        }
        INT64 => {
            Tensor::from_array((shape, host.to_i64().map_err(anyhow::Error::msg)?))?.into_dyn()
        }
        INT32 => Tensor::from_array((
            shape,
            host.to_i64()
                .map_err(anyhow::Error::msg)?
                .into_iter()
                .map(|v| v as i32)
                .collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        other => bail!("reference data with dtype {other} not supported"),
    })
}

/// Compares a run against the golden outputs.
///
/// Reported per backend, because the two answers are different questions: the
/// CPU EP against the golden validates the reference itself and the runtime
/// version, the Vulkan EP against the golden is what we are actually asking.
fn check_reference(label: &str, got: &Outputs, want: &Outputs, tol: f64, rtol: f64) -> bool {
    let mut ok = true;
    for (name, expected) in want {
        let Some((_, actual)) = got.iter().find(|(n, _)| n == name) else {
            println!("  reference {label}: output '{name}' not produced");
            ok = false;
            continue;
        };
        if actual.len() != expected.len() {
            println!(
                "  reference {label}: '{name}' length {} instead of {}",
                actual.len(),
                expected.len()
            );
            ok = false;
            continue;
        }
        let diff = actual
            .iter()
            .zip(expected)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mismatches = actual
            .iter()
            .zip(expected)
            .filter(|(x, y)| (*x - *y).abs() > tol + rtol * y.abs())
            .count();
        // for a classifier the argmax is the result: two close logits can
        // fall outside tolerance without changing the answer, and a different
        // argmax is an error even if the numbers look close
        let argmax = |v: &[f64]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
        };
        let (want_top, got_top) = (argmax(expected), argmax(actual));
        println!(
            "  reference {label:<9} {name:<24} max|Δ|={diff:.3e}  |ref|max={scale:.3e}  beyond tolerance: {mismatches}  argmax {:?}→{:?}",
            want_top, got_top
        );
        ok &= mismatches == 0 && want_top == got_top;
    }
    ok
}

/// Runs the model `iters` times, returning the outputs of the last iteration
/// and the timings of each iteration. The first includes pipeline compilation:
/// when measuring, use it only as a warm-up.
fn run(
    session: &mut Session,
    inputs: &[(String, Value)],
    iters: usize,
) -> Result<(Outputs, Vec<f64>)> {
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..iters {
        let feed: Vec<(String, &Value)> = inputs
            .iter()
            .map(|(name, value)| (name.clone(), value))
            .collect();
        let start = Instant::now();
        let outputs = session.run(feed)?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        last = Some(summarize(&outputs)?);
    }
    Ok((last.expect("at least one iteration"), times))
}

/// Runs the graph through the standalone facade on the same inputs.
///
/// Same kernels and same load-time rewrites as the EP — the differences are the
/// IR source (our parser rather than `OrtGraph`) and the fact that ORT dispatches
/// nothing here, so no node can quietly fall back to the CPU EP.
fn run_standalone(
    model: &Path,
    inputs: &[(String, Value)],
    iters: usize,
) -> Result<(Outputs, Vec<f64>)> {
    let session = onnx_vulkan::Session::load(model).map_err(|e| anyhow::anyhow!("{e}"))?;
    let feed: Vec<(String, onnx_vulkan::HostTensor)> = inputs
        .iter()
        .map(|(name, value)| Ok((name.clone(), host_tensor_from(value)?)))
        .collect::<Result<_>>()?;
    let names: Vec<String> = session.outputs().iter().map(|o| o.name.clone()).collect();

    let mut times = Vec::new();
    let mut last = Vec::new();
    for _ in 0..iters {
        let start = Instant::now();
        let run = session
            .run(feed.iter().map(|(n, t)| (n.as_str(), t.clone())))
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let outputs = run
            .get_many(&name_refs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        last = names
            .iter()
            .zip(outputs)
            .map(|(name, tensor)| Ok((name.clone(), extract_host(&tensor)?)))
            .collect::<Result<_>>()?;
        run.finish();
        // `Session::run` records deferred work. The inference is complete only
        // after its outputs have been read, so the wall clock includes the
        // batched readback and its single fence.
        times.push(start.elapsed().as_secs_f64() * 1000.0);
        // The Pareto is normally printed by the plugin's `OnRunEnd`, which does
        // not exist here. Without it the profiler pass of a `standalone` job
        // would report the **EP** sub-run's GPU time next to the facade's wall
        // clock — two different executions in the same row.
        //
        // Once per iteration and not once at the end, because that is what the
        // EP does and what `parse_run.py` assumes: it reads the *last* Pareto
        // block, i.e. a single steady-state run. Dumping once would sum every
        // iteration, first one included, and put a total next to a median.
        vk_compute::stats::dump_and_reset();
    }
    Ok((last, times))
}

/// The KV-cache pairing of a decoder export: output `present.N.key` feeds input
/// `past_key_values.N.key` at the next step.
///
/// Matched by name and nothing else. Other exports name the pair differently
/// (`past_key.N` / `present_key.N`); this rule covers the ones in the suite and
/// an empty result is reported rather than guessed at.
fn kv_pairs(session: &Session) -> Vec<(String, String)> {
    let inputs: Vec<&str> = session.inputs.iter().map(|i| i.name.as_str()).collect();
    session
        .outputs
        .iter()
        .filter_map(|out| {
            let suffix = out.name.strip_prefix("present")?;
            let name = format!("past_key_values{suffix}");
            inputs
                .contains(&name.as_str())
                .then_some((out.name.clone(), name))
        })
        .collect()
}

/// The step's non-cache inputs plus one backend's cache, as a borrowing feed.
fn feed_with<'a>(
    shared: &'a [(String, Value)],
    cache: &'a [(String, Value)],
) -> Vec<(String, &'a Value)> {
    shared
        .iter()
        .chain(cache)
        .map(|(name, value)| (name.clone(), value))
        .collect()
}

/// The wall time of a decode step, the cache it produced, and its other outputs.
type Step = (f64, Vec<(String, Value)>, Outputs);

/// One decode step: the wall time, the `present.*` tensors rebuilt as owned
/// values ready to be fed back, and every other output promoted to `f64`.
fn decode_step(
    session: &mut Session,
    feed: Vec<(String, &Value)>,
    pairs: &[(String, String)],
) -> Result<Step> {
    let start = Instant::now();
    let outputs = session.run(feed)?;
    let ms = start.elapsed().as_secs_f64() * 1000.0;

    let mut next = Vec::with_capacity(pairs.len());
    for (from, to) in pairs {
        let (shape, data) = outputs[from.as_str()].try_extract_tensor::<f32>()?;
        let shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        next.push((
            to.clone(),
            Tensor::from_array((shape, data.to_vec()))?.into_dyn(),
        ));
    }
    // the cache is the loop's state, not its result: what is worth comparing is
    // everything else, which for a decoder is the logits
    let cached: Vec<&str> = pairs.iter().map(|(from, _)| from.as_str()).collect();
    let rest = outputs
        .iter()
        .filter(|(name, _)| !cached.contains(name))
        .map(|(name, value)| Ok((name.to_string(), extract(&value)?)))
        .collect::<Result<Outputs>>()?;
    Ok((ms, next, rest))
}

/// Runs `steps` autoregressive steps on both sessions, comparing them at each
/// one. Returns false if any step left the tolerance.
///
/// Each backend feeds back its own cache. The inputs that are *not* cache —
/// `input_ids`, `attention_mask` — are generated once per step and shared, so
/// the only thing that can make the two diverge is the arithmetic.
fn decode_loop(
    cpu: &mut Session,
    second: &mut Session,
    label: &str,
    args: &Args,
    rng: &mut Rng,
) -> Result<bool> {
    let pairs = kv_pairs(cpu);
    if pairs.is_empty() {
        bail!("--decode: no present.*/past_key_values.* pair among the model's inputs and outputs");
    }
    let cached: Vec<&str> = pairs.iter().map(|(_, to)| to.as_str()).collect();
    let seq = args.dims.get("sequence_length").copied().unwrap_or(1);
    let mut past = args.dims.get("past_sequence_length").copied().unwrap_or(0);
    println!(
        "decode: {} steps of {seq} token(s), cache from {past} to {}, {} tensors fed back",
        args.decode,
        past + seq * args.decode as i64,
        pairs.len()
    );

    // step 0 starts from a generated cache, the same one for both backends
    let mut dims = args.dims.clone();
    let initial: Vec<(String, Value)> = cpu
        .inputs
        .iter()
        .filter(|i| cached.contains(&i.name.as_str()))
        .map(|input| {
            let value = make_input(&input.input_type, &dims, None, rng)?;
            Ok((input.name.clone(), value))
        })
        .collect::<Result<_>>()?;
    let mut cpu_cache: Vec<(String, Value)> = Vec::new();
    let mut vk_cache: Vec<(String, Value)> = Vec::new();
    let mut failed = false;

    for step in 0..args.decode {
        dims.insert("past_sequence_length".into(), past);
        dims.insert("sequence_length".into(), seq);
        dims.insert("total_sequence_length".into(), past + seq);
        let shared: Vec<(String, Value)> = cpu
            .inputs
            .iter()
            .filter(|i| !cached.contains(&i.name.as_str()))
            .map(|input| {
                let value = make_input(
                    &input.input_type,
                    &dims,
                    args.fill.get(&input.name).copied(),
                    rng,
                )
                .with_context(|| format!("input '{}'", input.name))?;
                Ok((input.name.clone(), value))
            })
            .collect::<Result<_>>()?;

        let (cpu_ms, cpu_next, cpu_out) = decode_step(
            cpu,
            feed_with(&shared, if step == 0 { &initial } else { &cpu_cache }),
            &pairs,
        )?;
        let (vk_ms, vk_next, vk_out) = decode_step(
            second,
            feed_with(&shared, if step == 0 { &initial } else { &vk_cache }),
            &pairs,
        )?;
        cpu_cache = cpu_next;
        vk_cache = vk_next;

        let (worst, mismatches) = cpu_out
            .iter()
            .zip(&vk_out)
            .map(|((name, a), (_, b))| {
                if a.len() != b.len() {
                    println!("  {name}: different lengths ({} vs {})", a.len(), b.len());
                    return (0.0, 1);
                }
                let diff = a
                    .iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max);
                let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let bad = a
                    .iter()
                    .zip(b)
                    .filter(|(x, y)| (*x - *y).abs() > args.tol + args.rtol * x.abs())
                    .count();
                (if scale > 0.0 { diff / scale } else { 0.0 }, bad)
            })
            .fold((0.0f64, 0usize), |(w, n), (r, b)| (w.max(r), n + b));
        failed |= mismatches > 0;
        println!(
            "  step {step:>3}  past={past:<6} CPU EP {cpu_ms:8.1} ms   {label} {vk_ms:8.1} ms   relative={worst:.2e}  beyond tolerance: {mismatches}"
        );
        past += seq;
    }
    Ok(!failed)
}

/// Element-wise worst relative difference between two runs, printed per output.
///
/// Separate from the CPU-vs-Vulkan block because it answers a different
/// question: not "is the GPU right" but "do our two hosts agree with each
/// other". They run the same kernels, so anything above float noise is a bug in
/// one of the two paths, not model sensitivity.
fn compare(label: &str, a: &Outputs, b: &Outputs, tol: f64, rtol: f64) -> (f64, usize) {
    let (mut worst, mut total_mismatches) = (0.0f64, 0usize);
    for ((name, x), (_, y)) in a.iter().zip(b) {
        if x.len() != y.len() {
            println!(
                "  {label} {name}: different lengths ({} vs {})",
                x.len(),
                y.len()
            );
            total_mismatches += 1;
            continue;
        }
        let diff = x
            .iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rel = if scale > 0.0 { diff / scale } else { 0.0 };
        let mismatches = x
            .iter()
            .zip(y)
            .filter(|(p, q)| (*p - *q).abs() > tol + rtol * p.abs())
            .count();
        worst = worst.max(rel);
        total_mismatches += mismatches;
        println!(
            "  {label} {name:<24} n={:<9} max|Δ|={diff:.3e}  relative={rel:.2e}  beyond tolerance: {mismatches}",
            x.len()
        );
    }
    (worst, total_mismatches)
}

/// Steady-state statistics: median, minimum and maximum of iterations after
/// the first (which includes pipeline compilation).
fn steady(times: &[f64]) -> (f64, f64, f64) {
    let tail = if times.len() > 1 { &times[1..] } else { times };
    let mut sorted = tail.to_vec();
    sorted.sort_by(f64::total_cmp);
    (
        sorted[sorted.len() / 2],
        sorted[0],
        sorted[sorted.len() - 1],
    )
}

fn main() -> Result<()> {
    env_logger::init();
    let args = parse_args()?;
    let dylib = std::env::var("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_ort_dylib());
    ort::init_from(dylib.to_string_lossy().as_ref()).commit()?;

    println!("modello: {}", args.model.display());

    // reference session: CPU EP only
    let opt = |b: ort::session::builder::SessionBuilder| {
        if args.no_opt {
            b.with_optimization_level(GraphOptimizationLevel::Disable)
        } else {
            Ok(b)
        }
    };
    let mut cpu = opt(Session::builder()?.with_memory_pattern(args.mem_pattern)?)?
        .commit_from_file(&args.model)?;
    let reference = args
        .reference
        .as_deref()
        .map(|dir| {
            load_reference(dir, &cpu).with_context(|| format!("reference {}", dir.display()))
        })
        .transpose()?;

    // the reference inputs are moved here: only the expected outputs remain,
    // needed after the two executions
    let (reference_inputs, reference) = match reference {
        Some(r) => (Some(r.inputs), Some(r.outputs)),
        None => (None, None),
    };

    let mut rng = Rng(args.seed);

    // second session: Vulkan EP, or CPU without optimizations in self-check.
    // Built before the first run because `--decode` interleaves the two.
    let vulkan_ep = PluginEp::vulkan(plugin_path());
    let mut registered = false;
    let mut second = if args.self_check {
        Session::builder()?
            .with_memory_pattern(args.mem_pattern)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .commit_from_file(&args.model)?
    } else {
        if !vulkan_ep.library.exists() {
            bail!(
                "plugin not found in {} (VULKAN_EP_PATH)",
                vulkan_ep.library.display()
            );
        }
        vulkan_ep.register()?;
        registered = true;
        let mut builder = opt(Session::builder()?.with_memory_pattern(args.mem_pattern)?)?;
        let devices = vulkan_ep.append_to_session(&mut builder)?;
        println!("(Vulkan EP: {devices} device)");
        builder.commit_from_file(&args.model)?
    };
    let label = if args.self_check {
        "CPU no-opt"
    } else {
        "Vulkan EP"
    };
    if args.decode > 0 {
        let ok = decode_loop(&mut cpu, &mut second, label, &args, &mut rng)?;
        drop(second);
        if registered {
            vulkan_ep.unregister()?;
        }
        if !ok {
            bail!(
                "divergent outputs beyond atol={:.1e} + rtol={:.1e}·|ref|",
                args.tol,
                args.rtol
            );
        }
        println!("OK: {} decode steps within tolerance", args.decode);
        return Ok(());
    }

    let inputs: Vec<(String, Value)> = match reference_inputs {
        Some(given) => given
            .into_iter()
            .map(|(name, value)| {
                let ValueType::Tensor { shape, .. } = value.dtype() else {
                    unreachable!("the reference contains tensors")
                };
                println!("  input {name:<28} {:?}  (reference)", shape.as_ref());
                Ok((name, value))
            })
            .collect::<Result<_>>()?,
        None => cpu
            .inputs
            .iter()
            .map(|input| {
                let value = make_input(
                    &input.input_type,
                    &args.dims,
                    args.fill.get(&input.name).copied(),
                    &mut rng,
                )
                .with_context(|| format!("input '{}'", input.name))?;
                let ValueType::Tensor { shape, .. } = value.dtype() else {
                    unreachable!("make_input produces tensors")
                };
                println!("  input {:<28} {:?}", input.name, shape.as_ref());
                Ok((input.name.clone(), value))
            })
            .collect::<Result<_>>()?,
    };

    let (cpu_out, cpu_times) = run(&mut cpu, &inputs, args.iters)?;
    let (cpu_ms, cpu_min, cpu_max) = steady(&cpu_times);
    println!("CPU EP:    {cpu_ms:8.1} ms (regime)  [min {cpu_min:.1} max {cpu_max:.1}]");

    let (vk_out, vk_times) = run(&mut second, &inputs, args.iters)?;
    let (vk_ms, vk_min, vk_max) = steady(&vk_times);
    println!("{label}: {vk_ms:8.1} ms (regime)  [min {vk_min:.1} max {vk_max:.1}]");

    // comparison
    let mut worst = 0.0f64;
    let mut failed = false;
    for ((name, a), (_, b)) in cpu_out.iter().zip(&vk_out) {
        let report = compare_tensor(a, b, args.tol, args.rtol);
        let diff = report.max_abs;
        // reference scale, to give meaning to the absolute error
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rel = if scale > 0.0 { diff / scale } else { 0.0 };
        let mismatches = report.mismatches;
        worst = worst.max(rel);
        println!(
            "  output {name:<28} n={:<9} max|Δ|={diff:.3e}  |ref|max={scale:.3e}               relative={rel:.2e}  beyond tolerance: {mismatches}",
            a.len()
        );
        failed |= !report.passed() || a.dtype != b.dtype;
        if args.dump > 0 {
            for (i, (x, y)) in a
                .iter()
                .zip(b)
                .enumerate()
                .filter(|(_, (x, y))| (*x - *y).abs() > args.tol + args.rtol * x.abs())
                .take(args.dump)
            {
                println!("    [{i}] cpu={x:+.6e}  vulkan={y:+.6e}");
            }
        }
    }

    if let Some(path) = &args.report_json {
        #[derive(Serialize)]
        struct Report<'a> {
            schema_version: u32,
            model: String,
            reference_backend: &'static str,
            candidate_backend: &'a str,
            atol: f64,
            rtol: f64,
            outputs: Vec<ComparisonJson<'a>>,
        }
        let outputs = cpu_out
            .iter()
            .zip(&vk_out)
            .map(|((name, reference), (_, candidate))| {
                comparison_json(name, reference, candidate, args.tol, args.rtol)
            })
            .collect();
        let report = Report {
            schema_version: 1,
            model: args.model.display().to_string(),
            reference_backend: "cpu",
            candidate_backend: label,
            atol: args.tol,
            rtol: args.rtol,
            outputs,
        };
        let mut bytes = serde_json::to_vec_pretty(&report)?;
        bytes.push(b'\n');
        std::fs::write(path, bytes)
            .with_context(|| format!("writing comparison report {}", path.display()))?;
    }

    // third backend: the standalone engine, on the very same inputs. The EP run
    // has to be finished first — both want the Vulkan device.
    let standalone_out = if args.standalone {
        let (out, times) = run_standalone(&args.model, &inputs, args.iters)?;
        let (ms, min, max) = steady(&times);
        println!("standalone: {ms:8.1} ms (regime)  [min {min:.1} max {max:.1}]");
        let (rel_ep, mismatches) =
            compare("standalone vs EP  ", &vk_out, &out, args.tol, args.rtol);
        let (rel_cpu, _) = compare("standalone vs CPU ", &cpu_out, &out, args.tol, args.rtol);
        println!(
            "parity: standalone vs EP relative={rel_ep:.2e}, vs CPU EP relative={rel_cpu:.2e}"
        );
        // the two hosts run the same kernels: a divergence here is ours
        failed |= mismatches > 0;
        Some(out)
    } else {
        None
    };

    // fourth backend: Microsoft's standalone WebGPU plugin EP, on the very same
    // inputs. Registered last and torn down inside this block, for the reason
    // the standalone run has: Dawn opens its own Vulkan device on this GPU.
    //
    // Its outputs are compared but **never** set `failed`: this is an external
    // reference, and a divergence in it is not a defect of ours.
    let webgpu_out = if args.webgpu {
        let ep = PluginEp::webgpu(args.webgpu_lib.clone().unwrap_or_else(webgpu_path));
        if !ep.library.exists() {
            bail!(
                "WebGPU EP not found at {} — run scripts/fetch-webgpu-ep.sh",
                ep.library.display()
            );
        }
        ep.register()?;
        // the EP's own defaults: no graph capture, no preferred-layout
        // override. Tuning one side and not the other is not a comparison.
        let mut builder = opt(Session::builder()?.with_memory_pattern(args.mem_pattern)?)?;
        let devices = ep.append_to_session(&mut builder)?;
        println!("(WebGPU EP: {devices} device)");
        let mut session = builder.commit_from_file(&args.model)?;
        let (out, times) = run(&mut session, &inputs, args.iters)?;
        let (ms, min, max) = steady(&times);
        println!("webgpu:    {ms:8.1} ms (regime)  [min {min:.1} max {max:.1}]");
        let (rel_ep, _) = compare("webgpu vs EP      ", &vk_out, &out, args.tol, args.rtol);
        let (rel_cpu, _) = compare("webgpu vs CPU     ", &cpu_out, &out, args.tol, args.rtol);
        println!("reference: webgpu vs EP relative={rel_ep:.2e}, vs CPU EP relative={rel_cpu:.2e}");
        drop(session);
        ep.unregister()?;
        Some(out)
    } else {
        None
    };

    if let Some(reference) = &reference {
        let backend = if args.self_check {
            "cpu-no-opt"
        } else {
            "vulkan"
        };
        let cpu_ok = check_reference("cpu", &cpu_out, reference, args.tol, args.rtol);
        let second_ok = check_reference(backend, &vk_out, reference, args.tol, args.rtol);
        if let Some(out) = &standalone_out {
            failed |= !check_reference("standalone", out, reference, args.tol, args.rtol);
        }
        if let Some(out) = &webgpu_out {
            // printed, not gated: see the block above
            check_reference("webgpu", out, reference, args.tol, args.rtol);
        }
        if !cpu_ok {
            // the CPU EP out of tolerance from the golden says nothing about
            // our backend: either the reference does not belong to this model,
            // or the ORT version has changed the result
            println!("  ! the CPU EP itself diverges from the reference: comparison inconclusive");
        }
        failed |= !second_ok;
    }

    drop(second);
    if registered {
        vulkan_ep.unregister()?;
    }

    if failed {
        bail!(
            "divergent outputs beyond atol={:.1e} + rtol={:.1e}·|ref|",
            args.tol,
            args.rtol
        );
    }
    println!(
        "OK: within atol={:.1e} + rtol={:.1e}·|ref| (worst relative error {worst:.2e})",
        args.tol, args.rtol
    );
    Ok(())
}
