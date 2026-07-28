//! Runs any ONNX model twice — once on CPU EP only, once with the
//! Vulkan EP registered — and compares the outputs.
//!
//! Used to validate op coverage on real models without writing an app for
//! each architecture: inputs are generated from session metadata, and the
//! reference is ORT itself on the same graph. Whether the model is quantized
//! well or poorly does not matter: two EPs are compared, not two models.
//!
//! ```text
//! model-runner <model.onnx> [--dim NAME=N] [--iters N] [--tol F] [--seed S]
//! ```
//!
//! Dynamic dimensions default to 1 if not specified with `--dim`.

mod plugin;

use anyhow::{Context, Result, bail};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionOutputs};
use ort::tensor::TensorElementType;
use ort::value::{Tensor, Value, ValueType};
use std::collections::HashMap;
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
        let value = args
            .next()
            .with_context(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--dim" => {
                let (name, n) = value.split_once('=').context("--dim expects NAME=N")?;
                out.dims.insert(name.to_string(), n.parse()?);
            }
            "--iters" => out.iters = value.parse::<usize>()?.max(1),
            "--tol" => out.tol = value.parse()?,
            "--rtol" => out.rtol = value.parse()?,
            "--dump" => out.dump = value.parse()?,
            "--seed" => out.seed = value.parse()?,
            "--reference" => out.reference = Some(PathBuf::from(value)),
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
fn make_input(ty: &ValueType, dims: &HashMap<String, i64>, rng: &mut Rng) -> Result<Value> {
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
        TensorElementType::Float32 => {
            Tensor::from_array((shape, (0..n).map(|_| rng.next_f32()).collect::<Vec<f32>>()))?
                .into_dyn()
        }
        TensorElementType::Int64 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| rng.next_below(64) as i64)
                .collect::<Vec<_>>(),
        ))?
        .into_dyn(),
        TensorElementType::Int32 => Tensor::from_array((
            shape,
            (0..n)
                .map(|_| rng.next_below(64) as i32)
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

/// Output values as `f64`, to compare graphs with different dtypes.
fn extract(value: &Value) -> Result<Vec<f64>> {
    let ValueType::Tensor { ty, .. } = value.dtype() else {
        bail!("non-tensor output");
    };
    Ok(match ty {
        TensorElementType::Float32 => value
            .try_extract_tensor::<f32>()?
            .1
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        TensorElementType::Int64 => value
            .try_extract_tensor::<i64>()?
            .1
            .iter()
            .map(|&v| v as f64)
            .collect(),
        TensorElementType::Int32 => value
            .try_extract_tensor::<i32>()?
            .1
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        TensorElementType::Bool => value
            .try_extract_tensor::<bool>()?
            .1
            .iter()
            .map(|&v| f64::from(u8::from(v)))
            .collect(),
        // Quantized dtypes: most intermediates in int8 graphs.
        TensorElementType::Uint8 => value
            .try_extract_tensor::<u8>()?
            .1
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        TensorElementType::Int8 => value
            .try_extract_tensor::<i8>()?
            .1
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        other => bail!("output dtype {other:?} not handled by the runner"),
    })
}

fn summarize(outputs: &SessionOutputs) -> Result<Outputs> {
    outputs
        .iter()
        .map(|(name, value)| Ok((name.to_string(), extract(&value)?)))
        .collect()
}

/// Output of a run: value name and contents promoted to `f64`.
type Outputs = Vec<(String, Vec<f64>)>;

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
        let bytes =
            std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
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
            host.to_f32()
                .map_err(|e| anyhow::anyhow!("{path}: {e}"))?
                .into_iter()
                .map(f64::from)
                .collect(),
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
                let value = make_input(&input.input_type, &args.dims, &mut rng)
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

    // second session: Vulkan EP, or CPU without optimizations in self-check
    let mut registered = false;
    let mut second = if args.self_check {
        Session::builder()?
            .with_memory_pattern(args.mem_pattern)?
            .with_optimization_level(GraphOptimizationLevel::Disable)?
            .commit_from_file(&args.model)?
    } else {
        let path = plugin_path();
        if !path.exists() {
            bail!("plugin not found in {} (VULKAN_EP_PATH)", path.display());
        }
        plugin::register(&path)?;
        registered = true;
        let mut builder = opt(Session::builder()?.with_memory_pattern(args.mem_pattern)?)?;
        let devices = plugin::append_to_session(&mut builder)?;
        println!("(Vulkan EP: {devices} device)");
        builder.commit_from_file(&args.model)?
    };
    let (vk_out, vk_times) = run(&mut second, &inputs, args.iters)?;
    let (vk_ms, vk_min, vk_max) = steady(&vk_times);
    println!(
        "{}: {vk_ms:8.1} ms (regime)  [min {vk_min:.1} max {vk_max:.1}]",
        if args.self_check {
            "CPU no-opt"
        } else {
            "Vulkan EP"
        }
    );

    // comparison
    let mut worst = 0.0f64;
    let mut failed = false;
    for ((name, a), (_, b)) in cpu_out.iter().zip(&vk_out) {
        if a.len() != b.len() {
            println!("  {name}: different lengths ({} vs {})", a.len(), b.len());
            failed = true;
            continue;
        }
        let diff = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        // reference scale, to give meaning to the absolute error
        let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let rel = if scale > 0.0 { diff / scale } else { 0.0 };
        let mismatches = a
            .iter()
            .zip(b)
            .filter(|(x, y)| (*x - *y).abs() > args.tol + args.rtol * x.abs())
            .count();
        worst = worst.max(rel);
        println!(
            "  output {name:<28} n={:<9} max|Δ|={diff:.3e}  |ref|max={scale:.3e}               relative={rel:.2e}  beyond tolerance: {mismatches}",
            a.len()
        );
        failed |= mismatches > 0;
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

    if let Some(reference) = &reference {
        let backend = if args.self_check {
            "cpu-no-opt"
        } else {
            "vulkan"
        };
        let cpu_ok = check_reference("cpu", &cpu_out, reference, args.tol, args.rtol);
        let second_ok = check_reference(backend, &vk_out, reference, args.tol, args.rtol);
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
        plugin::unregister()?;
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
