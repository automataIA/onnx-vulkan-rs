//! Runs an `.onnx` through the public API, with no ONNX Runtime in the process.
//!
//! ```text
//! cargo run --release -p onnx-vulkan --example run -- model.onnx [--dim SYM=N] [--fill NAME=V] [--runs N]
//! ```
//!
//! `--dim SYM=N` binds a symbolic dimension (default 1); `--fill NAME=V` fills
//! an integer input with `V` (default 0), because a random length is not a
//! length. Float inputs get a fixed pseudo-random sequence, so two runs of the
//! same command are comparable.
//!
//! `--runs N` runs the same session N times. That is what shows the session is
//! warm: the second run compiles no shader and uploads no weight.

use onnx_vulkan::{Dim, ElementType, HostTensor, KvCache, Session, TensorInfo};
use std::collections::HashMap;
use std::time::Instant;

/// Cache tensors of a decoder: `present.N.x` and the `past_key_values.N.x` it
/// comes back as, paired by name.
fn kv_pairs(session: &Session) -> Vec<(String, String)> {
    let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name.as_str()).collect();
    session
        .outputs()
        .iter()
        .filter_map(|out| {
            let past = format!("past_key_values{}", out.name.strip_prefix("present")?);
            inputs
                .contains(&past.as_str())
                .then_some((out.name.clone(), past))
        })
        .collect()
}

/// `steps` decode steps, twice: once carrying the cache through host memory
/// and once with it resident in VRAM, written in place.
///
/// The stateless loop is the oracle — it is the path validated against the
/// ONNX Runtime CPU EP — and the two are compared at every step, on every
/// output that is not cache. A resident cache that is off by a token, or that
/// loses what an earlier step wrote, shows up here and nowhere else.
fn decode(
    session: &Session,
    steps: usize,
    dims: &HashMap<String, i64>,
    fills: &HashMap<String, i64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let pairs = kv_pairs(session);
    if pairs.is_empty() {
        return Err("--decode: this model has no present.*/past_key_values.* pair".into());
    }
    let logits: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name.clone())
        .filter(|name| !pairs.iter().any(|(present, _)| present == name))
        .collect();
    println!(
        "decoding {steps} steps over {} cache tensors, checking {} output(s)",
        pairs.len(),
        logits.len()
    );

    // the non-cache inputs of step `past`: the dimensions move, the values do not
    let step_inputs = |past: usize| -> Result<Vec<(String, HostTensor)>, String> {
        let mut dims = dims.clone();
        dims.insert("past_sequence_length".into(), past as i64);
        dims.insert("sequence_length".into(), 1);
        dims.insert("total_sequence_length".into(), past as i64 + 1);
        session
            .inputs()
            .iter()
            .filter(|info| !pairs.iter().any(|(_, name)| name == &info.name))
            .map(|info| build(info, &dims, fills).map(|t| (info.name.clone(), t)))
            .collect()
    };

    // the stateless loop needs a cache to start from, and at step 0 that is an
    // empty one: `[batch, kv_heads, 0, head_size]`
    let mut host_cache: Vec<(String, HostTensor)> = pairs
        .iter()
        .map(|(present, past)| {
            let shape = session
                .outputs()
                .iter()
                .find(|o| &o.name == present)
                .and_then(|o| o.shape.clone())
                .ok_or(format!("{present}: unknown shape"))?;
            let fixed = |axis: usize| match shape[axis] {
                Dim::Fixed(n) => n,
                _ => 1,
            };
            Ok((
                past.clone(),
                HostTensor::from_f32(vec![fixed(0), fixed(1), 0, fixed(3)], &[]),
            ))
        })
        .collect::<Result<_, String>>()?;
    let cache = KvCache::new(session, 2048)?;
    for step in 0..steps {
        let shared = step_inputs(step)?;

        let stateless = Instant::now();
        let feed: Vec<(&str, HostTensor)> = shared
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor.clone()))
            .chain(
                host_cache
                    .iter()
                    .map(|(name, tensor)| (name.as_str(), tensor.clone())),
            )
            .collect();
        let run = session.run(feed)?;
        let want: Vec<Vec<f32>> = logits
            .iter()
            .map(|name| Ok(run.get(name)?.to_f32()?))
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;
        host_cache = pairs
            .iter()
            .map(|(present, past)| run.get(present).map(|t| (past.clone(), t)))
            .collect::<Result<_, _>>()?;
        run.finish();
        let stateless = stateless.elapsed();

        let resident = Instant::now();
        let run = session.run_cached(
            shared.iter().map(|(n, t)| (n.as_str(), t.clone())),
            &cache,
        )?;
        let got: Vec<Vec<f32>> = logits
            .iter()
            .map(|name| Ok(run.get(name)?.to_f32()?))
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;
        run.finish();
        let resident = resident.elapsed();

        let mut worst = 0.0f32;
        for (name, (want, got)) in logits.iter().zip(want.iter().zip(&got)) {
            if want.len() != got.len() {
                return Err(format!("{name}: {} values, expected {}", got.len(), want.len()).into());
            }
            worst = want
                .iter()
                .zip(got)
                .fold(worst, |acc, (w, g)| acc.max((w - g).abs()));
        }
        println!(
            "step {step}: cache {} tokens · stateless {stateless:?} · resident {resident:?} \
             · max|Δ| {worst:.3e}",
            cache.len()
        );
        if worst > 1e-3 {
            return Err(format!("step {step}: the resident cache diverged by {worst:.3e}").into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: run <model.onnx> [--dim SYM=N] [--fill NAME=V] [--runs N]")?;

    let (mut dims, mut fills) = (HashMap::new(), HashMap::new());
    let mut runs = 1usize;
    let mut decode_steps = 0usize;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs an argument"))?;
        match flag.as_str() {
            "--runs" => runs = value.parse()?,
            "--decode" => decode_steps = value.parse()?,
            "--dim" | "--fill" => {
                let (name, number) = value
                    .split_once('=')
                    .ok_or(format!("{flag} wants NAME=N"))?;
                let number: i64 = number.parse()?;
                if flag == "--dim" {
                    dims.insert(name.to_string(), number);
                } else {
                    fills.insert(name.to_string(), number);
                }
            }
            other => return Err(format!("unknown option '{other}'").into()),
        }
    }

    let started = Instant::now();
    let session = Session::load(&path)?;
    println!(
        "loaded {path} in {:?} — {} nodes after rewriting",
        started.elapsed(),
        session.node_count()
    );
    for info in session.inputs() {
        println!("  input  {}: {info}", info.name);
    }
    for info in session.outputs() {
        println!("  output {}: {info}", info.name);
    }

    if decode_steps > 0 {
        return decode(&session, decode_steps, &dims, &fills);
    }

    let inputs: Vec<(String, HostTensor)> = session
        .inputs()
        .iter()
        .map(|info| build(info, &dims, &fills).map(|t| (info.name.clone(), t)))
        .collect::<Result<_, _>>()?;

    for i in 0..runs {
        let ran = Instant::now();
        let run = session.run(inputs.iter().map(|(n, t)| (n.as_str(), t.clone())))?;
        for info in session.outputs() {
            let tensor = run.get(&info.name)?;
            let sum: f64 = tensor.to_f32()?.iter().map(|v| *v as f64).sum();
            if i + 1 == runs {
                println!(
                    "  output {}: dtype {} shape {:?} · sum {sum:.6}",
                    info.name, tensor.dtype, tensor.shape
                );
            }
        }
        println!("run {} of {runs} in {:?}", i + 1, ran.elapsed());
        run.finish();
    }
    Ok(())
}

/// Builds one tensor per graph input from the inferred type.
fn build(
    info: &TensorInfo,
    dims: &HashMap<String, i64>,
    fills: &HashMap<String, i64>,
) -> Result<HostTensor, String> {
    let shape: Vec<i64> = info
        .shape
        .as_ref()
        .ok_or(format!("input '{}': unknown shape", info.name))?
        .iter()
        .map(|d| match d {
            Dim::Fixed(n) => Ok(*n),
            // a symbolic dimension is not in the file: the caller knows it
            Dim::Symbol(s) => Ok(dims.get(s).copied().unwrap_or(1)),
            Dim::Unknown => Err(format!("input '{}': unknown dimension", info.name)),
        })
        .collect::<Result<_, _>>()?;
    let count: usize = shape.iter().product::<i64>().max(0) as usize;

    match info.dtype {
        Some(ElementType::Float32) => Ok(HostTensor::from_f32(shape, &pseudo(count))),
        Some(ElementType::Int64) | Some(ElementType::Int32) => {
            let fill = fills.get(&info.name).copied().unwrap_or(0);
            Ok(HostTensor::from_i64(shape, &vec![fill; count]))
        }
        other => Err(format!(
            "input '{}': dtype {other:?} not generable",
            info.name
        )),
    }
}

/// Deterministic sequence: two runs of the same command must give the same
/// checksum, otherwise the comparison says nothing.
fn pseudo(n: usize) -> Vec<f32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 30) as f32) - 1.0
        })
        .collect()
}
