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

use onnx_vulkan::{Dim, ElementType, HostTensor, Session, TensorInfo};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: run <model.onnx> [--dim SYM=N] [--fill NAME=V] [--runs N]")?;

    let (mut dims, mut fills) = (HashMap::new(), HashMap::new());
    let mut runs = 1usize;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} needs an argument"))?;
        match flag.as_str() {
            "--runs" => runs = value.parse()?,
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
