//! Loads an `.onnx` and runs it, with **no ONNX Runtime in the process**.
//!
//! The end-to-end check of Phase 1: the frontend builds the IR *and* the type
//! of every value, the core's `Executor` runs it. Input shapes come from the
//! inference — the only thing that has to be supplied is the value of the
//! symbolic dimensions, which by definition the file does not contain.
//!
//! ```text
//! cargo run --release -p onnx-vulkan-frontend --example run-standalone -- \
//!     models/parakeet-tdt-0.6b-v3-onnx/encoder-model.int8.onnx \
//!     --dim audio_signal_dynamic_axes_1=1 \
//!     --dim audio_signal_dynamic_axes_2=744 \
//!     --dim length_dynamic_axes_1=1 --fill length=744
//! ```
//!
//! `--dim SYM=N` binds a symbolic dimension (default 1); `--fill NAME=V` fills
//! an integer input with `V` (default 0), because a random length is not a
//! length. Float inputs get a fixed pseudo-random sequence.
//! With `RUN_DUMP=<dir>` every output is written as little-endian f32.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT32, INT64};
use onnx_vulkan_core::{Executor, Tensor};
use onnx_vulkan_frontend::{Dim, Model};
use std::collections::HashMap;
use std::time::Instant;
use vk_compute::VkContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("uso: run-standalone <model.onnx> [--dim SYM=N] [--fill NAME=V]")?;

    let (mut dims, mut fills) = (HashMap::new(), HashMap::new());
    while let Some(flag) = args.next() {
        let value = args.next().ok_or(format!("{flag} vuole un argomento"))?;
        let (name, number) = value
            .split_once('=')
            .ok_or(format!("{flag} vuole NOME=N"))?;
        let number: i64 = number.parse()?;
        match flag.as_str() {
            "--dim" => dims.insert(name.to_string(), number),
            "--fill" => fills.insert(name.to_string(), number),
            other => return Err(format!("opzione sconosciuta '{other}'").into()),
        };
    }

    let started = Instant::now();
    let model = onnx_vulkan_frontend::load(&path)?;
    println!(
        "caricato {path} in {:?} — {} nodi, {} initializer",
        started.elapsed(),
        model.graph.nodes.len(),
        model.graph.initializers.len()
    );
    for conflict in model.conflicts.iter().take(3) {
        println!("  ! {conflict}");
    }

    let inputs = build_inputs(&model, &dims, &fills)?;
    for (name, tensor) in &inputs {
        println!(
            "  input {name}: dtype {} shape {:?}",
            tensor.dtype, tensor.shape
        );
    }

    let outputs_wanted = model.graph.outputs.clone();
    let context = VkContext::new()?;
    let built = Instant::now();
    let executor = Executor::new(&context, model.graph)?;
    println!("executor pronto in {:?}", built.elapsed());

    let ran = Instant::now();
    let bound: Vec<(&str, Tensor)> = inputs
        .iter()
        .map(|(name, tensor)| (name.as_str(), Tensor::Host(tensor.clone())))
        .collect();
    let results = executor.run(bound)?;

    let dump = std::env::var("RUN_DUMP").ok();
    for name in &outputs_wanted {
        let tensor = results.host(name)?;
        let values = tensor.to_f32()?;
        let sum: f64 = values.iter().map(|v| *v as f64).sum();
        println!(
            "  output {name}: dtype {} shape {:?} · somma {sum:.6}",
            tensor.dtype, tensor.shape
        );
        if let Some(dir) = &dump {
            let path = std::path::Path::new(dir).join(format!("{}.f32", name.replace('/', "_")));
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(&path, bytes)?;
            println!("    scritto {}", path.display());
        }
    }
    println!("eseguito in {:?}", ran.elapsed());
    results.finish();
    Ok(())
}

/// Builds one tensor per graph input from the inferred type.
fn build_inputs(
    model: &Model,
    dims: &HashMap<String, i64>,
    fills: &HashMap<String, i64>,
) -> Result<Vec<(String, HostTensor)>, String> {
    model
        .graph
        .inputs
        .iter()
        .map(|name| {
            let declared = model.input_type(name).ok_or(format!(
                "input '{name}': tipo sconosciuto, niente da costruire"
            ))?;
            let shape: Vec<i64> = declared
                .shape
                .as_ref()
                .ok_or(format!("input '{name}': forma sconosciuta"))?
                .iter()
                .map(|d| match d {
                    Dim::Fixed(n) => Ok(*n),
                    // a symbolic dimension is not in the file: it is the caller
                    // who knows its value
                    Dim::Symbol(s) => Ok(dims.get(s).copied().unwrap_or(1)),
                    Dim::Unknown => Err(format!("input '{name}': unknown dimension")),
                })
                .collect::<Result<_, _>>()?;
            let count: usize = shape.iter().product::<i64>().max(0) as usize;

            let dtype = declared
                .dtype
                .ok_or(format!("input '{name}': dtype ignoto"))?;
            let tensor = match dtype {
                FLOAT => HostTensor::from_f32(shape, &pseudo(count)),
                INT64 | INT32 => {
                    let fill = fills.get(name).copied().unwrap_or(0);
                    HostTensor::from_i64(shape, &vec![fill; count])
                }
                other => return Err(format!("input '{name}': dtype {other} non generabile")),
            };
            Ok((name.clone(), tensor))
        })
        .collect()
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
