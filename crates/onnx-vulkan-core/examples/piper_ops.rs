//! Small end-to-end exercises for the operators used by Piper graphs.
//!
//! Timings include graph execution and output readback. Reported throughput is
//! based on estimated logical input/output bytes, not measured device traffic;
//! the operation counts are approximate and do not establish application-level
//! performance or acceleration.

use anyhow::{Context, Result, ensure};
use onnx_vulkan_core::host_ops::{BOOL, FLOAT, HostTensor};
use onnx_vulkan_core::{AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute};
use std::hint::black_box;
use std::time::{Duration, Instant};
use vk_compute::VkContext;

const WARMUP: usize = 2;
const DEFAULT_ITERS: usize = 5;

struct Case {
    label: String,
    graph: GraphIr,
    input: HostTensor,
    input_name: &'static str,
    output_name: &'static str,
    logical_bytes: u64,
    estimated_ops: u64,
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_owned(),
        opset: 21,
        name: format!("piper_ops_{op}"),
        inputs: inputs.iter().map(|value| (*value).to_owned()).collect(),
        outputs: outputs.iter().map(|value| (*value).to_owned()).collect(),
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_owned(), value.clone()))
            .collect(),
    }
}

fn graph(nodes: Vec<NodeIr>, input: &str, output: &str) -> GraphIr {
    GraphIr {
        nodes,
        inputs: vec![input.to_owned()],
        outputs: vec![output.to_owned()],
        ..Default::default()
    }
}

fn parse_iters() -> Result<usize> {
    let mut args = std::env::args().skip(1);
    let mut iters = DEFAULT_ITERS;
    while let Some(arg) = args.next() {
        ensure!(arg == "--iters", "usage: piper_ops [--iters N]");
        let value = args.next().context("--iters requires a positive integer")?;
        iters = value
            .parse()
            .context("--iters must be a positive integer")?;
        ensure!(iters > 0, "--iters must be greater than zero");
    }
    Ok(iters)
}

fn float_unary_case(op: &str) -> Case {
    let count = 65_536usize;
    let values: Vec<f32> = (0..count)
        .map(|index| ((index % 1009) as f32 - 504.0) / 73.0)
        .collect();
    let input = HostTensor::from_f32(vec![count as i64], &values);
    let graph = graph(vec![node(op, &["x"], &["y"], &[])], "x", "y");
    Case {
        label: format!("{op} FLOAT[{count}]"),
        graph,
        input,
        input_name: "x",
        output_name: "y",
        logical_bytes: (count * 4 * 2) as u64,
        // Ceil is counted as one primitive operation. Softplus counts its
        // simple scalar arithmetic only; exp/log special functions are excluded.
        estimated_ops: (count * if op == "Softplus" { 5 } else { 1 }) as u64,
    }
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for count in [256usize, 4096, 65_536] {
        let mask: Vec<u8> = (0..count)
            .map(|index| u8::from(index % 8 == 1 || index % 31 == 0))
            .collect();
        let nnz = mask.iter().filter(|&&value| value != 0).count();
        let input = HostTensor::new(BOOL, vec![count as i64], mask);
        let graph = graph(vec![node("NonZero", &["x"], &["y"], &[])], "x", "y");
        cases.push(Case {
            label: format!("NonZero BOOL[{count}]"),
            graph,
            input,
            input_name: "x",
            output_name: "y",
            // Logical input bytes plus the exact rank-one int64 coordinate output.
            logical_bytes: (count + nnz * 8) as u64,
            // One predicate comparison per input element; compaction/indexing
            // work and the small count-buffer traffic are not included.
            estimated_ops: count as u64,
        });
    }

    let count = 65_536usize;
    let random_input = HostTensor::from_f32(vec![count as i64], &vec![0.0; count]);
    let random_graph = graph(
        vec![node(
            "RandomNormalLike",
            &["x"],
            &["y"],
            &[
                ("dtype", AttrValue::Int(i64::from(FLOAT))),
                ("seed", AttrValue::Float(42.0)),
            ],
        )],
        "x",
        "y",
    );
    cases.push(Case {
        label: format!("RandomNormalLike FLOAT[{count}]"),
        graph: random_graph,
        input: random_input,
        input_name: "x",
        output_name: "y",
        // The source tensor supplies shape only and is not uploaded by this op.
        logical_bytes: (count * 4) as u64,
        // Arithmetic-equivalent estimate excludes transcendental functions.
        estimated_ops: (count * 16) as u64,
    });

    cases.push(float_unary_case("Ceil"));
    cases.push(float_unary_case("Softplus"));

    let count = 128usize * 256;
    let values: Vec<f32> = (0..count)
        .map(|index| ((index % 257) as f32 - 128.0) / 31.0)
        .collect();
    let reduction_graph = graph(
        vec![node(
            "ReduceSum",
            &["x"],
            &["y"],
            &[
                ("axes", AttrValue::Ints(vec![1, 2])),
                ("keepdims", AttrValue::Int(0)),
            ],
        )],
        "x",
        "y",
    );
    cases.push(Case {
        label: "ReduceSum FLOAT[1,128,256] axes=[1,2]".to_owned(),
        graph: reduction_graph,
        input: HostTensor::from_f32(vec![1, 128, 256], &values),
        input_name: "x",
        output_name: "y",
        logical_bytes: ((count + 1) * 4) as u64,
        estimated_ops: (count - 1) as u64,
    });
    cases
}

fn benchmark_case(cache: &KernelCache<'_>, case: &Case, iters: usize) -> Result<Duration> {
    let mut elapsed = Duration::ZERO;
    for iteration in 0..(WARMUP + iters) {
        let mut env = ExecutionEnv::new(cache, &case.graph.initializers);
        env.set(case.input_name, Tensor::Host(case.input.clone()));
        let start = (iteration >= WARMUP).then(Instant::now);
        execute(&case.graph, &mut env).with_context(|| format!("execute {}", case.label))?;
        let output = env
            .host(case.output_name)
            .with_context(|| format!("read back {}", case.label))?;
        black_box(output);
        if let Some(start) = start {
            elapsed += start.elapsed();
        }
        env.finish();
    }
    Ok(elapsed / iters as u32)
}

fn main() -> Result<()> {
    let iters = parse_iters()?;
    let context = VkContext::new().context("create Vulkan context")?;
    let cache = KernelCache::new(&context);
    println!(
        "Piper operator exercise: {WARMUP} warmups + {iters} measured runs; execute and output readback timed"
    );
    println!(
        "Throughput uses estimated logical input/output bytes; TFLOP/s uses approximate operation counts. Neither is device-traffic or end-to-end acceleration proof."
    );
    for case in cases() {
        let average = benchmark_case(&cache, &case, iters)?;
        let seconds = average.as_secs_f64();
        let gb_per_second = case.logical_bytes as f64 / seconds / 1.0e9;
        let tflops = case.estimated_ops as f64 / seconds / 1.0e12;
        println!(
            "{:<42} {:8.3} ms  {:7.4} logical GB/s  {:9.6} estimated TFLOP/s  ({} bytes, ~{} ops)",
            case.label,
            seconds * 1_000.0,
            gb_per_second,
            tflops,
            case.logical_bytes,
            case.estimated_ops
        );
    }
    Ok(())
}
