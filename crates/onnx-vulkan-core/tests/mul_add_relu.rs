//! `Mul(x, scale) → Add(·, bias) → Relu` by constants, fused into `MulAddRelu`
//! at load time, must answer **bit for bit** like the three nodes it replaces.
//!
//! The reference is the same graph with the scaled value declared as a graph
//! output: an intermediate that escapes cannot be fused, so the rewrite leaves
//! that graph alone and it runs the original three kernels on the same device.
//! The pipelines compiled tell which path the fused node took.

use onnx_vulkan_core::host_ops::FLOAT;
use onnx_vulkan_core::{Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use std::collections::HashMap;
use vk_compute::VkContext;

fn node(op: &str, inputs: &[&str], output: &str) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 17,
        name: format!("{op}_{output}"),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec![output.to_string()],
        attrs: HashMap::new(),
    }
}

/// Deterministic values in `[-scale, scale)` (a 64-bit LCG's high bits).
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn constant(shape: &[i64], seed: u64) -> InitializerIr {
    let count = shape.iter().product::<i64>() as usize;
    InitializerIr {
        dtype: FLOAT,
        shape: shape.to_vec(),
        data: values(count, seed, 1.5)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    }
}

struct Case<'a> {
    x: &'a [i64],
    scale: &'a [i64],
    bias: &'a [i64],
    /// `Mul(scale, x)` instead of `Mul(x, scale)`
    scale_first: bool,
}

impl Case<'_> {
    fn graph(&self) -> GraphIr {
        let mul = if self.scale_first {
            ["scale", "x"]
        } else {
            ["x", "scale"]
        };
        GraphIr {
            nodes: vec![
                node("Mul", &mul, "scaled"),
                node("Add", &["bias", "scaled"], "shifted"),
                node("Relu", &["shifted"], "y"),
            ],
            initializers: HashMap::from([
                ("scale".into(), constant(self.scale, 2)),
                ("bias".into(), constant(self.bias, 3)),
            ]),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    /// `y` through `Executor`, the ops left after its load-time rewrite, and
    /// whether the fused pipeline was compiled.
    fn run(&self, context: &VkContext, graph: GraphIr) -> (Vec<u32>, Vec<String>, bool) {
        let numel = self.x.iter().product::<i64>() as usize;
        let x = HostTensor::from_f32(self.x.to_vec(), &values(numel, 1, 3.0));
        let executor = Executor::new(context, graph).expect("executor");
        let ops = executor
            .graph()
            .nodes
            .iter()
            .map(|n| n.op.clone())
            .collect();
        let outputs = executor.run(vec![("x", Tensor::Host(x))]).expect("run");
        let y = outputs.host("y").expect("output");
        assert_eq!(y.shape, self.x);
        let bits = y.to_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        outputs.finish();
        let fused_kernel = executor
            .cache()
            .pipeline_keys()
            .iter()
            .any(|key| key.operation() == "MulAddRelu");
        (bits, ops, fused_kernel)
    }

    /// Fused against unfused, bit for bit; `kernel` says whether the fused
    /// kernel or the three-step fallback must have run.
    fn check(&self, kernel: bool) {
        let context = VkContext::new().expect("Vulkan context");

        let mut reference = self.graph();
        reference.outputs.push("scaled".into());
        let (expected, ops, fused_kernel) = self.run(&context, reference);
        assert_eq!(ops, ["Mul", "Add", "Relu"]);
        assert!(!fused_kernel);

        let (got, ops, fused_kernel) = self.run(&context, self.graph());
        assert_eq!(ops, ["MulAddRelu"]);
        assert_eq!(got.len(), expected.len());
        let differing = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
        assert_eq!(differing, 0, "{differing} of {} elements differ", got.len());
        // about half the sums are negative: the Relu was applied, both ways
        let zeros = got.iter().filter(|&&bits| bits == 0).count();
        assert!(zeros > got.len() / 4, "{zeros} of {} zero", got.len());
        assert_eq!(fused_kernel, kernel);
    }
}

#[test]
fn per_channel_affine_after_a_group_norm() {
    // SAM 3's pixel decoder, fewer channels and a smaller map
    Case {
        x: &[1, 64, 20, 20],
        scale: &[64, 1, 1],
        bias: &[64, 1, 1],
        scale_first: true,
    }
    .check(true);
}

#[test]
fn batch_odd_plane_and_one_bias_for_the_tensor() {
    Case {
        x: &[2, 48, 9, 7],
        scale: &[1, 48, 1, 1],
        bias: &[],
        scale_first: false,
    }
    .check(true);
}

#[test]
fn rank_two_with_one_value_per_feature() {
    Case {
        x: &[37, 48],
        scale: &[48],
        bias: &[1, 48],
        scale_first: false,
    }
    .check(true);
}

#[test]
fn scale_along_the_width_falls_back_to_three_steps() {
    Case {
        x: &[1, 64, 20, 20],
        scale: &[1, 1, 1, 20],
        bias: &[64, 1, 1],
        scale_first: true,
    }
    .check(false);
}
