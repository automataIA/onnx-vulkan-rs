//! `ConvInteger → Cast → Mul → Add` fused into `ConvIntegerScaleBias` at load
//! time must answer **bit for bit** like the four nodes it replaces.
//!
//! The reference is the same graph with the dequantized convolution declared
//! as a graph output: an intermediate that escapes cannot be fused, so the
//! rewrite leaves that graph alone and it runs the original four kernels on the
//! same device. The pipelines compiled tell which path the fused node took.

use onnx_vulkan_core::host_ops::{FLOAT, INT8, UINT8};
use onnx_vulkan_core::shaders::matmul_integer::coop_variant;
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
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

fn pseudo(count: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Deterministic values in `[-scale, scale)`.
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    pseudo(count * 3, seed)
        .chunks_exact(3)
        .map(|c| {
            let bits = u32::from(c[0]) | u32::from(c[1]) << 8 | u32::from(c[2]) << 16;
            (bits as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

fn initializer(dtype: i32, shape: Vec<i64>, data: Vec<u8>) -> InitializerIr {
    InitializerIr { dtype, shape, data }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// One convolution of a dynamically quantized model: `x [1, C_in, H, W]`,
/// quantized in the graph by `DynamicQuantizeLinear` (or fed as int8), times a
/// constant weight, dequantized by a scale and biased.
struct Layer {
    c_in: i64,
    c_out: i64,
    hw: [i64; 2],
    kernel: i64,
    stride: i64,
    pad: i64,
    weight_signed: bool,
    /// int8 X fed from the host instead of `DynamicQuantizeLinear`'s uint8
    input_signed: bool,
    /// `None`: the scalar product of the two scales, as the exporters write it
    scale_shape: Option<Vec<i64>>,
    bias_shape: Vec<i64>,
}

impl Layer {
    fn graph(&self) -> GraphIr {
        let (w_dtype, w_zp) = if self.weight_signed {
            (INT8, (-7i8) as u8)
        } else {
            (UINT8, 121)
        };
        let k = self.kernel;
        let mut conv = node("ConvInteger", &["a", "w", "a_zp", "w_zp"], "acc");
        conv.attrs.extend([
            ("kernel_shape".to_string(), AttrValue::Ints(vec![k, k])),
            ("strides".to_string(), AttrValue::Ints(vec![self.stride; 2])),
            ("pads".to_string(), AttrValue::Ints(vec![self.pad; 4])),
        ]);
        let mut cast = node("Cast", &["acc"], "accf");
        cast.attrs
            .insert("to".into(), AttrValue::Int(i64::from(FLOAT)));
        let mut nodes = vec![
            conv,
            cast,
            node("Mul", &["accf", "scale"], "scaled"),
            node("Add", &["scaled", "bias"], "y"),
        ];
        let weights = (self.c_out * self.c_in * k * k) as usize;
        let biases = self.bias_shape.iter().product::<i64>() as usize;
        let mut initializers = HashMap::from([
            (
                "w".into(),
                initializer(
                    w_dtype,
                    vec![self.c_out, self.c_in, k, k],
                    pseudo(weights, 2),
                ),
            ),
            ("w_zp".into(), initializer(w_dtype, vec![], vec![w_zp])),
            (
                "bias".into(),
                initializer(
                    FLOAT,
                    self.bias_shape.clone(),
                    f32_bytes(&values(biases, 3, 0.5)),
                ),
            ),
        ]);
        let mut inputs = Vec::new();
        if self.input_signed {
            initializers.insert("a_zp".into(), initializer(INT8, vec![], vec![0xfb]));
            inputs.push("a".to_string());
        } else {
            nodes.insert(
                0,
                NodeIr {
                    outputs: vec!["a".into(), "a_dq_scale".into(), "a_zp".into()],
                    ..node("DynamicQuantizeLinear", &["x"], "a")
                },
            );
            inputs.push("x".to_string());
        }
        match &self.scale_shape {
            None => {
                nodes.insert(0, node("Mul", &["a_scale", "w_scale"], "scale"));
                initializers.insert(
                    "w_scale".into(),
                    initializer(FLOAT, vec![], f32_bytes(&[3.1e-3])),
                );
                inputs.push("a_scale".into());
            }
            Some(shape) => {
                let count = shape.iter().product::<i64>() as usize;
                let scale: Vec<f32> = values(count, 4, 1e-4)
                    .iter()
                    .map(|v| v.abs() + 1e-5)
                    .collect();
                initializers.insert(
                    "scale".into(),
                    initializer(FLOAT, shape.clone(), f32_bytes(&scale)),
                );
            }
        }
        GraphIr {
            nodes,
            initializers,
            inputs,
            outputs: vec!["y".into()],
            ..Default::default()
        }
    }

    /// `y` through `Executor`, the ops left after its load-time rewrite, and
    /// whether the fused convolution pipeline was compiled.
    fn run(&self, context: &VkContext, graph: GraphIr) -> (Vec<u32>, Vec<String>, bool) {
        let shape = vec![1, self.c_in, self.hw[0], self.hw[1]];
        let numel = shape.iter().product::<i64>() as usize;
        let a = if self.input_signed {
            ("a", HostTensor::new(INT8, shape, pseudo(numel, 1)))
        } else {
            ("x", HostTensor::from_f32(shape, &values(numel, 1, 3.0)))
        };
        let executor = Executor::new(context, graph).expect("executor");
        let ops = executor
            .graph()
            .nodes
            .iter()
            .map(|n| n.op.clone())
            .collect();
        let mut feeds = vec![(a.0, Tensor::Host(a.1))];
        if self.scale_shape.is_none() {
            feeds.push((
                "a_scale",
                Tensor::Host(HostTensor::from_f32(vec![], &[0.0123])),
            ));
        }
        let outputs = executor.run(feeds).expect("run");
        let y = outputs.host("y").expect("output");
        assert_eq!(y.shape[..2], [1, self.c_out]);
        let bits = y.to_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        outputs.finish();
        let fused_kernel = executor
            .cache()
            .pipeline_keys()
            .iter()
            .any(|key| key.operation() == "ConvInteger_coop_epilogue");
        (bits, ops, fused_kernel)
    }

    /// Fused against unfused, bit for bit. `kernel` says whether the epilogue
    /// must run in the cooperative kernel, where the device has one, or the
    /// four-step fallback must.
    fn check(&self, kernel: bool) {
        let context = VkContext::new().expect("Vulkan context");

        let mut reference = self.graph();
        reference.outputs.push("accf".into());
        let (expected, ops, fused_kernel) = self.run(&context, reference);
        assert!(ops.iter().any(|op| op == "ConvInteger"));
        assert!(!fused_kernel);

        let (got, ops, fused_kernel) = self.run(&context, self.graph());
        assert!(
            ops.iter().any(|op| op == "ConvIntegerScaleBias"),
            "not fused: {ops:?}"
        );
        assert_eq!(got.len(), expected.len());
        let differing = got.iter().zip(&expected).filter(|(g, e)| g != e).count();
        assert_eq!(differing, 0, "{differing} of {} elements differ", got.len());
        assert_eq!(fused_kernel, kernel && coop_variant(&context).is_some());
    }
}

#[test]
fn padded_3x3_with_a_bias_per_channel() {
    // SAM 3's pixel decoder convolution, fewer channels and a smaller image:
    // HWC im2col, cooperative product, epilogue per row
    Layer {
        c_in: 64,
        c_out: 40,
        hw: [20, 20],
        kernel: 3,
        stride: 1,
        pad: 1,
        weight_signed: true,
        input_signed: false,
        scale_shape: None,
        bias_shape: vec![1, 40, 1, 1],
    }
    .check(true);
}

#[test]
fn strided_3x3_on_signed_input_with_a_scale_per_channel() {
    // a 15×15 plane is off word boundaries: the CHW im2col feeds the product
    Layer {
        c_in: 32,
        c_out: 24,
        hw: [15, 15],
        kernel: 3,
        stride: 2,
        pad: 1,
        weight_signed: false,
        input_signed: true,
        scale_shape: Some(vec![24, 1, 1]),
        bias_shape: vec![24, 1, 1],
    }
    .check(true);
}

#[test]
fn pointwise_with_a_bias_per_channel() {
    Layer {
        c_in: 64,
        c_out: 48,
        hw: [24, 24],
        kernel: 1,
        stride: 1,
        pad: 0,
        weight_signed: true,
        input_signed: false,
        scale_shape: None,
        bias_shape: vec![1, 48, 1, 1],
    }
    .check(true);
}

#[test]
fn bias_along_the_width_falls_back_to_four_steps() {
    Layer {
        c_in: 64,
        c_out: 32,
        hw: [20, 20],
        kernel: 3,
        stride: 1,
        pad: 1,
        weight_signed: true,
        input_signed: false,
        scale_shape: None,
        bias_shape: vec![1, 1, 1, 20],
    }
    .check(false);
}

#[test]
fn one_output_channel_falls_back_to_four_steps() {
    // SAM 3's semantic projection: below the cooperative tile
    Layer {
        c_in: 64,
        c_out: 1,
        hw: [24, 24],
        kernel: 1,
        stride: 1,
        pad: 0,
        weight_signed: true,
        input_signed: false,
        scale_shape: None,
        bias_shape: vec![1, 1, 1, 1],
    }
    .check(false);
}

#[test]
fn im2col_past_the_slice_budget_runs_in_slices() {
    // 17.8 MB of HWC im2col: two slices of the 16 MiB budget, each one written
    // into its columns of the output
    Layer {
        c_in: 256,
        c_out: 32,
        hw: [88, 88],
        kernel: 3,
        stride: 1,
        pad: 1,
        weight_signed: true,
        input_signed: false,
        scale_shape: None,
        bias_shape: vec![1, 32, 1, 1],
    }
    .check(true);
}

#[test]
fn a_short_last_slice_moves_back_to_a_whole_tile() {
    // 130 pixels of CHW im2col (a 10×13 plane is off word boundaries), 26 MB:
    // slices [0, 128) and [114, 130), the overlap stored twice
    Layer {
        c_in: 4096,
        c_out: 16,
        hw: [10, 13],
        kernel: 7,
        stride: 1,
        pad: 3,
        weight_signed: false,
        input_signed: false,
        scale_shape: Some(vec![16, 1, 1]),
        bias_shape: vec![16, 1, 1],
    }
    .check(true);
}
