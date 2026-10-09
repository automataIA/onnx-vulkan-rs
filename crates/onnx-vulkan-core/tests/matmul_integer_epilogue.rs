//! `MatMulInteger → Cast → Mul → Add` fused into `MatMulIntegerScaleBias` at
//! load time must answer **bit for bit** like the four nodes it replaces.
//!
//! The reference is the same graph with the dequantized product declared as a
//! graph output: an intermediate that escapes cannot be fused, so the rewrite
//! leaves that graph alone and it runs the original four kernels on the same
//! device. Bit equality, not a tolerance, is the contract — the epilogue is
//! written to round exactly as the separate `Cast`, `Mul` and `Add`, and a
//! `Relu` after them, fused too, as the unary kernel. A `DynamicQuantizeLinear`
//! of the result, whose range the cooperative kernel folds while it stores,
//! must quantize to the same bytes, scale and zero point as its own min/max
//! pass over the unfused result.

use onnx_vulkan_core::host_ops::{FLOAT, INT8, UINT8};
use onnx_vulkan_core::shaders::matmul_integer::{
    ANY_K_EPILOGUE_KEY, SMALL_EPILOGUE_KEY, coop_variant, coop_variant_for, small_applies,
};
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

/// One linear layer of a dynamically quantized model: `a [.., M, K]` u8 times
/// a constant `w [K, N]`, dequantized by `a_scale · w_scale` and biased.
/// `scale` names either that product or an initializer of `scale_shape`.
struct Layer {
    a_shape: Vec<i64>,
    n: usize,
    weight_signed: bool,
    /// `None`: the scalar product of the two scales, as the exporters write it
    scale_shape: Option<Vec<i64>>,
    /// A quantized inside the graph by `DynamicQuantizeLinear`, so it is a
    /// device tensor (as in SAM 3's detector), instead of fed from the host
    quantized_in_graph: bool,
    /// a `Relu` after the `Add`, which the rewrite folds into the epilogue
    relu: bool,
    /// `y` quantized again by `DynamicQuantizeLinear`, as SAM 3's detector
    /// does after every feed-forward layer
    quantized_after: bool,
    /// with `scale_shape: None`, one weight scale per column (a per-channel
    /// quantized weight, as SAM 3's text and vision encoders have) instead of
    /// one for the tensor
    weight_scale_per_column: bool,
}

impl Layer {
    fn graph(&self) -> GraphIr {
        let k = *self.a_shape.last().unwrap();
        let n = self.n as i64;
        let (w_dtype, w_zp) = if self.weight_signed {
            (INT8, (-7i8) as u8)
        } else {
            (UINT8, 121)
        };
        let mut cast = node("Cast", &["acc"], "accf");
        cast.attrs
            .insert("to".into(), AttrValue::Int(i64::from(FLOAT)));
        let mut nodes = vec![
            node("MatMulInteger", &["a", "w", "a_zp", "w_zp"], "acc"),
            cast,
            node("Mul", &["accf", "scale"], "scaled"),
            node(
                "Add",
                &["bias", "scaled"],
                if self.relu { "biased" } else { "y" },
            ),
        ];
        if self.relu {
            nodes.push(node("Relu", &["biased"], "y"));
        }
        let mut outputs = vec!["y".to_string()];
        if self.quantized_after {
            let quantized = ["q", "q_scale", "q_zp"].map(String::from);
            nodes.push(NodeIr {
                outputs: quantized.to_vec(),
                ..node("DynamicQuantizeLinear", &["y"], "q")
            });
            outputs.extend(quantized);
        }
        let mut initializers = HashMap::from([
            (
                "w".into(),
                initializer(w_dtype, vec![k, n], pseudo((k * n) as usize, 2)),
            ),
            ("w_zp".into(), initializer(w_dtype, vec![], vec![w_zp])),
            (
                "bias".into(),
                initializer(FLOAT, vec![n], f32_bytes(&values(self.n, 3, 0.5))),
            ),
        ]);
        let mut inputs = Vec::new();
        if self.quantized_in_graph {
            nodes.insert(
                0,
                NodeIr {
                    outputs: vec!["a".into(), "a_dq_scale".into(), "a_zp".into()],
                    ..node("DynamicQuantizeLinear", &["x"], "a")
                },
            );
            inputs.push("x".to_string());
        } else {
            initializers.insert("a_zp".into(), initializer(UINT8, vec![], vec![131]));
            inputs.push("a".to_string());
        }
        match &self.scale_shape {
            None => {
                // the scale of A where the graph quantizes it, as in SAM 3,
                // so the product is a device value
                let (at, a_scale) = if self.quantized_in_graph {
                    (1, "a_dq_scale")
                } else {
                    (0, "a_scale")
                };
                nodes.insert(at, node("Mul", &[a_scale, "w_scale"], "scale"));
                let w_scale = if self.weight_scale_per_column {
                    let scale: Vec<f32> = values(self.n, 5, 4e-3)
                        .iter()
                        .map(|v| v.abs() + 1e-4)
                        .collect();
                    initializer(FLOAT, vec![n], f32_bytes(&scale))
                } else {
                    initializer(FLOAT, vec![], f32_bytes(&[3.1e-3]))
                };
                initializers.insert("w_scale".into(), w_scale);
                if !self.quantized_in_graph {
                    inputs.push("a_scale".into());
                }
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
            outputs,
            ..Default::default()
        }
    }

    /// `y` through `Executor`, the ops left after its load-time rewrite, and
    /// the epilogue pipelines it compiled; with `quantized_after`, the bytes of
    /// the quantization too.
    fn run(&self, context: &VkContext, graph: GraphIr) -> Run {
        let numel = self.a_shape.iter().product::<i64>() as usize;
        let a = if self.quantized_in_graph {
            let x = values(numel, 1, 3.0);
            ("x", HostTensor::from_f32(self.a_shape.clone(), &x))
        } else {
            (
                "a",
                HostTensor::new(UINT8, self.a_shape.clone(), pseudo(numel, 1)),
            )
        };
        let executor = Executor::new(context, graph).expect("executor");
        let ops = executor
            .graph()
            .nodes
            .iter()
            .map(|n| n.op.clone())
            .collect();
        let folds_range = executor.graph().nodes.iter().any(|n| {
            n.op == "MatMulIntegerScaleBias" && n.attrs.get("range") == Some(&AttrValue::Int(1))
        });
        let mut feeds = vec![(a.0, Tensor::Host(a.1))];
        if self.scale_shape.is_none() && !self.quantized_in_graph {
            feeds.push((
                "a_scale",
                Tensor::Host(HostTensor::from_f32(vec![], &[0.0123])),
            ));
        }
        let outputs = executor.run(feeds).expect("run");
        let y = outputs.host("y").expect("output");
        let mut shape = self.a_shape.clone();
        *shape.last_mut().unwrap() = self.n as i64;
        assert_eq!(y.shape, shape);
        let bits = y.to_f32().unwrap().iter().map(|v| v.to_bits()).collect();
        let mut quantized = Vec::new();
        if self.quantized_after {
            for name in ["q", "q_scale", "q_zp"] {
                quantized.extend(&outputs.host(name).expect("quantized output").data);
            }
        }
        outputs.finish();
        let keys = executor.cache().pipeline_keys();
        let operations = keys.iter().map(|key| key.operation());
        Run {
            bits,
            ops,
            epilogues: operations
                .clone()
                .filter(|op| op.ends_with("_epilogue"))
                .collect(),
            quantized,
            folds_range,
            range_pass: operations.clone().any(|op| op == "DQL_partial"),
            multiplies: operations.clone().any(|op| op == MUL_KERNEL),
        }
    }

    /// Fused against unfused, bit for bit. `kernel` says whether the epilogue
    /// shader must run (the any-K one for a K not a multiple of four, the
    /// one-output-per-thread one for fewer than 16 rows or columns, else the
    /// cooperative matrix one where the device has a variant for this shape),
    /// or the four-step fallback must.
    fn check(&self, kernel: bool) {
        let context = VkContext::new().expect("Vulkan context");

        let mut reference = self.graph();
        reference.outputs.push("accf".into());
        let expected = self.run(&context, reference);
        assert!(expected.ops.iter().any(|op| op == "MatMulInteger"));
        assert!(expected.epilogues.is_empty());

        let got = self.run(&context, self.graph());
        assert!(
            got.ops.iter().any(|op| op == "MatMulIntegerScaleBias"),
            "not fused: {:?}",
            got.ops
        );
        // with a constant scale the reference's Mul → Add → Relu becomes a
        // `MulAddRelu`, exact against the three kernels (tests/mul_add_relu.rs)
        let reference_relu = |op: &String| op == "Relu" || op == "MulAddRelu";
        assert_eq!(expected.ops.iter().any(reference_relu), self.relu);
        assert!(!got.ops.iter().any(|op| op == "Relu"), "{:?}", got.ops);
        // the product of the two scales is formed by the fused node
        if self.scale_shape.is_none() {
            assert!(expected.ops.iter().any(|op| op == "Mul"));
            assert!(!got.ops.iter().any(|op| op == "Mul"), "{:?}", got.ops);
        }
        assert_eq!(got.bits.len(), expected.bits.len());
        let differing = got
            .bits
            .iter()
            .zip(&expected.bits)
            .filter(|(g, e)| g != e)
            .count();
        assert_eq!(
            differing,
            0,
            "{differing} of {} elements differ",
            got.bits.len()
        );
        if self.relu {
            // the Relu was applied, both ways
            let zeros = got.bits.iter().filter(|&&bits| bits == 0).count();
            assert!(
                zeros > got.bits.len() / 8,
                "{zeros} of {} zero",
                got.bits.len()
            );
        }

        // which path ran is visible only in the pipelines it compiled
        let k = *self.a_shape.last().unwrap();
        let m: i64 = self.a_shape[..self.a_shape.len() - 1].iter().product();
        let want: Vec<&str> = match coop_variant_for(&context, m as usize, self.n) {
            _ if kernel && k % 4 != 0 => vec![ANY_K_EPILOGUE_KEY],
            _ if kernel && small_applies(m as usize, k as usize, self.n) => {
                vec![SMALL_EPILOGUE_KEY]
            }
            Some(v) if kernel => vec![v.epilogue_key],
            _ => Vec::new(),
        };
        assert_eq!(got.epilogues, want);

        assert_eq!(got.quantized, expected.quantized, "quantization differs");
        assert_eq!(got.folds_range, self.quantized_after);
        if self.quantized_after {
            // the cooperative and small kernels fold the range; after any
            // other path the quantization makes its own pass
            assert!(expected.range_pass);
            let coop = !want.is_empty() && want != [ANY_K_EPILOGUE_KEY];
            assert_eq!(got.range_pass, !coop || self.quantized_in_graph);
        }
        if self.scale_shape.is_none() && self.quantized_in_graph {
            // a device scale product: the cooperative and small kernels form
            // it, any other path runs the `Mul` on the device first
            let coop = !want.is_empty() && want != [ANY_K_EPILOGUE_KEY];
            assert!(expected.multiplies);
            assert_eq!(got.multiplies, !coop);
        }
    }
}

struct Run {
    bits: Vec<u32>,
    ops: Vec<String>,
    epilogues: Vec<&'static str>,
    /// bytes of `q`, `q_scale` and `q_zp`, with `quantized_after`
    quantized: Vec<u8>,
    /// the rewrite asked the fused node for the range
    folds_range: bool,
    /// the quantization's own min/max pass was compiled
    range_pass: bool,
    /// the elementwise `Mul` kernel was compiled
    multiplies: bool,
}

/// The pipeline key of the elementwise `Mul`: its expression.
const MUL_KERNEL: &str = "a[off_a] * b[off_b]";

#[test]
fn epilogue_matches_the_four_kernels_on_a_single_block() {
    // partial tiles on both axes, a single block
    Layer {
        a_shape: vec![2, 50, 64],
        n: 200,
        weight_signed: false,
        scale_shape: None,
        quantized_in_graph: false,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn epilogue_matches_the_four_kernels_on_several_blocks() {
    // several blocks, the last band clamped on both axes, K = 96 with a
    // zero-filled tail chunk in the shared-memory variant
    Layer {
        a_shape: vec![700, 96],
        n: 330,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: false,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn epilogue_on_a_grid_for_the_largest_block() {
    // enough 128×128 blocks that the largest build is picked (the cases above
    // take the 64×64 one), partial on both axes
    let context = VkContext::new().expect("Vulkan context");
    assert_eq!(
        coop_variant_for(&context, 1100, 1040).map(|v| v.key),
        coop_variant(&context).map(|v| v.key)
    );
    Layer {
        a_shape: vec![1100, 64],
        n: 1040,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: false,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn epilogue_with_one_scale_per_column() {
    Layer {
        a_shape: vec![1, 128, 32],
        n: 48,
        weight_signed: false,
        scale_shape: Some(vec![1, 48]),
        quantized_in_graph: false,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn scale_over_rows_falls_back_to_four_steps() {
    // one scale per row broadcasts along M: not what the epilogue reads
    Layer {
        a_shape: vec![64, 32],
        n: 32,
        weight_signed: false,
        scale_shape: Some(vec![64, 1]),
        quantized_in_graph: false,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(false);
}

#[test]
fn any_k_epilogue_matches_the_four_kernels_on_the_box_projection() {
    // SAM 3's box relative-position projection at a tenth of the queries:
    // K = 2, one scale for the tensor
    Layer {
        a_shape: vec![1, 20, 72, 2],
        n: 256,
        weight_signed: false,
        scale_shape: None,
        quantized_in_graph: true,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn any_k_epilogue_with_one_scale_per_column_and_odd_k() {
    Layer {
        a_shape: vec![37, 7],
        n: 50,
        weight_signed: true,
        scale_shape: Some(vec![50]),
        quantized_in_graph: true,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn any_k_scale_over_rows_falls_back_to_four_steps() {
    Layer {
        a_shape: vec![40, 3],
        n: 24,
        weight_signed: false,
        scale_shape: Some(vec![40, 1]),
        quantized_in_graph: true,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(false);
}

#[test]
fn relu_folded_into_the_epilogue() {
    // SAM 3's detector feed-forward: a Relu after the dequantization
    Layer {
        a_shape: vec![700, 96],
        n: 330,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: true,
        relu: true,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn relu_folded_into_the_any_k_epilogue() {
    Layer {
        a_shape: vec![37, 7],
        n: 50,
        weight_signed: true,
        scale_shape: Some(vec![50]),
        quantized_in_graph: true,
        relu: true,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn relu_folded_and_scale_over_rows_falls_back_to_five_steps() {
    Layer {
        a_shape: vec![64, 32],
        n: 32,
        weight_signed: false,
        scale_shape: Some(vec![64, 1]),
        quantized_in_graph: false,
        relu: true,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(false);
}

#[test]
fn quantization_reads_the_range_the_epilogue_folded() {
    // SAM 3's detector feed-forward: Relu, then quantized for the next layer;
    // several blocks, so the range gathers atomics from many workgroups
    Layer {
        a_shape: vec![700, 96],
        n: 330,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: false,
        relu: true,
        quantized_after: true,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn folded_range_with_negative_values_on_the_largest_block() {
    // no Relu: the minimum is negative too
    Layer {
        a_shape: vec![1100, 64],
        n: 1040,
        weight_signed: false,
        scale_shape: Some(vec![1040]),
        quantized_in_graph: false,
        relu: false,
        quantized_after: true,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn folded_range_on_a_single_block() {
    Layer {
        a_shape: vec![2, 50, 64],
        n: 200,
        weight_signed: false,
        scale_shape: None,
        quantized_in_graph: false,
        relu: false,
        quantized_after: true,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn any_k_epilogue_leaves_the_range_to_the_quantization() {
    // the any-K kernel takes an A already on the device
    Layer {
        a_shape: vec![37, 7],
        n: 50,
        weight_signed: true,
        scale_shape: Some(vec![50]),
        quantized_in_graph: true,
        relu: true,
        quantized_after: true,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn fallback_leaves_the_range_to_the_quantization() {
    Layer {
        a_shape: vec![64, 32],
        n: 32,
        weight_signed: false,
        scale_shape: Some(vec![64, 1]),
        quantized_in_graph: false,
        relu: true,
        quantized_after: true,
        weight_scale_per_column: false,
    }
    .check(false);
}

#[test]
fn epilogue_forms_a_product_with_one_weight_scale_per_column() {
    // the text encoder's shape class: few rows, many columns
    Layer {
        a_shape: vec![1, 32, 256],
        n: 320,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: true,
        relu: false,
        quantized_after: true,
        weight_scale_per_column: true,
    }
    .check(true);
}

#[test]
fn product_with_one_weight_scale_per_column_on_the_largest_block() {
    Layer {
        a_shape: vec![700, 96],
        n: 330,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: false,
        relu: true,
        quantized_after: false,
        weight_scale_per_column: true,
    }
    .check(true);
}

#[test]
fn any_k_runs_the_product_first() {
    Layer {
        a_shape: vec![37, 7],
        n: 50,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: true,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: true,
    }
    .check(true);
}

#[test]
fn small_epilogue_on_a_single_token() {
    // one row, SAM 3's presence token: no 16×16 tile fits, one output per
    // thread with the product, the Relu and the range
    Layer {
        a_shape: vec![1, 1, 256],
        n: 48,
        weight_signed: false,
        scale_shape: None,
        quantized_in_graph: true,
        relu: true,
        quantized_after: true,
        weight_scale_per_column: true,
    }
    .check(true);
}

#[test]
fn small_epilogue_on_a_narrow_head() {
    // many rows, few columns: SAM 3's box and point heads
    Layer {
        a_shape: vec![3, 200, 64],
        n: 8,
        weight_signed: true,
        scale_shape: None,
        quantized_in_graph: true,
        relu: false,
        quantized_after: true,
        weight_scale_per_column: true,
    }
    .check(true);
}

#[test]
fn small_epilogue_with_one_scale_and_a_single_column() {
    Layer {
        a_shape: vec![1, 1, 256],
        n: 1,
        weight_signed: true,
        scale_shape: Some(vec![]),
        quantized_in_graph: false,
        relu: false,
        quantized_after: false,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn quantization_reads_the_range_the_small_epilogue_folded() {
    // A fed from the host: the only `DynamicQuantizeLinear` is the one after,
    // so its min/max pass compiles only if the kernel did not fold the range
    Layer {
        a_shape: vec![2, 150, 96],
        n: 4,
        weight_signed: true,
        scale_shape: Some(vec![4]),
        quantized_in_graph: false,
        relu: false,
        quantized_after: true,
        weight_scale_per_column: false,
    }
    .check(true);
}

#[test]
fn fallback_runs_the_product_first() {
    // K a multiple of 4 but of no cooperative K tile, with 16 rows and
    // columns or more: the four steps after the product
    Layer {
        a_shape: vec![48, 36],
        n: 40,
        weight_signed: false,
        scale_shape: None,
        quantized_in_graph: true,
        relu: true,
        quantized_after: true,
        weight_scale_per_column: true,
    }
    .check(false);
}
