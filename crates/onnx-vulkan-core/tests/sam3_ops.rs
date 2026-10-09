//! Graph-level coverage for the cases SAM 3's prompt decoder hits: a
//! `MatMulInteger` with K = 2, an integer `Clip`, a `DynamicQuantizeLinear`
//! of an empty tensor ("no boxes"), and the `Tile`/`Expand`/u8 `Transpose`
//! of image-sized activations that must stay on the GPU, whether they come
//! from a node, a graph input or a constant. The text-prompt detector adds
//! `Sign`, a GroupNorm exported as `InstanceNormalization` and a MatMul-form
//! `Einsum`.

use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, Executor, GraphIr, HostTensor, InitializerIr, KernelCache, NodeIr,
    Tensor, execute,
};
use vk_compute::VkContext;

const UINT8: i32 = 2;
const INT32: i32 = 6;
const INT64: i32 = 7;

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_sam3_test"),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_string()).collect(),
        attrs: Default::default(),
    }
}

/// Runs one node with host inputs and returns its outputs read back.
fn run(op: NodeIr, inputs: Vec<(&str, HostTensor)>) -> Vec<HostTensor> {
    run_with(op, inputs, &[])
        .into_iter()
        .map(|(tensor, _)| tensor)
        .collect()
}

/// As `run`, uploading the inputs named in `upload` first; each output comes
/// with whether it was left on the device.
fn run_with(
    op: NodeIr,
    inputs: Vec<(&str, HostTensor)>,
    upload: &[&str],
) -> Vec<(HostTensor, bool)> {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        inputs: inputs.iter().map(|(name, _)| (*name).to_string()).collect(),
        outputs: op.outputs.clone(),
        nodes: vec![op],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    for (name, tensor) in inputs {
        env.set(name, Tensor::Host(tensor));
    }
    for name in upload {
        env.ensure_device_dtype(name).expect("upload");
    }
    execute(&graph, &mut env).expect("graph execution");
    let outputs = graph
        .outputs
        .iter()
        .map(|name| (env.host(name).expect("host output"), env.on_device(name)))
        .collect();
    env.finish();
    outputs
}

fn ints(dtype: i32, shape: Vec<i64>, values: &[i64]) -> HostTensor {
    let data = match dtype {
        UINT8 => values.iter().map(|v| *v as u8).collect(),
        INT32 => values
            .iter()
            .flat_map(|v| (*v as i32).to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    HostTensor::new(dtype, shape, data)
}

#[test]
fn matmul_integer_with_k_not_a_multiple_of_four() {
    // A [3, 2] u8, B [2, 3] u8: (a - 10) · (b - 128)
    let a = ints(UINT8, vec![3, 2], &[10, 11, 12, 255, 0, 10]);
    let b = ints(UINT8, vec![2, 3], &[128, 130, 0, 129, 128, 255]);
    let out = run(
        node("MatMulInteger", &["a", "b", "azp", "bzp"], &["y"]),
        vec![
            ("a", a),
            ("b", b),
            ("azp", ints(UINT8, vec![], &[10])),
            ("bzp", ints(UINT8, vec![], &[128])),
        ],
    );
    assert_eq!(out[0].dtype, INT32);
    assert_eq!(out[0].shape, vec![3, 3]);
    assert_eq!(
        out[0].to_i64().expect("i32 output"),
        vec![1, 0, 127, 245, 4, 30859, 0, -20, 1280]
    );
}

#[test]
fn integer_clip_keeps_the_dtype() {
    let out = run(
        node("Clip", &["x", "lo", ""], &["y"]),
        vec![
            ("x", ints(INT64, vec![4], &[-1, 0, 1, 7])),
            ("lo", ints(INT64, vec![], &[0])),
        ],
    );
    assert_eq!(out[0].dtype, INT64);
    assert_eq!(out[0].to_i64().expect("i64 output"), vec![0, 0, 1, 7]);
}

#[test]
fn dynamic_quantize_of_an_empty_tensor_matches_onnx_runtime() {
    let out = run(
        node("DynamicQuantizeLinear", &["x"], &["y", "scale", "zp"]),
        vec![("x", HostTensor::from_f32(vec![1, 0, 4], &[]))],
    );
    assert_eq!((out[0].dtype, out[0].shape.clone()), (UINT8, vec![1, 0, 4]));
    assert_eq!(out[1].to_f32().expect("scale"), vec![1.0]);
    assert_eq!(out[2].to_i64().expect("zero point"), vec![0]);
}

#[test]
fn tile_with_unit_repeats_copies_on_the_device() {
    let values: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let out = run_with(
        node("Tile", &["x", "repeats"], &["y"]),
        vec![
            ("x", HostTensor::from_f32(vec![1, 2, 3, 4], &values)),
            ("repeats", ints(INT64, vec![4], &[1, 1, 1, 1])),
        ],
        &["x"],
    );
    let (tensor, on_device) = &out[0];
    assert!(on_device, "a unit Tile must not leave the GPU");
    assert_eq!(tensor.shape, vec![1, 2, 3, 4]);
    assert_eq!(tensor.to_f32().expect("f32 output"), values);
}

#[test]
fn expand_of_a_host_vector_to_an_image_runs_on_the_gpu() {
    // [1, 3, 1, 1] → [1, 3, 40, 40]: 19 KB out, above the host threshold
    let out = run_with(
        node("Expand", &["x", "shape"], &["y"]),
        vec![
            (
                "x",
                HostTensor::from_f32(vec![1, 3, 1, 1], &[0.5, -1.0, 2.0]),
            ),
            ("shape", ints(INT64, vec![4], &[1, 1, 40, 40])),
        ],
        &[],
    );
    let (tensor, on_device) = &out[0];
    assert!(on_device, "an image-sized Expand must run on the GPU");
    assert_eq!(tensor.shape, vec![1, 3, 40, 40]);
    let got = tensor.to_f32().expect("f32 output");
    for (channel, value) in [0.5f32, -1.0, 2.0].into_iter().enumerate() {
        assert!(
            got[channel * 1600..(channel + 1) * 1600]
                .iter()
                .all(|v| *v == value)
        );
    }
}

#[test]
fn small_expand_stays_on_host() {
    let out = run_with(
        node("Expand", &["x", "shape"], &["y"]),
        vec![
            ("x", HostTensor::from_f32(vec![2, 1], &[1.0, 2.0])),
            ("shape", ints(INT64, vec![2], &[2, 3])),
        ],
        &[],
    );
    let (tensor, on_device) = &out[0];
    assert!(!on_device);
    assert_eq!(
        tensor.to_f32().expect("f32 output"),
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn u8_transpose_runs_on_the_device() {
    // [2, 3, 5] → perm [2, 0, 1] → [5, 2, 3]: 30 bytes, not a whole number of
    // words, and every output word gathers bytes from different source words
    let values: Vec<i64> = (0..30).map(|v| (v * 37 + 11) % 256).collect();
    let mut transpose = node("Transpose", &["x"], &["y"]);
    transpose
        .attrs
        .insert("perm".to_string(), AttrValue::Ints(vec![2, 0, 1]));
    let out = run_with(
        transpose,
        vec![("x", ints(UINT8, vec![2, 3, 5], &values))],
        &["x"],
    );
    let (tensor, on_device) = &out[0];
    assert!(on_device, "a device u8 Transpose must not leave the GPU");
    assert_eq!(tensor.dtype, UINT8);
    assert_eq!(tensor.shape, vec![5, 2, 3]);
    let mut expected = Vec::new();
    for k in 0..5 {
        for i in 0..2 {
            for j in 0..3 {
                expected.push(values[i * 15 + j * 5 + k] as u8);
            }
        }
    }
    assert_eq!(tensor.data, expected);
}

#[test]
fn unit_tile_of_a_large_constant_stays_on_the_device() {
    // SAM 3 tiles a 5.3 MB positional encoding by one: copied on host, the
    // chain down to the first GPU op ran on the CPU in standalone
    let values: Vec<f32> = (0..4800).map(|v| v as f32 * 0.25).collect();
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut graph = GraphIr {
        inputs: vec!["repeats".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![node("Tile", &["x", "repeats"], &["y"])],
        ..Default::default()
    };
    graph.initializers.insert(
        "x".to_string(),
        InitializerIr {
            dtype: 1,
            shape: vec![1, 3, 40, 40],
            data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        },
    );
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("repeats", Tensor::Host(ints(INT64, vec![4], &[1, 1, 1, 1])));
    execute(&graph, &mut env).expect("graph execution");
    assert!(
        env.on_device("y"),
        "a large constant must be copied in VRAM"
    );
    assert_eq!(
        env.host("y").expect("output").to_f32().expect("f32"),
        values
    );
    env.finish();
}

#[test]
fn large_f32_graph_input_is_bound_on_the_device() {
    // the EP receives image embeddings already in VRAM; the standalone run
    // must too, or every op reading them directly runs on host
    let context = VkContext::new().expect("Vulkan context");
    let mut graph = GraphIr {
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        nodes: vec![node("Unsqueeze", &["x", "axes"], &["y"])],
        ..Default::default()
    };
    graph.initializers.insert(
        "axes".to_string(),
        InitializerIr {
            dtype: INT64,
            shape: vec![1],
            data: 0i64.to_le_bytes().to_vec(),
        },
    );
    let executor = Executor::new(&context, graph).expect("executor");
    let values: Vec<f32> = (0..2048).map(|v| v as f32).collect();
    let outputs = executor
        .run(vec![(
            "x",
            Tensor::Host(HostTensor::from_f32(vec![2048], &values)),
        )])
        .expect("run");
    assert!(
        outputs.on_device("y"),
        "an 8 KB input must be uploaded at bind"
    );
    assert_eq!(outputs.host("y").expect("output").shape, vec![1, 2048]);
    outputs.finish();
}

fn with_attr(mut node: NodeIr, name: &str, value: AttrValue) -> NodeIr {
    node.attrs.insert(name.to_string(), value);
    node
}

fn einsum(equation: &str) -> NodeIr {
    with_attr(
        node("Einsum", &["a", "b"], &["y"]),
        "equation",
        AttrValue::String(equation.to_string()),
    )
}

#[test]
fn sign_of_floats() {
    let out = run(
        node("Sign", &["x"], &["y"]),
        vec![(
            "x",
            HostTensor::from_f32(vec![4], &[-2.0, 0.0, 3.5, -1e-30]),
        )],
    );
    assert_eq!(out[0].to_f32().expect("f32"), vec![-1.0, 0.0, 1.0, -1.0]);
}

#[test]
fn instance_norm_of_long_rows_far_from_zero() {
    // 20000 values per slice (three 8192-element chunks, the last partial)
    // around 1000 with a spread of ~1: E[x²] - mean² in f32 would lose the
    // variance entirely
    let (n, c, len) = (2usize, 3usize, 20000usize);
    let x: Vec<f32> = (0..n * c * len)
        .map(|i| 1000.0 + (i as f32 * 0.37).sin() * (1.0 + (i / len) as f32))
        .collect();
    let scale = [0.5f32, 1.0, 2.0];
    let bias = [0.0f32, -1.0, 3.0];
    let out = run(
        with_attr(
            node("InstanceNormalization", &["x", "s", "b"], &["y"]),
            "epsilon",
            AttrValue::Float(1e-5),
        ),
        vec![
            (
                "x",
                HostTensor::from_f32(vec![n as i64, c as i64, 100, 200], &x),
            ),
            ("s", HostTensor::from_f32(vec![3], &scale)),
            ("b", HostTensor::from_f32(vec![3], &bias)),
        ],
    );
    assert_eq!(out[0].shape, vec![2, 3, 100, 200]);
    let got = out[0].to_f32().expect("f32");
    for (row, slice) in x.chunks(len).enumerate() {
        let mean = slice.iter().map(|v| *v as f64).sum::<f64>() / len as f64;
        let var = slice
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / len as f64;
        let ch = row % c;
        for (i, v) in slice.iter().enumerate() {
            let want =
                (*v as f64 - mean) / (var + 1e-5).sqrt() * scale[ch] as f64 + bias[ch] as f64;
            let have = got[row * len + i] as f64;
            assert!(
                (have - want).abs() < 2e-3,
                "row {row} [{i}]: {have} vs {want}"
            );
        }
    }
}

#[test]
fn einsum_matmul_form_of_the_sam3_mask_head() {
    // bqc,bchw->bqhw with b=2, q=3, c=4, h=2, w=3
    let a: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 3.0).collect();
    let b: Vec<f32> = (0..48).map(|i| (i % 7) as f32 - 2.0).collect();
    let out = run(
        einsum("bqc,bchw->bqhw"),
        vec![
            ("a", HostTensor::from_f32(vec![2, 3, 4], &a)),
            ("b", HostTensor::from_f32(vec![2, 4, 2, 3], &b)),
        ],
    );
    assert_eq!(out[0].shape, vec![2, 3, 2, 3]);
    let mut want = vec![0f32; 36];
    for bi in 0..2 {
        for q in 0..3 {
            for p in 0..6 {
                want[(bi * 3 + q) * 6 + p] = (0..4)
                    .map(|ci| a[(bi * 3 + q) * 4 + ci] * b[(bi * 4 + ci) * 6 + p])
                    .sum();
            }
        }
    }
    assert_eq!(out[0].to_f32().expect("f32"), want);
}

#[test]
fn einsum_claims_only_matmul_forms() {
    for equation in ["bqc,bchw->bqhw", "ij,jk->ik", " ij , jk -> ik ", "i,j->ij"] {
        assert!(
            onnx_vulkan_core::is_implemented_node(&einsum(equation)),
            "{equation} should be claimed"
        );
    }
    // a transpose, an ellipsis, an implicit output, a diagonal, a label summed
    // on one side only, a reordered output
    for equation in [
        "ij,kj->ik",
        "...ij,...jk->...ik",
        "ij,jk",
        "ii,ij->ij",
        "ij,jk->i",
        "bij,bjk->bki",
    ] {
        assert!(
            !onnx_vulkan_core::is_implemented_node(&einsum(equation)),
            "{equation} should be refused"
        );
    }
}

#[test]
fn is_nan_guard_after_softmax_stays_on_the_device() {
    // Where(IsNaN(p), 0, p): the exporter's guard after attention; 7 values,
    // so the last bool word is partial
    let values = [
        1.0,
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        f32::from_bits(0x7fc0_0001),
    ];
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        inputs: vec!["p".to_string(), "zero".to_string()],
        outputs: vec!["nan".to_string(), "y".to_string()],
        nodes: vec![
            node("IsNaN", &["p"], &["nan"]),
            node("Where", &["nan", "zero", "p"], &["y"]),
        ],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("p", Tensor::Host(HostTensor::from_f32(vec![7], &values)));
    env.set("zero", Tensor::Host(HostTensor::from_f32(vec![1], &[0.0])));
    env.ensure_device_dtype("p").expect("upload");
    execute(&graph, &mut env).expect("graph execution");
    assert_eq!(
        env.host("nan").expect("mask").data,
        vec![0, 1, 1, 0, 0, 0, 1]
    );
    let y = env.host("y").expect("y").to_f32().expect("f32");
    assert_eq!(
        y[..6],
        [1.0, 0.0, 0.0, f32::INFINITY, f32::NEG_INFINITY, 0.0]
    );
    assert_eq!(y[6], 0.0);
    env.finish();
}

#[test]
fn is_nan_of_a_device_tensor_is_computed_on_the_device() {
    // alone: a consumer such as `Where` would upload a host mask again
    let out = run_with(
        node("IsNaN", &["p"], &["nan"]),
        vec![(
            "p",
            HostTensor::from_f32(vec![5], &[f32::NAN, 1.0, 2.0, 3.0, f32::NAN]),
        )],
        &["p"],
    );
    assert!(out[0].1, "IsNaN read a device tensor back to the host");
    assert_eq!(out[0].0.data, vec![1, 0, 0, 0, 1]);
}
