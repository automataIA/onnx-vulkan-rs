//! A device `Transpose` runs on its collapsed axes: axes of size 1 dropped,
//! input axes that stay adjacent merged. Each route — a plain copy, whole
//! 16-byte rows when the innermost axis stays innermost (SAM 3's window
//! partition), 32×32 tiles of 4-byte units when the innermost axis moves or
//! keeps less than a row word, the element kernels otherwise — must move every
//! byte where the permutation says, compared here against a host permutation.
//!
//! Identity `Pad` and `Slice` of a device activation are copies: their
//! output must be the input, bit for bit.

use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, HostTensor, KernelCache, NodeIr, Tensor, execute,
};
use vk_compute::VkContext;

const FLOAT: i32 = 1;
const UINT8: i32 = 2;
const INT64: i32 = 7;

fn node(op: &str, inputs: &[&str], attrs: &[(&str, AttrValue)]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_rows_test"),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec!["y".to_string()],
        attrs: attrs
            .iter()
            .map(|(key, value)| ((*key).to_string(), value.clone()))
            .collect(),
    }
}

/// Deterministic bytes (a 64-bit LCG's high byte): as f32 words they hold
/// NaNs, infinities and subnormals, so any re-encoding would show.
fn bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        })
        .collect()
}

fn ints(values: &[i64]) -> HostTensor {
    HostTensor::new(
        INT64,
        vec![values.len() as i64],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// Runs `op` on `x` uploaded to the device (the other inputs stay on host);
/// returns the output and the operations of the kernels it compiled.
fn run(op: NodeIr, x: HostTensor, rest: Vec<(&str, HostTensor)>) -> (HostTensor, Vec<String>) {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let graph = GraphIr {
        inputs: op.inputs.clone(),
        outputs: vec!["y".to_string()],
        nodes: vec![op],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("x", Tensor::Host(x));
    env.ensure_device_dtype("x").expect("upload");
    for (name, tensor) in rest {
        env.set(name, Tensor::Host(tensor));
    }
    execute(&graph, &mut env).expect("graph execution");
    assert!(env.on_device("y"), "a device activation must stay there");
    let y = env.host("y").expect("output");
    env.finish();
    let kernels = cache
        .pipeline_keys()
        .iter()
        .map(|key| key.operation().to_string())
        .collect();
    (y, kernels)
}

/// The permutation on host, element by element.
fn permuted(data: &[u8], size: usize, shape: &[i64], perm: &[usize]) -> Vec<u8> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as usize;
    }
    let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p] as usize).collect();
    let n: usize = out_shape.iter().product();
    let mut out = Vec::with_capacity(n * size);
    for o in 0..n {
        let (mut rest, mut src) = (o, 0);
        for d in (0..rank).rev() {
            src += (rest % out_shape[d]) * strides[perm[d]];
            rest /= out_shape[d];
        }
        out.extend_from_slice(&data[src * size..(src + 1) * size]);
    }
    out
}

/// Transposes on the device and checks every byte; `kernel` is the
/// operation that must have run (`None`: no kernel, a plain copy).
fn check(dtype: i32, shape: &[i64], perm: &[usize], kernel: Option<&str>) {
    let size = if dtype == UINT8 { 1 } else { 4 };
    let n = shape.iter().product::<i64>() as usize;
    let data = bytes(n as u64 * 31 + perm.len() as u64, n * size);
    let perm_attr = AttrValue::Ints(perm.iter().map(|&p| p as i64).collect());
    let (y, kernels) = run(
        node("Transpose", &["x"], &[("perm", perm_attr)]),
        HostTensor::new(dtype, shape.to_vec(), data.clone()),
        Vec::new(),
    );
    let label = format!("dtype {dtype} {shape:?} perm {perm:?}");
    let out_shape: Vec<i64> = perm.iter().map(|&p| shape[p]).collect();
    assert_eq!(y.shape, out_shape, "{label}");
    assert_eq!(y.data, permuted(&data, size, shape, perm), "{label}");
    let ran: Vec<&str> = kernels
        .iter()
        .map(String::as_str)
        .filter(|op| op.starts_with("Transpose"))
        .collect();
    assert_eq!(ran, kernel.into_iter().collect::<Vec<_>>(), "{label}");
}

#[test]
fn window_partition_copies_whole_rows() {
    // SAM 3's [B, a, u, b, v, C] → [B, a, b, u, v, C], and back
    check(
        FLOAT,
        &[1, 3, 4, 3, 4, 8],
        &[0, 1, 3, 2, 4, 5],
        Some("Transpose_rows"),
    );
    check(
        FLOAT,
        &[2, 3, 3, 4, 4, 8],
        &[0, 1, 3, 2, 4, 5],
        Some("Transpose_rows"),
    );
    // one 4-byte element per row word, and bytes 16 to a row word
    check(FLOAT, &[5, 3, 4], &[1, 0, 2], Some("Transpose_rows"));
    check(UINT8, &[3, 5, 2, 16], &[2, 1, 0, 3], Some("Transpose_rows"));
}

#[test]
fn rows_that_are_not_whole_words_use_tiles_or_the_element_kernels() {
    // 12-byte rows: the innermost axis stays, but not on 16-byte words, and
    // moves as a 3-word unit
    check(
        FLOAT,
        &[1, 3, 4, 3, 1, 3],
        &[0, 1, 3, 2, 4, 5],
        Some("Transpose_tiled"),
    );
    // 20-byte rows: a unit larger than a tile holds
    check(FLOAT, &[3, 4, 5], &[1, 0, 2], Some("Transpose"));
    check(UINT8, &[3, 5, 6], &[1, 0, 2], Some("Transpose_bytes"));
    // the innermost axis moves
    check(
        FLOAT,
        &[2, 3, 37, 10],
        &[0, 1, 3, 2],
        Some("Transpose_tiled"),
    );
    // six batch axes: more than the tiled kernel takes
    check(
        FLOAT,
        &[2, 3, 4, 5, 6, 7, 2, 3],
        &[7, 5, 3, 1, 6, 4, 2, 0],
        Some("Transpose"),
    );
    check(UINT8, &[2, 3, 5], &[2, 0, 1], Some("Transpose_bytes"));
}

#[test]
fn tiles_cover_partial_edges_units_and_batch_axes() {
    // SAM 3's NHWC ↔ NCHW, smaller: tiles cut at 32 on both axes
    check(
        FLOAT,
        &[1, 45, 70, 67],
        &[0, 3, 1, 2],
        Some("Transpose_tiled"),
    );
    check(
        FLOAT,
        &[1, 67, 45, 70],
        &[0, 2, 3, 1],
        Some("Transpose_tiled"),
    );
    // its pixel shuffle [N, H, W, kh, C, kw] → [N, C, H, kh, W, kw]: pairs
    // of words move together, H and kh are batch axes
    check(
        FLOAT,
        &[1, 9, 37, 2, 40, 2],
        &[0, 4, 1, 3, 2, 5],
        Some("Transpose_tiled"),
    );
    // four batch axes around the tile, the largest batch the kernel takes
    check(
        FLOAT,
        &[3, 2, 33, 2, 5, 34],
        &[1, 5, 4, 0, 3, 2],
        Some("Transpose_tiled"),
    );
}

#[test]
fn axes_of_size_one_and_merged_axes_collapse() {
    // [1, 4, 1, 6] → [1, 6, 4, 1] is [4, 6] → [6, 4]
    check(FLOAT, &[1, 4, 1, 6], &[2, 3, 0, 1], Some("Transpose_tiled"));
    // [2, 3, 4, 5] → [4, 5, 2, 3] is [6, 20] → [20, 6]
    check(FLOAT, &[2, 3, 4, 5], &[2, 3, 0, 1], Some("Transpose_tiled"));
    // nothing moves: a copy
    check(FLOAT, &[1, 5, 1], &[2, 1, 0], None);
    check(FLOAT, &[3, 4], &[0, 1], None);
    check(UINT8, &[1, 7], &[1, 0], None);
    check(FLOAT, &[1, 1, 1], &[2, 0, 1], None);
}

#[test]
fn identity_pad_and_slice_copy_the_input() {
    let shape = vec![1, 6, 6, 8];
    let data = bytes(5, 6 * 6 * 8 * 4);
    let x = || HostTensor::new(FLOAT, shape.clone(), data.clone());
    let (y, kernels) = run(
        node(
            "Pad",
            &["x", "pads"],
            &[("mode", AttrValue::String("constant".into()))],
        ),
        x(),
        vec![("pads", ints(&[0; 8]))],
    );
    assert_eq!((y.shape.clone(), y.data), (shape.clone(), data.clone()));
    assert!(!kernels.iter().any(|op| op == "Pad"), "{kernels:?}");

    let (y, kernels) = run(
        node("Slice", &["x", "starts", "ends", "axes", "steps"], &[]),
        x(),
        vec![
            ("starts", ints(&[0, 0])),
            ("ends", ints(&[6, i64::MAX])),
            ("axes", ints(&[1, 2])),
            ("steps", ints(&[1, 1])),
        ],
    );
    assert_eq!((y.shape.clone(), y.data), (shape.clone(), data.clone()));
    assert!(!kernels.iter().any(|op| op == "Slice"), "{kernels:?}");

    // a real crop still slices
    let (y, kernels) = run(
        node("Slice", &["x", "starts", "ends", "axes"], &[]),
        x(),
        vec![
            ("starts", ints(&[0])),
            ("ends", ints(&[5])),
            ("axes", ints(&[1])),
        ],
    );
    assert_eq!(y.shape, vec![1, 5, 6, 8]);
    assert_eq!(y.data, data[..5 * 6 * 8 * 4]);
    assert!(kernels.iter().any(|op| op == "Slice"), "{kernels:?}");
}
