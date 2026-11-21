//! `Pad` in the modes `edge`, `reflect` and `wrap`, on the device (f32) and
//! on the host (other types), against numpy's `np.pad`. Supertonic's text
//! encoder pads with `edge`; before these modes a node in them was claimed
//! and then failed at run time, which ends hosts such as sherpa-onnx.
//!
//! A mode the kernel lacks, and the opset 18 `axes` input, are refused at
//! load time so that the node goes to another provider.

use onnx_vulkan_core::host_ops::PadMode;
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, HostTensor, KernelCache, NodeIr, Tensor, execute,
    is_implemented_node,
};
use vk_compute::VkContext;

const FLOAT: i32 = 1;
const INT64: i32 = 7;

fn pad_node(mode: Option<&str>, inputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: "Pad".to_string(),
        opset: 18,
        name: "Pad_modes_test".to_string(),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: vec!["y".to_string()],
        attrs: mode
            .map(|m| [("mode".to_string(), AttrValue::String(m.to_string()))].into())
            .unwrap_or_default(),
    }
}

fn ints(values: &[i64]) -> HostTensor {
    HostTensor::new(
        INT64,
        vec![values.len() as i64],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

fn floats(shape: &[i64], values: &[f32]) -> HostTensor {
    HostTensor::new(
        FLOAT,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
}

/// Runs a `Pad` of `x` by `pads`; `x` is uploaded when `device`.
fn run(mode: &str, x: HostTensor, pads: &[i64], device: bool) -> HostTensor {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let node = pad_node(Some(mode), &["x", "pads"]);
    assert!(is_implemented_node(&node), "mode {mode} must be claimed");
    let graph = GraphIr {
        inputs: node.inputs.clone(),
        outputs: vec!["y".to_string()],
        nodes: vec![node],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("x", Tensor::Host(x));
    if device {
        env.ensure_device_dtype("x").expect("upload");
    }
    env.set("pads", Tensor::Host(ints(pads)));
    execute(&graph, &mut env).expect("graph execution");
    let y = env.host("y").expect("output");
    env.finish();
    y
}

fn as_f32(t: &HostTensor) -> Vec<f32> {
    t.data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn as_i64(t: &HostTensor) -> Vec<i64> {
    t.data
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// np.pad(x, ((0, 1), (2, 1)), mode=...) for x = [[1, 2, 3], [4, 5, 6]]
const MATRIX: [(&str, [f32; 18]); 3] = [
    (
        "edge",
        [
            1., 1., 1., 2., 3., 3., 4., 4., 4., 5., 6., 6., 4., 4., 4., 5., 6., 6.,
        ],
    ),
    (
        "reflect",
        [
            3., 2., 1., 2., 3., 2., 6., 5., 4., 5., 6., 5., 3., 2., 1., 2., 3., 2.,
        ],
    ),
    (
        "wrap",
        [
            2., 3., 1., 2., 3., 1., 5., 6., 4., 5., 6., 4., 2., 3., 1., 2., 3., 1.,
        ],
    ),
];

// np.pad([1, 2, 3], (5, 5), mode=...): pads wider than the dimension
const WIDE: [(&str, [f32; 13]); 3] = [
    ("edge", [1., 1., 1., 1., 1., 1., 2., 3., 3., 3., 3., 3., 3.]),
    (
        "reflect",
        [2., 1., 2., 3., 2., 1., 2., 3., 2., 1., 2., 3., 2.],
    ),
    ("wrap", [2., 3., 1., 2., 3., 1., 2., 3., 1., 2., 3., 1., 2.]),
];

#[test]
fn device_matches_numpy() {
    for (mode, want) in MATRIX {
        let x = floats(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let y = run(mode, x, &[0, 2, 1, 1], true);
        assert_eq!(y.shape, vec![3, 6], "{mode}");
        assert_eq!(as_f32(&y), want, "{mode}");
    }
    for (mode, want) in WIDE {
        let y = run(mode, floats(&[3], &[1., 2., 3.]), &[5, 5], true);
        assert_eq!(as_f32(&y), want, "{mode} wide");
    }
    let y = run("reflect", floats(&[1], &[7.]), &[2, 2], true);
    assert_eq!(as_f32(&y), [7.; 5], "reflect of one element");
}

#[test]
fn negative_pads_crop_in_every_mode() {
    for mode in ["edge", "reflect", "wrap"] {
        let y = run(mode, floats(&[4], &[1., 2., 3., 4.]), &[-1, 2], true);
        let want: &[f32] = match mode {
            "edge" => &[2., 3., 4., 4., 4.],
            "reflect" => &[2., 3., 4., 3., 2.],
            _ => &[2., 3., 4., 2., 3.],
        };
        assert_eq!(as_f32(&y), want, "{mode}");
        let x = HostTensor::new(
            INT64,
            vec![4],
            [1i64, 2, 3, 4]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        let host = run(mode, x, &[-1, 2], false);
        assert_eq!(
            as_i64(&host),
            want.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "{mode}"
        );
    }
}

#[test]
fn host_matches_numpy() {
    for (mode, want) in MATRIX {
        let x = HostTensor::new(
            INT64,
            vec![2, 3],
            [1i64, 2, 3, 4, 5, 6]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        );
        let y = run(mode, x, &[0, 2, 1, 1], false);
        let want: Vec<i64> = want.iter().map(|&v| v as i64).collect();
        assert_eq!(as_i64(&y), want, "{mode}");
    }
}

#[test]
fn source_index() {
    assert_eq!(PadMode::Edge.source(-3, 4), Some(0));
    assert_eq!(PadMode::Edge.source(9, 4), Some(3));
    assert_eq!(PadMode::Reflect.source(-1, 4), Some(1));
    assert_eq!(PadMode::Reflect.source(4, 4), Some(2));
    assert_eq!(PadMode::Wrap.source(-1, 4), Some(3));
    assert_eq!(PadMode::Constant.source(-1, 4), None);
    assert_eq!(PadMode::Edge.source(0, 0), None);
}

#[test]
fn unknown_mode_and_axes_are_refused_at_load() {
    assert!(is_implemented_node(&pad_node(None, &["x", "pads"])));
    assert!(is_implemented_node(&pad_node(
        Some("constant"),
        &["x", "pads", "v"]
    )));
    assert!(!is_implemented_node(&pad_node(
        Some("mirror"),
        &["x", "pads"]
    )));
    assert!(!is_implemented_node(&pad_node(
        Some("edge"),
        &["x", "pads", "", "axes"]
    )));
    assert!(is_implemented_node(&pad_node(
        Some("edge"),
        &["x", "pads", "", ""]
    )));
}

#[test]
fn empty_dimension_is_an_error_not_a_crash() {
    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let node = pad_node(Some("edge"), &["x", "pads"]);
    let graph = GraphIr {
        inputs: node.inputs.clone(),
        outputs: vec!["y".to_string()],
        nodes: vec![node],
        ..Default::default()
    };
    let mut env = ExecutionEnv::new(&cache, &graph.initializers);
    env.set("x", Tensor::Host(floats(&[0], &[])));
    env.set("pads", Tensor::Host(ints(&[1, 1])));
    assert!(execute(&graph, &mut env).is_err());
    env.finish();
}
