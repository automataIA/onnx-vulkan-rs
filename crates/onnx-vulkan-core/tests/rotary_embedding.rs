//! `RotaryEmbedding` (com.microsoft) against a scalar reference.
//!
//! The op is the rotation `GroupQueryAttention` fuses, so the arithmetic is
//! already covered there — what this test exists for is everything around it:
//! the two input layouts the schema admits, positions read from a tensor rather
//! than derived from the cache length, and a `rotary_embedding_dim` narrower
//! than the head. The positions are deliberately **not** `0, 1, 2, …`: qwen2.5-
//! VL's mRoPE is why this op exists as a node instead of an attribute, and a
//! kernel that silently used the token index would pass a monotonic test.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

/// Head size, and the cache is wide enough for a full-head rotation.
const H: usize = 8;
const CACHE_HALF: usize = H / 2;

fn ramp(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 29 + seed * 7) % 19) as f32 - 9.0) / 8.0)
        .collect()
}

struct Case {
    /// `[b, n, s, H]` when set, `[b, s, n·H]` otherwise.
    bnsh: bool,
    heads: usize,
    seq: usize,
    /// `rotary_embedding_dim`; 0 means the whole head.
    rot_dim: usize,
    positions: Vec<i64>,
}

/// Half-split rotary, longhand, on whichever layout the case asks for.
fn reference(case: &Case, x: &[f32], cos: &[f32], sin: &[f32]) -> Vec<f32> {
    let rot = if case.rot_dim == 0 { H } else { case.rot_dim };
    let half = rot / 2;
    let mut out = x.to_vec();
    for seq in 0..case.seq {
        let pos = case.positions[seq] as usize;
        for head in 0..case.heads {
            let base = if case.bnsh {
                (head * case.seq + seq) * H
            } else {
                (seq * case.heads + head) * H
            };
            for d in 0..rot {
                let j = if d < half { d } else { d - half };
                let (c, s) = (cos[pos * CACHE_HALF + j], sin[pos * CACHE_HALF + j]);
                out[base + d] = if d < half {
                    x[base + d] * c - x[base + d + half] * s
                } else {
                    x[base + d] * c + x[base + d - half] * s
                };
            }
        }
    }
    out
}

fn run_case(case: Case) {
    let count = case.heads * case.seq * H;
    let x = ramp(count, 1);
    let max_pos = case.positions.iter().copied().max().unwrap_or(0) as usize + 1;
    // a real rotation table: cos² + sin² = 1, so a swapped pair is a wrong
    // magnitude and not only a wrong sign
    let angles: Vec<f32> = (0..max_pos * CACHE_HALF)
        .map(|i| (i % 13) as f32 * 0.41)
        .collect();
    let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();

    let mut attrs = vec![("interleaved".to_string(), AttrValue::Int(0))];
    if case.rot_dim != 0 {
        attrs.push((
            "rotary_embedding_dim".to_string(),
            AttrValue::Int(case.rot_dim as i64),
        ));
    }
    if !case.bnsh {
        attrs.push(("num_heads".to_string(), AttrValue::Int(case.heads as i64)));
    }
    let node = NodeIr {
        domain: "com.microsoft".into(),
        op: "RotaryEmbedding".into(),
        opset: 1,
        name: "rotary".into(),
        inputs: ["x", "position_ids", "cos_cache", "sin_cache"]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        outputs: vec!["out".into()],
        attrs: attrs.into_iter().collect(),
    };
    assert!(is_implemented_node(&node), "the node must be claimable");

    let ir = GraphIr {
        nodes: vec![node],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into()],
        ..Default::default()
    };
    let x_shape = if case.bnsh {
        vec![1, case.heads as i64, case.seq as i64, H as i64]
    } else {
        vec![1, case.seq as i64, (case.heads * H) as i64]
    };

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set("x", Tensor::Host(HostTensor::from_f32(x_shape, &x)));
    env.set(
        "position_ids",
        Tensor::Host(HostTensor::from_i64(
            vec![1, case.seq as i64],
            &case.positions,
        )),
    );
    for (name, data) in [("cos_cache", &cos), ("sin_cache", &sin)] {
        env.set(
            name,
            Tensor::Host(HostTensor::from_f32(
                vec![max_pos as i64, CACHE_HALF as i64],
                data,
            )),
        );
    }
    execute(&ir, &mut env).expect("graph execution");
    let got = env
        .host("out")
        .expect("output on host")
        .to_f32()
        .expect("f32 output");
    env.finish();

    let expected = reference(&case, &x, &cos, &sin);
    assert_eq!(expected.len(), got.len(), "length");
    for (i, (e, a)) in expected.iter().zip(&got).enumerate() {
        assert!(
            (e - a).abs() <= 1e-5,
            "out[{i}] = {a}, reference {e} (bnsh {}, rot_dim {})",
            case.bnsh,
            case.rot_dim
        );
    }
}

/// The layout qwen2.5-VL's decoder exports: `[b, n, s, H]`, whole head rotated,
/// positions out of order — which is what mRoPE hands the node.
#[test]
fn bnsh_layout_rotates_at_the_given_positions() {
    run_case(Case {
        bnsh: true,
        heads: 3,
        seq: 4,
        rot_dim: 0,
        positions: vec![5, 0, 2, 2],
    });
}

/// The other layout the schema admits, where `num_heads` is the only thing
/// separating heads from channels.
#[test]
fn bsnh_layout_uses_num_heads_to_split_the_hidden_size() {
    run_case(Case {
        bnsh: false,
        heads: 2,
        seq: 3,
        rot_dim: 0,
        positions: vec![1, 4, 0],
    });
}

/// A partial rotation: the channels at or past `rotary_embedding_dim` must come
/// out untouched, which a kernel rotating the whole head would break.
#[test]
fn channels_past_the_rotary_dim_are_copied() {
    run_case(Case {
        bnsh: true,
        heads: 2,
        seq: 2,
        rot_dim: 4,
        positions: vec![3, 1],
    });
}

/// A decode step: one token, and its position is the only thing that says where
/// in the sequence it sits.
#[test]
fn single_token_reads_its_own_position() {
    run_case(Case {
        bnsh: true,
        heads: 4,
        seq: 1,
        rot_dim: 0,
        positions: vec![7],
    });
}
