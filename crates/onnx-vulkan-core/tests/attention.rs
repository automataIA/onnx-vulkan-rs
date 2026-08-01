//! `GroupQueryAttention` against a scalar reference.
//!
//! The op fuses four things that are usually four nodes — rotary embedding,
//! the KV-cache append, causal and sliding-window masking, and the grouped
//! query mapping — so the reference here is written out longhand and the test
//! is what says the fusion is faithful. The geometry is gemma3-1b's, shrunk:
//! 4 query heads over 1 key/value head, half-split rotary, and the two window
//! settings its layers alternate between (512 local, -1 global).

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

const NH: usize = 4;
const KVH: usize = 1;
const H: usize = 8;

/// Deterministic values in a small range: the reference and the kernel sum in
/// different orders, and huge magnitudes would hide that behind rounding.
fn ramp(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 37 + seed * 11) % 23) as f32 - 11.0) / 16.0)
        .collect()
}

struct Case {
    seq: usize,
    past: usize,
    window: i64,
    rotary: bool,
    bias: bool,
}

/// The whole op, in scalars. Layouts as documented in `shaders::attention`.
#[allow(clippy::too_many_arguments)]
fn reference(
    case: &Case,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    past_k: &[f32],
    past_v: &[f32],
    cos: &[f32],
    sin: &[f32],
    bias: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (seq, past) = (case.seq, case.past);
    let total = past + seq;
    let half = H / 2;
    let scale = 1.0 / (H as f32).sqrt();

    // rotary on one head-sized slice, half-split: channel j pairs with j + H/2
    let rotate = |x: &[f32], base: usize, pos: usize| -> Vec<f32> {
        (0..H)
            .map(|d| {
                if !case.rotary {
                    return x[base + d];
                }
                let j = if d < half { d } else { d - half };
                let (c, s) = (cos[pos * half + j], sin[pos * half + j]);
                if d < half {
                    x[base + d] * c - x[base + d + half] * s
                } else {
                    x[base + d] * c + x[base + d - half] * s
                }
            })
            .collect()
    };

    // present = past copied, then the new step appended at time `past`
    let mut present_k = vec![0.0; KVH * total * H];
    let mut present_v = vec![0.0; KVH * total * H];
    for h in 0..KVH {
        for t in 0..past {
            for d in 0..H {
                present_k[(h * total + t) * H + d] = past_k[(h * past + t) * H + d];
                present_v[(h * total + t) * H + d] = past_v[(h * past + t) * H + d];
            }
        }
        for s in 0..seq {
            let rotated = rotate(k, (s * KVH + h) * H, past + s);
            for d in 0..H {
                present_k[(h * total + past + s) * H + d] = rotated[d];
                present_v[(h * total + past + s) * H + d] = v[(s * KVH + h) * H + d];
            }
        }
    }

    let mut out = vec![0.0; seq * NH * H];
    for head in 0..NH {
        let hkv = head / (NH / KVH);
        for s in 0..seq {
            let pq = past + s;
            let query = rotate(q, (s * NH + head) * H, pq);
            let mut scores = vec![f32::NEG_INFINITY; total];
            for (t, score) in scores.iter_mut().enumerate() {
                // exactly `window` visible keys, the query's own position
                // included; see the note on `shaders::attention::SCORES`
                if t > pq || (case.window >= 0 && (pq - t) as i64 >= case.window) {
                    continue;
                }
                let mut acc = 0.0;
                for d in 0..H {
                    acc += query[d] * present_k[(hkv * total + t) * H + d];
                }
                *score = acc * scale;
                if case.bias {
                    *score += bias[s * total + t];
                }
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for d in 0..H {
                let mut acc = 0.0;
                for (t, e) in exps.iter().enumerate() {
                    acc += (e / sum) * present_v[(hkv * total + t) * H + d];
                }
                out[(s * NH + head) * H + d] = acc;
            }
        }
    }
    (out, present_k, present_v)
}

fn run_case(case: Case) {
    let (seq, past) = (case.seq, case.past);
    let total = past + seq;
    let max_pos = total + 4;
    let half = H / 2;

    let q = ramp(seq * NH * H, 1);
    let k = ramp(seq * KVH * H, 2);
    let v = ramp(seq * KVH * H, 3);
    let past_k = ramp(KVH * past * H, 4);
    let past_v = ramp(KVH * past * H, 5);
    // a real rotary table: cos² + sin² = 1, so the rotation is norm-preserving
    // and a swapped pair shows up as a wrong magnitude, not only a wrong sign
    let angles: Vec<f32> = (0..max_pos * half)
        .map(|i| (i % 17) as f32 * 0.37)
        .collect();
    let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
    let bias = ramp(seq * total, 6);

    let attrs = vec![
        ("num_heads", AttrValue::Int(NH as i64)),
        ("kv_num_heads", AttrValue::Int(KVH as i64)),
        ("scale", AttrValue::Float(0.0)),
        ("softcap", AttrValue::Float(0.0)),
        ("do_rotary", AttrValue::Int(i64::from(case.rotary))),
        ("rotary_interleaved", AttrValue::Int(0)),
        ("local_window_size", AttrValue::Int(case.window)),
    ];
    let inputs = [
        "query",
        "key",
        "value",
        "past_key",
        "past_value",
        "seqlens_k",
        "total_seq",
        "cos_cache",
        "sin_cache",
        "",
        if case.bias { "bias" } else { "" },
    ];
    let node = NodeIr {
        domain: "com.microsoft".into(),
        op: "GroupQueryAttention".into(),
        opset: 1,
        name: "gqa".into(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: ["out", "present_key", "present_value"]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        attrs: attrs
            .into_iter()
            .map(|(name, value)| (name.to_string(), value))
            .collect(),
    };
    assert!(is_implemented_node(&node), "the node must be claimable");

    let ir = GraphIr {
        nodes: vec![node],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into(), "present_key".into(), "present_value".into()],
        ..Default::default()
    };

    let host = [
        ("query", vec![1, seq as i64, (NH * H) as i64], &q),
        ("key", vec![1, seq as i64, (KVH * H) as i64], &k),
        ("value", vec![1, seq as i64, (KVH * H) as i64], &v),
        (
            "past_key",
            vec![1, KVH as i64, past as i64, H as i64],
            &past_k,
        ),
        (
            "past_value",
            vec![1, KVH as i64, past as i64, H as i64],
            &past_v,
        ),
        ("cos_cache", vec![max_pos as i64, half as i64], &cos),
        ("sin_cache", vec![max_pos as i64, half as i64], &sin),
        ("bias", vec![1, 1, seq as i64, total as i64], &bias),
    ];

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    for (name, shape, data) in host {
        env.set(
            name,
            Tensor::Host(HostTensor::from_f32(shape, data.as_slice())),
        );
    }
    execute(&ir, &mut env).expect("graph execution");
    let got: Vec<Vec<f32>> = ir
        .outputs
        .iter()
        .map(|name| env.host(name).expect("output on host").to_f32().unwrap())
        .collect();
    env.finish();

    let (out, present_k, present_v) =
        reference(&case, &q, &k, &v, &past_k, &past_v, &cos, &sin, &bias);
    for (name, expected, actual) in [
        ("output", &out, &got[0]),
        ("present_key", &present_k, &got[1]),
        ("present_value", &present_v, &got[2]),
    ] {
        assert_eq!(expected.len(), actual.len(), "{name}: length");
        for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
            assert!(
                (e - a).abs() <= 1e-5,
                "{name}[{i}] = {a}, reference {e} (seq {seq}, past {past}, window {})",
                case.window
            );
        }
    }
}

/// Prefill: no cache to copy, the mask is purely causal.
#[test]
fn prefill_is_causal_and_appends_to_an_empty_cache() {
    run_case(Case {
        seq: 6,
        past: 0,
        window: -1,
        rotary: true,
        bias: false,
    });
}

/// Decode: one token, the whole cache is copied forward, and the rotary
/// position comes from the past length alone.
#[test]
fn decode_step_reads_the_cache_and_rotates_at_the_past_position() {
    run_case(Case {
        seq: 1,
        past: 9,
        window: -1,
        rotary: true,
        bias: false,
    });
}

/// The sliding window of gemma3's 22 local layers. Chosen shorter than the
/// sequence, so it actually masks something the causal mask would keep.
#[test]
fn sliding_window_masks_beyond_the_local_span() {
    run_case(Case {
        seq: 5,
        past: 7,
        window: 3,
        rotary: true,
        bias: false,
    });
}

/// The attention bias input, broadcast over heads, together with a window.
#[test]
fn attention_bias_is_added_before_the_softmax() {
    run_case(Case {
        seq: 4,
        past: 4,
        window: 5,
        rotary: true,
        bias: true,
    });
}

/// `do_rotary = 0` must leave query and key untouched.
#[test]
fn rotary_can_be_switched_off() {
    run_case(Case {
        seq: 3,
        past: 2,
        window: -1,
        rotary: false,
        bias: false,
    });
}
