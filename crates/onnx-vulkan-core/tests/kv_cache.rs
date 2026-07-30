//! A resident KV cache, written in place across decode steps.
//!
//! The oracle is the stateless path — `present_*` freshly allocated at
//! `stride == total` and the past copied forward — which is itself validated
//! against the ONNX Runtime CPU EP on gemma3-1b. What this test pins is that
//! binding the cache changes nothing but where the bytes live:
//!
//! - the buffer is laid out for `MAX_SEQ` tokens and the run writes token `t`
//!   at its own slot, so every step runs `stride = 1024` against `total` = 1,
//!   2, 3 — the `stride > total` branch, which until now was written and never
//!   executed;
//! - passing the cache back as `past_*` makes source and destination the same
//!   memory, so `GQA_past` is skipped and the tokens already there must
//!   survive untouched;
//! - the borrowed buffer must outlive the run, i.e. neither `release` nor
//!   `release_owned` may reclaim it.
//!
//! The buffer identity is asserted at every step, not only the numbers: a
//! binding the kernel ignored would allocate its own cache, fall back to
//! `stride == total`, and still agree with the oracle on every value. The
//! skipped `GQA_past` is not separately observable here — at `b · kvh == 1`
//! the copy it skips would be a self-copy between identical addresses, so it
//! is a no-op either way. What it costs is the whole cache per node per token,
//! which is a measurement, not an assertion.

use onnx_vulkan_core::host_ops::HostTensor;
use onnx_vulkan_core::{
    AttrValue, DeviceBuffer, DeviceTensor, Executor, GraphIr, NodeIr, Tensor, host_ops,
};
use std::collections::HashMap;
use vk_compute::{GpuBuffer, VkContext};

const NH: usize = 4;
const H: usize = 8;
const MAX_SEQ: usize = 1024;
const STEPS: usize = 3;

/// Deterministic values in a small range, distinct per step and per tensor.
fn ramp(count: usize, seed: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 31 + seed * 17) % 23) as f32 - 11.0) / 16.0)
        .collect()
}

fn graph(kvh: usize) -> GraphIr {
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
    ];
    let node = NodeIr {
        domain: "com.microsoft".into(),
        op: "GroupQueryAttention".into(),
        since_version: 1,
        name: "gqa".into(),
        inputs: inputs.iter().map(|s| (*s).to_string()).collect(),
        outputs: ["out", "present_key", "present_value"]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        attrs: [
            ("num_heads", AttrValue::Int(NH as i64)),
            ("kv_num_heads", AttrValue::Int(kvh as i64)),
            ("scale", AttrValue::Float(0.0)),
            ("do_rotary", AttrValue::Int(1)),
            ("rotary_interleaved", AttrValue::Int(0)),
            ("local_window_size", AttrValue::Int(-1)),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect(),
    };
    GraphIr {
        nodes: vec![node],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into(), "present_key".into(), "present_value".into()],
    }
}

/// Query, key and value of step `t`, plus the rotary tables.
fn step_inputs(t: usize, kvh: usize) -> Vec<(&'static str, HostTensor)> {
    let half = H / 2;
    let angles: Vec<f32> = (0..(MAX_SEQ + 1) * half)
        .map(|i| (i % 17) as f32 * 0.37)
        .collect();
    vec![
        (
            "query",
            HostTensor::from_f32(vec![1, 1, (NH * H) as i64], &ramp(NH * H, t + 1)),
        ),
        (
            "key",
            HostTensor::from_f32(vec![1, 1, (kvh * H) as i64], &ramp(kvh * H, t + 5)),
        ),
        (
            "value",
            HostTensor::from_f32(vec![1, 1, (kvh * H) as i64], &ramp(kvh * H, t + 9)),
        ),
        (
            "cos_cache",
            HostTensor::from_f32(
                vec![(MAX_SEQ + 1) as i64, half as i64],
                &angles.iter().map(|a| a.cos()).collect::<Vec<_>>(),
            ),
        ),
        (
            "sin_cache",
            HostTensor::from_f32(
                vec![(MAX_SEQ + 1) as i64, half as i64],
                &angles.iter().map(|a| a.sin()).collect::<Vec<_>>(),
            ),
        ),
    ]
}

/// The stateless path, `STEPS` decode steps with the cache carried on the host.
fn stateless(context: &'static VkContext, kvh: usize) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let executor = Executor::new(context, graph(kvh)).expect("executor");
    let mut past_k: Vec<f32> = Vec::new();
    let mut past_v: Vec<f32> = Vec::new();
    let mut steps = Vec::new();
    for t in 0..STEPS {
        let host = step_inputs(t, kvh);
        let cache_shape = vec![1, kvh as i64, t as i64, H as i64];
        let past = [
            ("past_key", HostTensor::from_f32(cache_shape.clone(), &past_k)),
            ("past_value", HostTensor::from_f32(cache_shape, &past_v)),
        ];
        let bound: Vec<(&str, Tensor<'_>)> = host
            .iter()
            .map(|(name, tensor)| (*name, Tensor::Host(tensor.clone())))
            .chain(
                past.iter()
                    .map(|(name, tensor)| (*name, Tensor::Host(tensor.clone()))),
            )
            .collect();
        let outputs = executor.run(bound).expect("stateless run");
        let read = |name: &str| outputs.host(name).expect(name).to_f32().unwrap();
        let (out, key, value) = (read("out"), read("present_key"), read("present_value"));
        outputs.finish();
        past_k = key.clone();
        past_v = value.clone();
        steps.push((out, key, value));
    }
    steps
}

/// The cache as this step's past: the first `t` tokens of the resident buffer.
fn past_view(buffer: &GpuBuffer, t: usize, kvh: usize) -> Tensor<'_> {
    Tensor::Device(DeviceTensor {
        dtype: host_ops::FLOAT,
        shape: vec![1, kvh as i64, t as i64, H as i64],
        elem_count: kvh * t * H,
        buf: DeviceBuffer::Borrowed(buffer),
    })
}

/// The resident path: one buffer per cache tensor, `MAX_SEQ` tokens long,
/// written in place and passed straight back as the next step's past.
fn resident(
    context: &'static VkContext,
    kvh: usize,
    key_buf: &GpuBuffer,
    value_buf: &GpuBuffer,
) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let capacity = kvh * MAX_SEQ * H;
    let executor = Executor::new(context, graph(kvh)).expect("executor");
    let mut steps = Vec::new();
    for t in 0..STEPS {
        let host = step_inputs(t, kvh);
        let mut inputs: Vec<(&str, Tensor<'_>)> = host
            .iter()
            .map(|(name, tensor)| (*name, Tensor::Host(tensor.clone())))
            .collect();
        inputs.push(("past_key", past_view(key_buf, t, kvh)));
        inputs.push(("past_value", past_view(value_buf, t, kvh)));
        let outputs = executor
            .run_with_outputs(
                inputs,
                vec![
                    ("present_key", key_buf, capacity),
                    ("present_value", value_buf, capacity),
                ],
            )
            .expect("resident run");
        // without this the test would pass on a fallback: an ignored binding
        // allocates a fresh `[1, kvh, total, H]` cache, `stride` collapses back
        // to `total`, and every number below still agrees
        for (name, want) in [("present_key", key_buf), ("present_value", value_buf)] {
            let Some(Tensor::Device(tensor)) = outputs.value(name) else {
                panic!("{name} is not on the device");
            };
            assert_eq!(
                tensor.buffer().buffer,
                want.buffer,
                "{name} at step {t} is not the buffer that was bound"
            );
        }
        // `out` first: reading it is the flush, so the cache buffers hold this
        // step's tokens by the time they are downloaded
        let out = outputs.host("out").expect("out").to_f32().unwrap();
        let cache = |buffer: &GpuBuffer| {
            destride(
                &context.download(buffer).expect("cache download"),
                kvh,
                t + 1,
            )
        };
        let (key, value) = (cache(key_buf), cache(value_buf));
        outputs.finish();
        steps.push((out, key, value));
    }
    steps
}

/// The written prefix of a padded cache, in the `[kvh, tokens, H]` layout the
/// stateless path produces.
///
/// Row `h` of the resident buffer starts `MAX_SEQ` tokens in, not `tokens` in,
/// which is the whole difference between the two layouts and the reason
/// `Outputs::host` refuses this tensor above `kvh = 1`: its leading elements are
/// row 0's tokens followed by padding, not the cache.
fn destride(bytes: &[u8], kvh: usize, tokens: usize) -> Vec<f32> {
    let all: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (0..kvh)
        .flat_map(|h| {
            let base = h * MAX_SEQ * H;
            all[base..base + tokens * H].to_vec()
        })
        .collect()
}

fn close(name: &str, step: usize, want: &[f32], got: &[f32]) {
    assert_eq!(want.len(), got.len(), "{name}: length at step {step}");
    for (i, (w, g)) in want.iter().zip(got).enumerate() {
        assert!(
            (w - g).abs() <= 1e-6,
            "{name}[{i}] at step {step}: {g}, stateless says {w}"
        );
    }
}

/// The whole point: three decode steps at `stride = 1024`, agreeing with the
/// path that rebuilds the cache every time.
///
/// Run at `kv_heads = 1` (multi-query, gemma3's shape) and at `kv_heads = 2`
/// with 4 query heads (grouped, which is what Llama 3 and Qwen export). The
/// second is not a variation on the first: with one row the padding sits behind
/// every written token and any stride arithmetic that ignores it still lands in
/// the right place, so a kernel that used `total` where it should use the
/// physical stride passes at `kvh = 1` and puts row 1's tokens on top of row 0's
/// here.
fn resident_matches_stateless(kvh: usize) {
    // leaked so the buffers can borrow a `'static` context, as `Session` does
    let context: &'static VkContext = Box::leak(Box::new(VkContext::new().expect("Vulkan")));
    let bytes = (kvh * MAX_SEQ * H * 4) as u64;
    let key_buf = context.create_storage_buffer(bytes).expect("key cache");
    let value_buf = context.create_storage_buffer(bytes).expect("value cache");

    let want = stateless(context, kvh);
    let got = resident(context, kvh, &key_buf, &value_buf);

    for (step, ((w_out, w_key, w_value), (g_out, g_key, g_value))) in
        want.iter().zip(&got).enumerate()
    {
        close("out", step, w_out, g_out);
        // the written prefix of each row of a 1024-token buffer: what the tokens
        // of earlier steps must still contain, untouched by the step that
        // skipped `GQA_past`
        close("present_key", step, w_key, g_key);
        close("present_value", step, w_value, g_value);
    }
}

#[test]
fn a_multi_query_cache_matches_the_stateless_path() {
    resident_matches_stateless(1);
}

#[test]
fn a_grouped_cache_matches_the_stateless_path() {
    resident_matches_stateless(2);
}
