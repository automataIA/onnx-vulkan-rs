//! The f16 cooperative-matrix `FlashAttention` kernels (heads of 32 and 64),
//! turned on with `ONNX_VULKAN_ATTENTION_FP16=1`: the subgroup kernel, and on
//! devices with `VK_NV_cooperative_matrix2` the workgroup-scope kernel for up
//! to `WG_MAX_KEYS` keys. Each test runs key counts on both sides of it.
//!
//! The switch is read once per process, so these tests live in their own
//! binary and every one of them turns it on.
//!
//! Tolerance: the kernel rounds Q, K, V and the probabilities to f16 and
//! accumulates in f32. Against attention in f64 over the same f16-rounded Q, K
//! and V, the only remaining roundings are the probabilities' (relative
//! 2^-11) and f32 accumulation, so every output is held to
//! `|Δ| ≤ 1e-3 · max|v|`. An addressing or rescaling bug moves outputs by
//! O(max|v|). This bounds the kernel; it is not the application's contract,
//! which is its own quality metric.

use onnx_vulkan_core::interp::ATTENTION_FP16_ENV;
use onnx_vulkan_core::shaders::flash_attention::WG_MAX_KEYS;
use onnx_vulkan_core::{AttrValue, Executor, GraphIr, HostTensor, InitializerIr, NodeIr, Tensor};
use vk_compute::VkContext;

fn opt_in() {
    // SAFETY: every test of this binary sets the same value before any
    // runtime code reads the environment.
    unsafe { std::env::set_var(ATTENTION_FP16_ENV, "1") };
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeIr {
    NodeIr {
        domain: String::new(),
        op: op.to_string(),
        opset: 18,
        name: format!("{op}_{}", outputs[0]),
        inputs: inputs.iter().map(|name| (*name).to_string()).collect(),
        outputs: outputs.iter().map(|name| (*name).to_string()).collect(),
        attrs: Default::default(),
    }
}

/// `y = softmax(q · kt) · v`, with the value and output head transposes of
/// multi-head attention when `heads_inside`.
fn graph(heads_inside: bool) -> GraphIr {
    let heads_swap = |input: &str, output: &str| {
        let mut n = node("Transpose", &[input], &[output]);
        n.attrs
            .insert("perm".into(), AttrValue::Ints(vec![0, 2, 1, 3]));
        n
    };
    let mut nodes = vec![
        node("MatMul", &["q", "kt"], &["scores"]),
        node("Softmax", &["scores"], &["probs"]),
    ];
    if heads_inside {
        nodes.insert(0, heads_swap("v_in", "v"));
        nodes.push(node("MatMul", &["probs", "v"], &["heads"]));
        nodes.push(heads_swap("heads", "y"));
    } else {
        nodes.push(node("MatMul", &["probs", "v_in"], &["y"]));
    }
    GraphIr {
        inputs: vec!["q".into(), "kt".into(), "v_in".into()],
        outputs: vec!["y".into()],
        nodes,
        ..Default::default()
    }
}

/// Deterministic values in `[-scale, scale)`.
fn values(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `x` rounded to the nearest f16, ties to even; finite f16 range only.
fn round_f16(x: f32) -> f32 {
    if x.abs() < 6.103_515_6e-5 {
        // subnormal f16: multiples of 2^-24
        let quantum = 2f32.powi(-24);
        return (x / quantum).round_ties_even() * quantum;
    }
    let bits = x.to_bits();
    let lsb = (bits >> 13) & 1;
    f32::from_bits((bits + 0xfff + lsb) & !0x1fff)
}

/// Attention in f64 over `[bh, m, d]` queries, `[bh, d, n]` keys and
/// `[bh, n, dv]` values, all rounded to f16 first; output `[bh, m, dv]`.
/// `bias(slice, row, key)` is added to the scores unrounded, as the kernel
/// adds it in f32; a row whose denominator is zero comes out as zeros, as
/// after `Where(IsNaN(p), 0, p)`.
fn reference(
    q: &[f32],
    kt: &[f32],
    v: &[f32],
    [bh, m, n, d, dv]: [usize; 5],
    bias: impl Fn(usize, usize, usize) -> f64,
) -> Vec<f64> {
    let r = |x: &f32| f64::from(round_f16(*x));
    let (q, kt, v): (Vec<f64>, Vec<f64>, Vec<f64>) = (
        q.iter().map(r).collect(),
        kt.iter().map(r).collect(),
        v.iter().map(r).collect(),
    );
    let mut out = Vec::with_capacity(bh * m * dv);
    for s in 0..bh {
        let (qb, kb, vb) = (s * m * d, s * d * n, s * n * dv);
        for row in 0..m {
            let scores: Vec<f64> = (0..n)
                .map(|j| {
                    (0..d)
                        .map(|k| q[qb + row * d + k] * kt[kb + k * n + j])
                        .sum::<f64>()
                        + bias(s, row, j)
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::MIN, f64::max);
            let exps: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for c in 0..dv {
                let acc: f64 = (0..n).map(|j| exps[j] * v[vb + j * dv + c]).sum();
                out.push(if sum == 0.0 { 0.0 } else { acc / sum });
            }
        }
    }
    out
}

/// Runs `graph(heads_inside)` and returns the output shape, its values and the
/// FlashAttention pipelines compiled, as `operation/tactic`.
fn run(
    context: &VkContext,
    heads_inside: bool,
    feeds: [(&str, &HostTensor); 3],
) -> (Vec<i64>, Vec<f32>, Vec<String>) {
    let executor = Executor::new(context, graph(heads_inside)).expect("executor");
    let outputs = executor
        .run(
            feeds
                .iter()
                .map(|(name, t)| (*name, Tensor::Host((*t).clone())))
                .collect(),
        )
        .expect("run");
    let y = outputs.host("y").expect("output");
    let result = (y.shape.clone(), y.to_f32().unwrap());
    outputs.finish();
    let kernels = executor
        .cache()
        .pipeline_keys()
        .iter()
        .filter(|key| key.operation().starts_with("FlashAttention"))
        .map(|key| format!("{}/{}", key.operation(), key.tactic_id().unwrap_or("")))
        .collect();
    (result.0, result.1, kernels)
}

/// `[bh, m, d]`, `[bh, d, n]`, `[bh, n, dv]` against the f64 reference, with
/// keys that grow along the sequence so the running maximum moves from tile
/// to tile.
fn check(bh: usize, m: usize, n: usize, d: usize, dv: usize, kernel: &str) {
    opt_in();
    let q = values(bh * m * d, 1, 1.0);
    let kt: Vec<f32> = values(bh * d * n, 2, 1.0)
        .iter()
        .enumerate()
        .map(|(i, x)| x + (i % n) as f32 * 0.02)
        .collect();
    let v = values(bh * n * dv, 3, 2.0);
    let expected = reference(&q, &kt, &v, [bh, m, n, d, dv], |_, _, _| 0.0);
    let shape = |dims: [usize; 3]| dims.iter().map(|&x| x as i64).collect::<Vec<_>>();
    let tensors = [
        HostTensor::from_f32(shape([bh, m, d]), &q),
        HostTensor::from_f32(shape([bh, d, n]), &kt),
        HostTensor::from_f32(shape([bh, n, dv]), &v),
    ];
    let context = VkContext::new().expect("Vulkan context");
    let (out_shape, got, kernels) = run(
        &context,
        false,
        [
            ("q", &tensors[0]),
            ("kt", &tensors[1]),
            ("v_in", &tensors[2]),
        ],
    );
    assert_eq!(kernels, [kernel]);
    assert_eq!(out_shape, shape([bh, m, dv]));
    let bound = 1e-3 * 2.0;
    let worst = got
        .iter()
        .zip(&expected)
        .map(|(g, e)| (f64::from(*g) - e).abs())
        .fold(0.0, f64::max);
    if kernel.starts_with("FlashAttention_f16") {
        assert!(worst <= bound, "max |Δ| {worst:e} above {bound:e}");
    }
}

fn coop_f16(context: &VkContext) -> bool {
    context.coop_f16 && context.subgroup_size == 32
}

/// The f16 kernel the runtime picks for `n` keys of heads of `d`.
fn f16_kernel(context: &VkContext, n: usize, d: usize, suffix: &str) -> String {
    let wg = if context.coop_f16_wg && n <= WG_MAX_KEYS {
        "_wg"
    } else {
        ""
    };
    format!("FlashAttention_f16{wg}/d{d}_dv{d}{suffix}")
}

#[test]
fn f16_kernel_matches_reference_on_partial_tiles() {
    let context = VkContext::new().expect("Vulkan context");
    if !coop_f16(&context) {
        eprintln!("skipped: no f16 cooperative matrix on subgroups of 32");
        return;
    }
    for d in [64, 32] {
        // neither 100 queries nor 77 keys is a multiple of the 64-wide tiles
        check(3, 100, 77, d, d, &f16_kernel(&context, 77, d, ""));
        // several key tiles, one query row past a block
        check(2, 65, 300, d, d, &f16_kernel(&context, 300, d, ""));
        // past the workgroup-scope kernel's keys, with a partial last tile
        let n = WG_MAX_KEYS + 70;
        check(1, 65, n, d, d, &f16_kernel(&context, n, d, ""));
    }
}

#[test]
fn f16_switch_leaves_other_heads_on_the_f32_kernel() {
    opt_in();
    let context = VkContext::new().expect("Vulkan context");
    let bc = onnx_vulkan_core::shaders::flash_attention::key_tile(
        48,
        48,
        context.compute_limits().max_shared_memory_bytes,
    )
    .unwrap();
    drop(context);
    // heads of 48: the f32 kernel, held to its own 2e-5 in flash_attention.rs
    check(
        2,
        40,
        50,
        48,
        48,
        &format!("FlashAttention/d48_dv48_bc{bc}"),
    );
}

#[test]
fn f16_kernel_folds_the_head_transposes() {
    opt_in();
    let context = VkContext::new().expect("Vulkan context");
    if !coop_f16(&context) {
        eprintln!("skipped: no f16 cooperative matrix on subgroups of 32");
        return;
    }
    // SAM 3's window (heads of 64) and detector (heads of 32) attention heads
    // with a partial query block: values [B, n, H, DV] and output
    // [B, m, H, DV] addressed in place, against the same kernel on explicitly
    // transposed tensors, bit for bit
    for (h, d) in [(4usize, 64usize), (8, 32)] {
        for n in [576, WG_MAX_KEYS + 70] {
            folded_heads_match_explicit_transposes(&context, h, d, n);
        }
    }
}

fn folded_heads_match_explicit_transposes(context: &VkContext, h: usize, d: usize, n: usize) {
    let (b, m) = (2usize, 130usize);
    let dims = |x: [usize; 4]| x.iter().map(|&v| v as i64).collect::<Vec<_>>();
    let q = HostTensor::from_f32(dims([b, h, m, d]), &values(b * h * m * d, 11, 1.0));
    let kt = HostTensor::from_f32(dims([b, h, d, n]), &values(b * h * d * n, 12, 1.0));
    let v = HostTensor::from_f32(dims([b, n, h, d]), &values(b * n * h * d, 13, 2.0));
    let feeds = [("q", &q), ("kt", &kt), ("v_in", &v)];

    let mut explicit = graph(true);
    explicit.outputs.extend(["v".into(), "heads".into()]);
    let executor = Executor::new(context, explicit).expect("executor");
    let outputs = executor
        .run(
            feeds
                .iter()
                .map(|(name, t)| (*name, Tensor::Host((*t).clone())))
                .collect(),
        )
        .expect("run");
    let want = outputs.host("y").expect("output").to_f32().unwrap();
    outputs.finish();

    let (shape, got, kernels) = run(context, true, feeds);
    assert_eq!(kernels, [f16_kernel(context, n, d, "")]);
    assert_eq!(shape, dims([b, m, h, d]));
    let differing = got
        .iter()
        .zip(&want)
        .filter(|(g, w)| g.to_bits() != w.to_bits())
        .count();
    assert_eq!(differing, 0, "{differing} of {} elements differ", got.len());
}

#[test]
fn f16_kernel_adds_a_bias_and_zeroes_masked_rows() {
    opt_in();
    let context = VkContext::new().expect("Vulkan context");
    if !coop_f16(&context) {
        eprintln!("skipped: no f16 cooperative matrix on subgroups of 32");
        return;
    }
    for d in [32, 64] {
        // one bias per slice, as SAM 3's detector query attention, and one
        // for every slice, as its text cross-attention mask
        for shared in [false, true] {
            for n in [150, WG_MAX_KEYS + 70] {
                check_biased(&context, d, n, shared);
            }
        }
    }
}

/// `softmax(q · kt + bias)` with the NaN cleanup, on `[3, 100, d]` queries and
/// `n` keys, against the f64 reference. The bias is `[3, 100, n]`, or
/// `[100, n]` broadcast over the slices when `shared`; rows 0 and 70 are
/// `-inf` (zero outputs after the cleanup) and every third key elsewhere is
/// `f32::MIN`.
fn check_biased(context: &VkContext, d: usize, n: usize, shared: bool) {
    let (bh, m) = (3usize, 100usize);
    let q = values(bh * m * d, 21, 1.0);
    let kt = values(bh * d * n, 22, 1.0);
    let v = values(bh * n * d, 23, 2.0);
    let slices = if shared { 1 } else { bh };
    let mut bias = values(slices * m * n, 24, 2.0);
    for (i, x) in bias.iter_mut().enumerate() {
        let (row, key) = ((i / n) % m, i % n);
        if row == 0 || row == 70 {
            *x = f32::NEG_INFINITY;
        } else if key % 3 == 0 {
            *x = f32::MIN;
        }
    }
    let at = |s: usize, row: usize, key: usize| {
        f64::from(bias[(if shared { 0 } else { s }) * m * n + row * n + key])
    };
    let expected = reference(&q, &kt, &v, [bh, m, n, d, d], at);

    let zero = InitializerIr {
        dtype: 1,
        shape: vec![],
        data: 0f32.to_le_bytes().to_vec(),
    };
    let graph = GraphIr {
        inputs: vec!["q".into(), "kt".into(), "v".into(), "bias".into()],
        outputs: vec!["y".into()],
        nodes: vec![
            node("MatMul", &["q", "kt"], &["scores"]),
            node("Add", &["scores", "bias"], &["biased"]),
            node("Softmax", &["biased"], &["probs"]),
            node("IsNaN", &["probs"], &["is_nan"]),
            node("Where", &["is_nan", "zero", "probs"], &["cleaned"]),
            node("MatMul", &["cleaned", "v"], &["y"]),
        ],
        initializers: [("zero".to_owned(), zero)].into(),
        ..Default::default()
    };
    let shape = |dims: &[usize]| dims.iter().map(|&x| x as i64).collect::<Vec<_>>();
    let bias_shape = if shared {
        shape(&[m, n])
    } else {
        shape(&[bh, m, n])
    };
    let feeds = [
        ("q", HostTensor::from_f32(shape(&[bh, m, d]), &q)),
        ("kt", HostTensor::from_f32(shape(&[bh, d, n]), &kt)),
        ("v", HostTensor::from_f32(shape(&[bh, n, d]), &v)),
        ("bias", HostTensor::from_f32(bias_shape, &bias)),
    ];
    let executor = Executor::new(context, graph).expect("executor");
    let outputs = executor
        .run(
            feeds
                .into_iter()
                .map(|(name, t)| (name, Tensor::Host(t)))
                .collect(),
        )
        .expect("run");
    let got = outputs.host("y").expect("output").to_f32().unwrap();
    outputs.finish();
    let kernels: Vec<String> = executor
        .cache()
        .pipeline_keys()
        .iter()
        .filter(|key| key.operation().starts_with("FlashAttention"))
        .map(|key| format!("{}/{}", key.operation(), key.tactic_id().unwrap_or("")))
        .collect();
    assert_eq!(kernels, [f16_kernel(context, n, d, "_bias")]);
    // `f64::max` skips NaN: a NaN output would pass the bound below
    assert!(got.iter().all(|x| x.is_finite()), "non-finite outputs");
    let bound = 1e-3 * 2.0;
    let worst = got
        .iter()
        .zip(&expected)
        .map(|(g, e)| (f64::from(*g) - e).abs())
        .fold(0.0, f64::max);
    assert!(
        worst <= bound,
        "d {d}, n {n}, shared {shared}: max |Δ| {worst:e} above {bound:e}"
    );
    let zeros = (0..bh)
        .flat_map(|s| [0, 70].map(|row| (s * m + row) * d))
        .all(|start| got[start..start + d].iter().all(|&x| x == 0.0));
    assert!(zeros, "masked rows are not zero");
}
