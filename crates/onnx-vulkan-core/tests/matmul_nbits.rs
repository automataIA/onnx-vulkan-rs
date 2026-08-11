//! `MatMulNBits` against a scalar reference.
//!
//! The whole op is a packing convention, so what the test has to pin down is
//! the packing: which nibble of a byte holds which `k`, and which nibble of the
//! zero-point row belongs to which block. Get either backwards and the result
//! is still a plausible-looking matmul — of the wrong matrix. The reference
//! below therefore unpacks the same way the ONNX Runtime CPU kernel does,
//! **element `2i` in the low nibble**, and the values are chosen so a swap
//! cannot cancel: the nibbles are a ramp, not a constant.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, UINT8};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

/// Packs `[n][k]` nibbles into `[N, n_blocks, block_size/2]` bytes.
fn pack_weights(
    n: usize,
    k: usize,
    block_size: usize,
    nibble: impl Fn(usize, usize) -> u8,
) -> Vec<u8> {
    let n_blocks = k.div_ceil(block_size);
    let mut out = vec![0u8; n * n_blocks * block_size / 2];
    for col in 0..n {
        for kk in 0..k {
            let byte = col * n_blocks * block_size / 2 + kk / 2;
            out[byte] |= (nibble(col, kk) & 0xF) << (4 * (kk % 2));
        }
    }
    out
}

/// Packs one 4-bit zero point per block into `[N, ceil(n_blocks/2)]` bytes.
fn pack_zero_points(n: usize, n_blocks: usize, zp: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let row = n_blocks.div_ceil(2);
    let mut out = vec![0u8; n * row];
    for col in 0..n {
        for b in 0..n_blocks {
            out[col * row + b / 2] |= (zp(col, b) & 0xF) << (4 * (b % 2));
        }
    }
    out
}

struct Case {
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
}

fn run_case(case: Case) {
    let Case {
        m,
        k,
        n,
        block_size,
    } = case;
    let n_blocks = k.div_ceil(block_size);

    // every nibble value 0..16 appears, and the pattern differs between
    // neighbouring `k` so a low/high nibble swap changes the product
    let nibble = |col: usize, kk: usize| ((col * 7 + kk * 3) % 16) as u8;
    let zero_point = |col: usize, b: usize| ((col + b * 5) % 16) as u8;
    let scale = |col: usize, b: usize| ((col * 3 + b) % 11) as f32 / 32.0 - 0.15;
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 13 % 29) as f32 - 14.0) / 8.0)
        .collect();

    let quant = pack_weights(n, k, block_size, nibble);
    let zp = pack_zero_points(n, n_blocks, zero_point);
    let scales: Vec<f32> = (0..n)
        .flat_map(|col| (0..n_blocks).map(move |b| scale(col, b)))
        .collect();

    let mut want = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                let b = kk / block_size;
                let w =
                    (f32::from(nibble(col, kk)) - f32::from(zero_point(col, b))) * scale(col, b);
                acc += a[row * k + kk] * w;
            }
            want[row * n + col] = acc;
        }
    }

    let node = NodeIr {
        domain: "com.microsoft".into(),
        op: "MatMulNBits".into(),
        opset: 1,
        name: "mmnb".into(),
        inputs: ["a", "quant", "scales", "zp"]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        outputs: vec!["out".into()],
        attrs: [
            ("K", AttrValue::Int(k as i64)),
            ("N", AttrValue::Int(n as i64)),
            ("bits", AttrValue::Int(4)),
            ("block_size", AttrValue::Int(block_size as i64)),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect(),
    };
    assert!(is_implemented_node(&node), "the node must be claimable");

    let ir = GraphIr {
        nodes: vec![node],
        initializers: HashMap::new(),
        inputs: Vec::new(),
        outputs: vec!["out".into()],
        ..Default::default()
    };

    let context = VkContext::new().expect("Vulkan context");
    let cache = KernelCache::new(&context);
    let mut env = ExecutionEnv::new(&cache, &ir.initializers);
    env.set(
        "a",
        Tensor::Host(HostTensor::from_f32(vec![m as i64, k as i64], &a)),
    );
    env.set(
        "quant",
        Tensor::Host(HostTensor::new(
            UINT8,
            vec![n as i64, n_blocks as i64, (block_size / 2) as i64],
            quant,
        )),
    );
    env.set(
        "scales",
        Tensor::Host(HostTensor::new(
            FLOAT,
            vec![(n * n_blocks) as i64],
            scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
        )),
    );
    env.set(
        "zp",
        Tensor::Host(HostTensor::new(
            UINT8,
            vec![n as i64, n_blocks.div_ceil(2) as i64],
            zp,
        )),
    );
    execute(&ir, &mut env).expect("graph execution");
    let got = env.host("out").expect("output on host").to_f32().unwrap();
    env.finish();

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.0),
            "element {i}: {g} != {w} (M={m} K={k} N={n} block={block_size})"
        );
    }
}

#[test]
fn decode_step_matches_the_unpacked_reference() {
    // gemma3-1b's smallest projection, at the one row a decode step has
    run_case(Case {
        m: 1,
        k: 1152,
        n: 256,
        block_size: 32,
    });
}

#[test]
fn several_rows_share_the_weight() {
    run_case(Case {
        m: 5,
        k: 256,
        n: 48,
        block_size: 32,
    });
}

#[test]
fn a_partial_last_block_is_not_read_past_k() {
    // K is not a multiple of block_size: the last block is padded in the file
    // and its tail must contribute nothing
    run_case(Case {
        m: 2,
        k: 100,
        n: 17,
        block_size: 32,
    });
}

/// `N` above `shaders::matmul_nbits::WIDE_MIN_N` routes a decode step to the
/// wide-column kernel — 64 columns per workgroup instead of one, whole blocks
/// read as `vec4<u32>` and unpacked with four accumulators — and this is what
/// says the two agree. Its reduction and its mapping from threads to columns are
/// both different, so nothing about the small-`N` cases above covers it.
#[test]
fn a_wide_decode_step_takes_the_wide_kernel() {
    run_case(Case {
        m: 1,
        k: 128,
        n: 4096,
        block_size: 32,
    });
}

/// The same kernel carries the unembedding widths, where it was the only claimed
/// form before the threshold was lowered to one. `N = 65536` is kept as a case
/// because its grid folds over two axes.
#[test]
fn an_unembedding_width_takes_the_wide_kernel() {
    run_case(Case {
        m: 1,
        k: 96,
        n: 65536,
        block_size: 32,
    });
}

/// The width thresholds apply to a decode step only: with more than one row the
/// shipped kernel runs whatever `N` is, and a wide prefill must not silently
/// take a kernel measured only at `M = 1`.
#[test]
fn a_wide_prefill_stays_on_the_row_kernel() {
    run_case(Case {
        m: 3,
        k: 128,
        n: 4096,
        block_size: 32,
    });
}

#[test]
fn an_odd_block_count_misaligns_the_zero_point_rows() {
    // 7 blocks means 4 bytes of zero points per column, so consecutive columns
    // start at byte offsets 0, 4, 8 -- aligned. 5 blocks means 3 bytes, which
    // is what actually exercises the unaligned path.
    run_case(Case {
        m: 1,
        k: 80,
        n: 33,
        block_size: 16,
    });
}
