//! `GatherBlockQuantized` against a scalar reference.
//!
//! Same concern as `matmul_nbits.rs`: the op is a packing convention, and a
//! swapped nibble still produces a plausible embedding. Two things are pinned
//! here that the matmul test cannot pin — that the **row** index selects the
//! table row (not a column) and that the block index advances along the
//! gathered row — so the nibble ramp varies with both row and column.

use onnx_vulkan_core::host_ops::{FLOAT, HostTensor, INT64, UINT8};
use onnx_vulkan_core::{
    AttrValue, ExecutionEnv, GraphIr, KernelCache, NodeIr, Tensor, execute, is_implemented_node,
};
use std::collections::HashMap;
use vk_compute::VkContext;

/// Packs `[rows][cols]` nibbles into `[rows, cols/2]` bytes, element `2i` low.
fn pack_table(rows: usize, cols: usize, nibble: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let mut out = vec![0u8; rows * cols / 2];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols / 2 + c / 2] |= (nibble(r, c) & 0xF) << (4 * (c % 2));
        }
    }
    out
}

/// Packs one 4-bit zero point per block into `[rows, ceil(n_blocks/2)]` bytes.
fn pack_zero_points(rows: usize, n_blocks: usize, zp: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let row_bytes = n_blocks.div_ceil(2);
    let mut out = vec![0u8; rows * row_bytes];
    for r in 0..rows {
        for b in 0..n_blocks {
            out[r * row_bytes + b / 2] |= (zp(r, b) & 0xF) << (4 * (b % 2));
        }
    }
    out
}

struct Case {
    rows: usize,
    cols: usize,
    block_size: usize,
    indices: Vec<i64>,
    index_shape: Vec<i64>,
}

fn run_case(case: Case) {
    let Case {
        rows,
        cols,
        block_size,
        indices,
        index_shape,
    } = case;
    let n_blocks = cols / block_size;

    // every nibble value appears, and neighbouring columns differ, so a
    // low/high swap changes the result
    let nibble = |r: usize, c: usize| ((r * 5 + c * 3) % 16) as u8;
    let zero_point = |r: usize, b: usize| ((r + b * 7) % 16) as u8;
    let scale = |r: usize, b: usize| ((r * 3 + b) % 11) as f32 / 32.0 - 0.15;

    let table = pack_table(rows, cols, nibble);
    let zp = pack_zero_points(rows, n_blocks, zero_point);
    let scales: Vec<f32> = (0..rows)
        .flat_map(|r| (0..n_blocks).map(move |b| scale(r, b)))
        .collect();

    let mut want = vec![0.0f32; indices.len() * cols];
    for (j, &index) in indices.iter().enumerate() {
        let r = if index < 0 {
            (index + rows as i64) as usize
        } else {
            index as usize
        };
        for c in 0..cols {
            let b = c / block_size;
            want[j * cols + c] =
                (f32::from(nibble(r, c)) - f32::from(zero_point(r, b))) * scale(r, b);
        }
    }

    let node = NodeIr {
        domain: "com.microsoft".into(),
        op: "GatherBlockQuantized".into(),
        opset: 1,
        name: "gbq".into(),
        inputs: ["table", "indices", "scales", "zp"]
            .iter()
            .map(|s| (*s).to_string())
            .collect(),
        outputs: vec!["out".into()],
        attrs: [
            ("bits", AttrValue::Int(4)),
            ("block_size", AttrValue::Int(block_size as i64)),
            ("gather_axis", AttrValue::Int(0)),
            ("quantize_axis", AttrValue::Int(1)),
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
        "table",
        Tensor::Host(HostTensor::new(
            UINT8,
            vec![rows as i64, (cols / 2) as i64],
            table,
        )),
    );
    env.set(
        "indices",
        Tensor::Host(HostTensor::new(
            INT64,
            index_shape.clone(),
            indices.iter().flat_map(|v| v.to_le_bytes()).collect(),
        )),
    );
    env.set(
        "scales",
        Tensor::Host(HostTensor::new(
            FLOAT,
            vec![rows as i64, n_blocks as i64],
            scales.iter().flat_map(|s| s.to_le_bytes()).collect(),
        )),
    );
    env.set(
        "zp",
        Tensor::Host(HostTensor::new(
            UINT8,
            vec![rows as i64, n_blocks.div_ceil(2) as i64],
            zp,
        )),
    );
    execute(&ir, &mut env).expect("graph execution");
    let got = env.host("out").expect("output on host").to_f32().unwrap();
    env.finish();

    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "element {i}: {g} != {w} (rows={rows} cols={cols} block={block_size})"
        );
    }
}

#[test]
fn decode_step_gathers_one_embedding_row() {
    // gemma3-1b's geometry, with a table small enough to reference by hand
    run_case(Case {
        rows: 64,
        cols: 1152,
        block_size: 32,
        indices: vec![37],
        index_shape: vec![1, 1],
    });
}

#[test]
fn a_prefill_gathers_several_rows_including_repeats() {
    run_case(Case {
        rows: 32,
        cols: 128,
        block_size: 32,
        indices: vec![0, 31, 7, 7, 1],
        index_shape: vec![1, 5],
    });
}

#[test]
fn negative_indices_wrap_from_the_end() {
    run_case(Case {
        rows: 16,
        cols: 64,
        block_size: 16,
        indices: vec![-1, -16, 3],
        index_shape: vec![3],
    });
}

#[test]
fn an_odd_block_count_misaligns_the_zero_point_rows() {
    // 5 blocks is 3 bytes of zero points per row, so consecutive rows start at
    // byte offsets 0, 3, 6 -- the unaligned path, as in `matmul_nbits`
    run_case(Case {
        rows: 9,
        cols: 80,
        block_size: 16,
        indices: vec![8, 0, 5],
        index_shape: vec![3],
    });
}
