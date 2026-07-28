//! The two variants of the integer matmul must agree, bit for bit.
//!
//! The kernel picks one at pipeline-compile time from
//! `VK_KHR_shader_integer_dot_product`, so on any given machine only one of
//! them ever runs — and the other stays untested exactly where it matters. The
//! packed path cannot subtract the zero point per byte before the dot product,
//! so it computes
//!
//! ```text
//! Σ(a−za)(b−zb) = Σab − za·Σb − zb·Σa + n·za·zb
//! ```
//!
//! which is an *algebraic rewrite*, not the same code with a faster
//! instruction. This test dispatches both shaders on the same buffers and
//! compares them with a CPU reference, so a mistake in that rewrite cannot hide
//! on hardware that takes the other branch.

use onnx_vulkan_core::shaders::matmul_integer::{
    COOP_BINDINGS, COOP_PUSH_BYTES, MATMUL_BINDINGS, MATMUL_PUSH_BYTES, PACK_B, PACK_BINDINGS,
    PACK_PUSH_BYTES, SIGN_FLIP_BYTE, SIGN_FLIP_WORD, TILE_SIZE, coop_applies, coop_variant, matmul,
};
use vk_compute::{VkContext, compile_wgsl};

const M: usize = 7;
const K: usize = 12; // multiple of 4, not of TILE: exercises padding
const N: usize = 5;

fn pseudo(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

fn as_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Runs a variant and returns [M, N].
fn run_variant(
    ctx: &VkContext,
    packed_dot: bool,
    a: &[u8],
    b_t: &[u8],
    azp: u8,
    bzp: u8,
) -> Vec<i32> {
    let a_words: Vec<u32> = a
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let b_words: Vec<u32> = b_t
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let buf_a = ctx
        .create_storage_buffer((a_words.len() * 4) as u64)
        .unwrap();
    let buf_b = ctx
        .create_storage_buffer((b_words.len() * 4) as u64)
        .unwrap();
    let buf_azp = ctx.create_storage_buffer(4).unwrap();
    let buf_bzp = ctx.create_storage_buffer(4).unwrap();
    let buf_out = ctx.create_storage_buffer((M * N * 4) as u64).unwrap();
    ctx.upload(&buf_a, &as_bytes(&a_words)).unwrap();
    ctx.upload(&buf_b, &as_bytes(&b_words)).unwrap();
    ctx.upload(&buf_azp, &[azp, 0, 0, 0]).unwrap();
    ctx.upload(&buf_bzp, &[bzp, 0, 0, 0]).unwrap();

    let mut push = Vec::with_capacity(MATMUL_PUSH_BYTES as usize);
    // m, k4, n, a_byte_flip, a_zp_xor, b_zp_xor — no signed operands here
    for v in [M as u32, (K / 4) as u32, N as u32, 0, 0, 0] {
        push.extend_from_slice(&v.to_le_bytes());
    }

    let spirv = compile_wgsl(&matmul(packed_dot)).expect("variant compilation");
    let pipeline = ctx
        .create_pipeline(&spirv, MATMUL_BINDINGS, MATMUL_PUSH_BYTES)
        .unwrap();
    ctx.dispatch(
        &pipeline,
        &[&buf_a, &buf_b, &buf_azp, &buf_bzp, &buf_out],
        &push,
        [
            (N as u32).div_ceil(TILE_SIZE),
            (M as u32).div_ceil(TILE_SIZE),
            1,
        ],
    )
    .unwrap();

    let bytes = ctx.download(&buf_out).unwrap();
    let out: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for buffer in [buf_a, buf_b, buf_azp, buf_bzp, buf_out] {
        ctx.destroy_buffer(buffer);
    }
    ctx.destroy_pipeline(pipeline);
    out
}

#[test]
fn both_variants_match_the_cpu_reference() {
    let ctx = VkContext::new().expect("contesto Vulkan");
    let a = pseudo(M * K, 11);
    let b = pseudo(K * N, 29);
    // non-zero zero points on both: with zero == 0 the decomposition
    // would be correct even if the correction terms were wrong
    let (azp, bzp) = (37u8, 211u8);

    // B transposed and packed along K, as `PACK_B` does
    let mut b_t = vec![0u8; N * K];
    for i in 0..K {
        for col in 0..N {
            b_t[col * K + i] = b[i * N + col];
        }
    }

    let mut want = vec![0i32; M * N];
    for (row, out) in want.chunks_exact_mut(N).enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = (0..K)
                .map(|k| {
                    (i32::from(a[row * K + k]) - i32::from(azp))
                        * (i32::from(b[k * N + col]) - i32::from(bzp))
                })
                .sum();
        }
    }

    let vector = run_variant(&ctx, false, &a, &b_t, azp, bzp);
    let packed = run_variant(&ctx, true, &a, &b_t, azp, bzp);

    assert_eq!(vector, want, "vector path");
    assert_eq!(packed, want, "dot4U8Packed path");
    assert_eq!(packed, vector, "the two variants must match");
}

/// The cooperative matrix variant against the same CPU reference.
///
/// Shapes are chosen to exercise what the shader does differently from the WGSL
/// kernels: M and N are not multiples of the 16×16 tile, so the last row and
/// column bands are computed by *clamped, overlapping* tiles rather than by
/// padding, and a mistake there shows up as wrong values in the overlap. K is a
/// multiple of 32 so that both the K=32 and the K=16 configurations apply.
///
/// Skipped, loudly, where the device has no cooperative matrix support (any CPU
/// runtime, so this never runs in a lavapipe CI): there is nothing to compare.
#[test]
fn coop_variant_matches_the_cpu_reference() {
    const CM: usize = 20;
    const CK: usize = 64;
    const CN: usize = 18;

    let ctx = VkContext::new().expect("contesto Vulkan");
    let Some(variant) = coop_variant(&ctx.coop_u8, ctx.subgroup_size) else {
        eprintln!(
            "SKIP: {} does not expose a usable u8 cooperative matrix \
             (coop_u8={:?}, subgroup={})",
            ctx.device_name, ctx.coop_u8, ctx.subgroup_size
        );
        return;
    };
    assert!(
        coop_applies(variant, CM, CK, CN, false),
        "le dimensioni del test devono ricadere nel percorso cooperative"
    );
    eprintln!("variante cooperative: {}", variant.key);

    let a = pseudo(CM * CK, 5);
    let b = pseudo(CK * CN, 17);
    let (azp, bzp) = (37u8, 211u8);

    let mut b_t = vec![0u8; CN * CK];
    for i in 0..CK {
        for col in 0..CN {
            b_t[col * CK + i] = b[i * CN + col];
        }
    }

    let mut want = vec![0i32; CM * CN];
    for (row, out) in want.chunks_exact_mut(CN).enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = (0..CK)
                .map(|k| {
                    (i32::from(a[row * CK + k]) - i32::from(azp))
                        * (i32::from(b[k * CN + col]) - i32::from(bzp))
                })
                .sum();
        }
    }

    let buf_a = ctx.create_storage_buffer((CM * CK) as u64).unwrap();
    let buf_b = ctx.create_storage_buffer((CN * CK) as u64).unwrap();
    let buf_azp = ctx.create_storage_buffer(4).unwrap();
    let buf_bzp = ctx.create_storage_buffer(4).unwrap();
    let buf_out = ctx.create_storage_buffer((CM * CN * 4) as u64).unwrap();
    ctx.upload(&buf_a, &a).unwrap();
    ctx.upload(&buf_b, &b_t).unwrap();
    ctx.upload(&buf_azp, &[azp, 0, 0, 0]).unwrap();
    ctx.upload(&buf_bzp, &[bzp, 0, 0, 0]).unwrap();

    let mut push = Vec::with_capacity(COOP_PUSH_BYTES as usize);
    // m, k, n, a_zp_xor, b_zp_xor
    for v in [CM as u32, CK as u32, CN as u32, 0, 0] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    let pipeline = ctx
        .create_pipeline(&variant.spirv(), COOP_BINDINGS, COOP_PUSH_BYTES)
        .unwrap();
    ctx.dispatch(
        &pipeline,
        &[&buf_a, &buf_b, &buf_azp, &buf_bzp, &buf_out],
        &push,
        [
            (CN as u32).div_ceil(TILE_SIZE),
            (CM as u32).div_ceil(TILE_SIZE),
            1,
        ],
    )
    .unwrap();

    let bytes = ctx.download(&buf_out).unwrap();
    let got: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for buffer in [buf_a, buf_b, buf_azp, buf_bzp, buf_out] {
        ctx.destroy_buffer(buffer);
    }
    ctx.destroy_pipeline(pipeline);

    assert_eq!(got, want, "percorso cooperative matrix");
}

/// The `int8` operand path, end to end: `PACK_B` flips the sign bit while it
/// transposes, and the kernel shifts the zero point by the same 128.
///
/// This is what makes cooperative matrices reachable for a signed weight at
/// all: the hardware combination is `u8 × u8` and a cooperative matrix has no
/// bitwise operations, so nothing can be flipped inside the kernel. Getting the
/// flip on the bytes but not on the zero point (or the reverse) shifts every
/// result by a multiple of 128·K, which a zero zero point would hide — hence a
/// non-zero, negative `bzp` here.
#[test]
fn signed_b_is_flipped_by_the_pack_and_matches_the_cpu_reference() {
    const SM: usize = 20;
    const SK: usize = 64;
    const SN: usize = 18;

    let ctx = VkContext::new().expect("contesto Vulkan");
    let Some(variant) = coop_variant(&ctx.coop_u8, ctx.subgroup_size) else {
        eprintln!("SKIP: {} non ha cooperative matrix u8", ctx.device_name);
        return;
    };

    let a = pseudo(SM * SK, 3);
    // B as `int8`: the same bytes reinterpreted with sign
    let b_raw = pseudo(SK * SN, 41);
    let (azp, bzp) = (37u8, -73i8);

    let mut want = vec![0i32; SM * SN];
    for (row, out) in want.chunks_exact_mut(SN).enumerate() {
        for (col, cell) in out.iter_mut().enumerate() {
            *cell = (0..SK)
                .map(|k| {
                    (i32::from(a[row * SK + k]) - i32::from(azp))
                        * (i32::from(b_raw[k * SN + col] as i8) - i32::from(bzp))
                })
                .sum();
        }
    }

    let buf_a = ctx.create_storage_buffer((SM * SK) as u64).unwrap();
    let buf_braw = ctx.create_storage_buffer((SK * SN) as u64).unwrap();
    let buf_bpacked = ctx.create_storage_buffer((SN * SK) as u64).unwrap();
    let buf_azp = ctx.create_storage_buffer(4).unwrap();
    let buf_bzp = ctx.create_storage_buffer(4).unwrap();
    let buf_out = ctx.create_storage_buffer((SM * SN * 4) as u64).unwrap();
    ctx.upload(&buf_a, &a).unwrap();
    ctx.upload(&buf_braw, &b_raw).unwrap();
    ctx.upload(&buf_azp, &[azp, 0, 0, 0]).unwrap();
    ctx.upload(&buf_bzp, &[bzp as u8, 0, 0, 0]).unwrap();

    // pack: transposes [K, N] → [N, K] and flips the sign bit
    let mut pack_push = Vec::with_capacity(PACK_PUSH_BYTES as usize);
    for v in [SK as u32, SN as u32, SIGN_FLIP_WORD] {
        pack_push.extend_from_slice(&v.to_le_bytes());
    }
    let pack = ctx
        .create_pipeline(
            &compile_wgsl(PACK_B).unwrap(),
            PACK_BINDINGS,
            PACK_PUSH_BYTES,
        )
        .unwrap();
    ctx.dispatch(
        &pack,
        &[&buf_braw, &buf_bpacked],
        &pack_push,
        [
            (SN as u32).div_ceil(TILE_SIZE),
            ((SK / 4) as u32).div_ceil(TILE_SIZE),
            1,
        ],
    )
    .unwrap();

    let mut push = Vec::with_capacity(COOP_PUSH_BYTES as usize);
    // m, k, n, a_zp_xor (A unsigned), b_zp_xor (B was int8)
    for v in [SM as u32, SK as u32, SN as u32, 0, SIGN_FLIP_BYTE] {
        push.extend_from_slice(&v.to_le_bytes());
    }
    let pipeline = ctx
        .create_pipeline(&variant.spirv(), COOP_BINDINGS, COOP_PUSH_BYTES)
        .unwrap();
    ctx.dispatch(
        &pipeline,
        &[&buf_a, &buf_bpacked, &buf_azp, &buf_bzp, &buf_out],
        &push,
        [
            (SN as u32).div_ceil(TILE_SIZE),
            (SM as u32).div_ceil(TILE_SIZE),
            1,
        ],
    )
    .unwrap();

    let bytes = ctx.download(&buf_out).unwrap();
    let got: Vec<i32> = bytes
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    for buffer in [buf_a, buf_braw, buf_bpacked, buf_azp, buf_bzp, buf_out] {
        ctx.destroy_buffer(buffer);
    }
    ctx.destroy_pipeline(pack);
    ctx.destroy_pipeline(pipeline);

    assert!(
        coop_applies(variant, SM, SK, SN, false),
        "the case must fall into the cooperative path"
    );
    assert_eq!(got, want, "B int8 flipped by the pack");
}
