//! Fused `MatMul → Softmax → MatMul` attention without the scores tensor.
//!
//! The rewrite in `crate::rewrite::fuse_attention` turns the three nodes into
//! one when nothing else reads the scores or the probabilities. SAM 3's
//! detector encoder is what it is for: six self-attention layers over 5184
//! image tokens, each materializing `[1, 8, 5184, 5184]` f32 scores (860 MB)
//! and probabilities (another 860 MB). Those two allocations alone cost
//! ~53 ms per run, before the 3.4 GB of traffic per layer.
//!
//! Layouts, all contiguous with identical batch dimensions `bh`:
//!
//! | operand | shape |
//! |---|---|
//! | `a` (queries, scale already applied) | `[bh, m, D]` |
//! | `b` (keys, transposed) | `[bh, D, n]` |
//! | `v` (values) | `[bh, n, DV]` |
//! | `out` | `[bh, m, DV]` |
//!
//! `v` and `out` may instead keep the heads inside the sequence, as
//! `[B, n, H, DV]` and `[B, m, H, DV]` with `bh = B·H`: the push constants
//! `v_heads` and `out_heads` give `H`, and `1` is the layout above. That is
//! the `Transpose(·, [0, 2, 1, 3])` multi-head attention writes on both sides,
//! folded into the addressing.
//!
//! The `_bias` builds add a fifth binding, `bias`, to the scores before the
//! softmax: the score of query `r` and key `j` in slice `bh` reads
//! `bias[(bh / heads) * bias_outer + (bh % heads) * bias_head +
//! r * bias_row + j]`, so a zero stride broadcasts that dimension. With
//! `nan_to_zero`, a
//! row whose softmax denominator is not positive (every key at `-inf`, or a
//! NaN score) is written as zeros, as `Where(IsNaN(p), 0, p)` before the
//! second product would leave it.
//!
//! One workgroup computes [`BR`] query rows of one `bh` slice, four threads
//! per row, walking the keys a tile of [`KEY_TILES`] at a time with the
//! online softmax: a
//! running row maximum `m`, a running denominator `l`, and the output
//! accumulator rescaled by `exp(m_old - m_new)` whenever the maximum moves.
//! The arithmetic differs from the unfused path only in summation order.

/// Query rows per workgroup.
pub const BR: u32 = 64;
/// Keys per tile, preferred first. With 64 keys, heads of 64 need 53 248 bytes
/// of workgroup memory, above the 49 152 an RTX 4070 grants in Vulkan (there
/// is no opt-in as in CUDA); 32 keys halve the tile buffers.
pub const KEY_TILES: [u32; 2] = [64, 32];
pub const BINDINGS: u32 = 4;
/// With a bias: `a`, `b`, `v`, `out`, `bias`.
pub const BIAS_BINDINGS: u32 = 5;
/// `m`, `n`, `v_heads`, `out_heads`, `nan_to_zero`, then the bias addressing:
/// `heads`, `bias_outer`, `bias_head`, `bias_row` (ignored without a bias).
pub const PUSH_BYTES: u32 = 36;
/// Largest query/key head size the generated shader accepts: the query row
/// lives in registers.
pub const MAX_D: u32 = 64;
/// Largest value head size: each of the four threads of a row accumulates a
/// quarter of the output row in registers.
pub const MAX_DV: u32 = 64;
/// The value head size must be a multiple of this: each thread's quarter of
/// the output row is a whole number of `vec4`s.
pub const DV_MULTIPLE: u32 = 16;

/// The f16 cooperative-matrix kernel, `glsl/flash_attention_coop.comp`, built
/// for heads of 32 and of 64 ([`COOP_HEADS`]): same bindings, push constants
/// and layouts, products in f16 with f32 accumulation. A declared precision
/// change, never chosen on its own: see `interp::attention_fp16`.
const COOP_SPV_D32: &[u8] = include_bytes!("spv/flash_attention_coop_f16_d32.spv");
const COOP_SPV_D64: &[u8] = include_bytes!("spv/flash_attention_coop_f16_d64.spv");
const COOP_SPV_D32_BIAS: &[u8] = include_bytes!("spv/flash_attention_coop_f16_d32_bias.spv");
const COOP_SPV_D64_BIAS: &[u8] = include_bytes!("spv/flash_attention_coop_f16_d64_bias.spv");

/// Query, key and value head sizes (all equal) of the f16 kernel.
pub const COOP_HEADS: [u32; 2] = [32, 64];

/// The SPIR-V words of the f16 kernel for heads of `d`, one of [`COOP_HEADS`],
/// with or without the bias binding.
pub fn coop_spirv(d: u32, bias: bool) -> Vec<u32> {
    let bytes = match (d, bias) {
        (32, false) => COOP_SPV_D32,
        (64, false) => COOP_SPV_D64,
        (32, true) => COOP_SPV_D32_BIAS,
        (64, true) => COOP_SPV_D64_BIAS,
        _ => panic!("no f16 FlashAttention kernel for heads of {d}"),
    };
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The f16 workgroup-scope kernel, `glsl/flash_attention_wg.comp`, on
/// `VK_NV_cooperative_matrix2` (`VkContext::coop_f16_wg`): the cooperative
/// kernel's bindings, push constants, tiles and precision, with the softmax
/// done by matrix reductions and per-element operations. SPIR-V 1.6.
const WG_SPV_D32: &[u8] = include_bytes!("spv/flash_attention_wg_f16_d32.spv");
const WG_SPV_D64: &[u8] = include_bytes!("spv/flash_attention_wg_f16_d64.spv");
const WG_SPV_D32_BIAS: &[u8] = include_bytes!("spv/flash_attention_wg_f16_d32_bias.spv");
const WG_SPV_D64_BIAS: &[u8] = include_bytes!("spv/flash_attention_wg_f16_d64_bias.spv");

/// Most keys the workgroup-scope kernel takes over from the cooperative one.
/// On an RTX 4070 it is faster up to 576 keys (window attention 0.458 against
/// 0.629 ms, 32 text keys 0.025 against 0.058 ms) and slower at 5184 (global
/// 3.63 against 3.22 ms, heads of 32 1.16 against 0.81 ms): its Kᵀ and V
/// loads convert f32 in registers, where the cooperative kernel stages them
/// in shared memory ahead of use.
pub const WG_MAX_KEYS: usize = 1024;

/// The SPIR-V words of the workgroup-scope f16 kernel for heads of `d`, one
/// of [`COOP_HEADS`], with or without the bias binding. Dispatched as the
/// cooperative kernel: [`COOP_BR`] query rows per workgroup.
pub fn wg_spirv(d: u32, bias: bool) -> Vec<u32> {
    let bytes = match (d, bias) {
        (32, false) => WG_SPV_D32,
        (64, false) => WG_SPV_D64,
        (32, true) => WG_SPV_D32_BIAS,
        (64, true) => WG_SPV_D64_BIAS,
        _ => panic!("no workgroup-scope FlashAttention kernel for heads of {d}"),
    };
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Query rows per workgroup of the f16 kernel: four subgroups of 32, sixteen
/// rows each. Also its keys per tile.
pub const COOP_BR: u32 = 64;
pub const COOP_SUBGROUP: u32 = 32;
pub const COOP_INVOCATIONS: u32 = 4 * COOP_SUBGROUP;
/// Workgroup memory of the f16 kernel for heads of `d`: f16 tiles of Kᵀ
/// (`d` rows) and V (64 rows) at a pitch of 72, and 64 rows of f32 scores at a
/// pitch of 72 that the f16 probabilities overwrite.
pub const fn coop_shared_bytes(d: u32) -> u32 {
    2 * (d * 72 + 64 * 72) + 4 * 64 * 72
}

/// `vec4`s per key row in workgroup memory: the head padded to a multiple of 4,
/// plus one so the four keys a quarter-warp reads start in different banks.
fn key_pitch(d: u32) -> u32 {
    d.div_ceil(4) + 1
}

/// Workgroup memory the shader declares for head sizes `d` and `dv` and `bc`
/// keys per tile.
pub fn shared_bytes(d: u32, dv: u32, bc: u32) -> u32 {
    16 * bc * key_pitch(d) + 4 * (bc * dv + BR * (bc + 4) + 2 * 256)
}

/// The largest key tile whose workgroup memory fits `max_shared_bytes`.
pub fn key_tile(d: u32, dv: u32, max_shared_bytes: u32) -> Option<u32> {
    KEY_TILES
        .into_iter()
        .find(|&bc| shared_bytes(d, dv, bc) <= max_shared_bytes)
}

/// WGSL for query/key head size `d`, value head size `dv` and `bc` keys per
/// tile, one of [`KEY_TILES`], with the `bias` binding or without.
///
/// Every loop over a head, a tile or the thread's keys is unrolled here, in
/// Rust, into named scalars and `vec4`s. Written as WGSL loops over function-
/// scope arrays, the same kernel ran SAM 3's encoder attention at 0.6 TFLOPS on
/// an RTX 4070, 2.6× slower than the unfused path it replaced: the arrays were
/// not promoted to registers.
pub fn source(d: u32, dv: u32, bc: u32, bias: bool) -> String {
    assert!(
        (1..=MAX_D).contains(&d)
            && dv.is_multiple_of(DV_MULTIPLE)
            && (DV_MULTIPLE..=MAX_DV).contains(&dv)
            && KEY_TILES.contains(&bc)
    );
    let d4 = d.div_ceil(4);
    let pitch = key_pitch(d);
    let dv4 = dv / 4;
    // vec4 output columns per thread
    let cols = dv4 / 4;
    let keys = bc / 4;
    // vec4s per probability row
    let p_pitch = keys + 1;
    let mut s = String::new();
    let mut w = |line: String| {
        s.push_str(&line);
        s.push('\n');
    };
    w(format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
{bias_binding}
struct Push {{
    m: u32, n: u32, v_heads: u32, out_heads: u32, nan_to_zero: u32,
    heads: u32, bias_outer: u32, bias_head: u32, bias_row: u32,
}}
var<immediate> pc: Push;

const D: u32 = {d}u;
const DV: u32 = {dv}u;
const NEG_MAX: f32 = -3.4028235e38;

// keys of the tile: key j's head is ks[j * {pitch} ..], zero-padded to a vec4
var<workgroup> ks: array<vec4<f32>, {ks_len}>;
// values of the tile: key j's row is vs[j * {dv4} ..]
var<workgroup> vs: array<vec4<f32>, {vs_len}>;
// probabilities of the tile, one row of {bc} per query padded by one vec4:
// key j's is ps[r * {p_pitch} + j / 4][j % 4]
var<workgroup> ps: array<vec4<f32>, {ps_len}>;
var<workgroup> red_max: array<f32, 256>;
var<workgroup> red_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {{
    let lid = lid3.x;
    // four threads per query row; thread `part` owns keys part, part + 4, ..
    // of every tile and the output vec4s part, part + 4, ..: interleaved, the
    // four threads of a row read 64 contiguous bytes of a value row, where
    // contiguous quarters of a 64-wide row would hit the same banks
    let r = lid / 4u;
    let part = lid % 4u;
    let bh = wid.y;
    let row = wid.x * {BR}u + r;
    let b_base = bh * D * pc.n;
    // value row j of slice bh = (b, h) is v[((b * n + j) * v_heads + h) * DV ..]
    let v_base = ((bh / pc.v_heads) * pc.n * pc.v_heads + bh % pc.v_heads) * DV;
    let v_row = pc.v_heads * DV;
    // rows past `m` compute on a copy of the last row and store nothing
    let qrow = bh * pc.m * D + min(row, pc.m - 1u) * D;
    let bias_base = (bh / pc.heads) * pc.bias_outer + (bh % pc.heads) * pc.bias_head
        + min(row, pc.m - 1u) * pc.bias_row;"#,
        bias_binding = if bias {
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;\n"
        } else {
            ""
        },
        ks_len = bc * pitch,
        vs_len = bc * dv4,
        ps_len = BR * p_pitch,
    ));
    let q_elem = |k: u32| {
        if k < d {
            format!("a[qrow + {k}u]")
        } else {
            "0.0".to_owned()
        }
    };
    for c in 0..d4 {
        let e: Vec<String> = (0..4).map(|i| q_elem(4 * c + i)).collect();
        w(format!("    let q{c} = vec4<f32>({});", e.join(", ")));
    }
    for u in 0..cols {
        w(format!("    var o{u} = vec4<f32>(0.0);"));
    }
    w(format!(
        r#"    var m = NEG_MAX;
    var l = 0.0;
    let tiles = (pc.n + {bc}u - 1u) / {bc}u;
    for (var t = 0u; t < tiles; t = t + 1u) {{
        let key0 = t * {bc}u;
        // keys: b is [D, n], read coalesced along the key; the padding of the
        // last vec4 is zeroed, workgroup memory starts undefined
        for (var e = lid; e < {k_elems}u; e = e + 256u) {{
            let k = e / {bc}u;
            let j = e % {bc}u;
            var x = 0.0;
            if (k < D && key0 + j < pc.n) {{
                x = b[b_base + k * pc.n + key0 + j];
            }}
            ks[j * {pitch}u + k / 4u][k % 4u] = x;
        }}
        // values: rows of DV, read coalesced along the channel
        for (var e = lid; e < {vs_len}u; e = e + 256u) {{
            let j = e / {dv4}u;
            let at = v_base + (key0 + j) * v_row + (e % {dv4}u) * 4u;
            var x = vec4<f32>(0.0);
            if (key0 + j < pc.n) {{
                x = vec4<f32>(v[at], v[at + 1u], v[at + 2u], v[at + 3u]);
            }}
            vs[e] = x;
        }}
        workgroupBarrier();
        let tail = pc.n - key0;
        let kb = part * {pitch}u;"#,
        k_elems = bc * d4 * 4,
        vs_len = bc * dv4,
    ));
    for i in 0..keys {
        let terms: Vec<String> = (0..d4)
            .map(|c| format!("dot(q{c}, ks[kb + {}u])", 4 * i * pitch + c))
            .collect();
        w(format!("        var s{i} = {};", terms.join(" + ")));
        if bias {
            w(format!(
                "        if (part + {o}u < tail) {{ s{i} = s{i} + bias[bias_base + key0 + part + {o}u]; }}",
                o = 4 * i
            ));
        }
    }
    w("        var local_max = NEG_MAX;".to_owned());
    for i in 0..keys {
        w(format!(
            "        if (part + {}u < tail) {{ local_max = max(local_max, s{i}); }}",
            4 * i
        ));
    }
    w(format!(
        r#"        red_max[lid] = local_max;
        workgroupBarrier();
        let quad = r * 4u;
        let tile_max = max(max(red_max[quad], red_max[quad + 1u]),
                           max(red_max[quad + 2u], red_max[quad + 3u]));
        let m_new = max(m, tile_max);
        let alpha = exp(m - m_new);
        var local_sum = 0.0;
        let pb = r * {p_pitch}u;"#
    ));
    for i in 0..keys {
        w(format!(
            "        var p{i} = 0.0;\n        if (part + {o}u < tail) {{ p{i} = exp(s{i} - m_new); }}\n        ps[pb + {i}u][part] = p{i};\n        local_sum = local_sum + p{i};",
            o = 4 * i
        ));
    }
    w(r#"        red_sum[lid] = local_sum;
        workgroupBarrier();
        let tile_sum = (red_sum[quad] + red_sum[quad + 1u])
                     + (red_sum[quad + 2u] + red_sum[quad + 3u]);
        l = l * alpha + tile_sum;
        m = m_new;"#
        .to_owned());
    for u in 0..cols {
        w(format!("        o{u} = o{u} * alpha;"));
    }
    w("        let vb = part;".to_owned());
    for j in 0..bc {
        if j % 4 == 0 {
            w(format!("        let p4_{} = ps[pb + {}u];", j / 4, j / 4));
        }
        w(format!(
            "        let pv{j} = vec4<f32>(p4_{}.{});",
            j / 4,
            ["x", "y", "z", "w"][j as usize % 4]
        ));
        for u in 0..cols {
            w(format!(
                "        o{u} = fma(pv{j}, vs[vb + {}u], o{u});",
                j * dv4 + 4 * u
            ));
        }
    }
    w(r#"        workgroupBarrier();
    }
    if (row < pc.m) {
        let inv = 1.0 / l;
        // `!(l > 0)` is also true for a NaN denominator
        let clear = pc.nan_to_zero != 0u && !(l > 0.0);
        let heads = pc.out_heads;
        let ob = ((bh / heads) * pc.m * heads + row * heads + bh % heads) * DV + part * 4u;"#
        .to_owned());
    for u in 0..cols {
        for (i, comp) in ["x", "y", "z", "w"].iter().enumerate() {
            w(format!(
                "        out[ob + {}u] = select(o{u}.{comp} * inv, 0.0, clear);",
                16 * u + i as u32
            ));
        }
    }
    w("    }\n}".to_owned());
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sources_compile() {
        for (d, dv) in [(32, 32), (1, 16), (64, 64), (5, 48)] {
            for bc in KEY_TILES {
                for bias in [false, true] {
                    vk_compute::compile_wgsl(&source(d, dv, bc, bias))
                        .expect("valid flash attention shader");
                }
            }
        }
    }

    #[test]
    fn coop_kernels_fit_the_workgroup_memory_of_an_rtx_4070() {
        assert_eq!(coop_shared_bytes(64), 36_864);
        assert_eq!(coop_shared_bytes(32), 32_256);
        const { assert!(coop_shared_bytes(64) <= 49_152) };
        for d in COOP_HEADS {
            for bias in [false, true] {
                assert!(!coop_spirv(d, bias).is_empty());
            }
        }
        for spv in [
            COOP_SPV_D32,
            COOP_SPV_D64,
            COOP_SPV_D32_BIAS,
            COOP_SPV_D64_BIAS,
        ] {
            assert_eq!(spv.len() % 4, 0);
        }
    }

    #[test]
    fn key_tile_shrinks_to_fit_the_device() {
        // the RTX 4070 grants 48 KiB of workgroup memory in Vulkan
        assert_eq!(shared_bytes(32, 32, 64), 36_864);
        assert_eq!(key_tile(32, 32, 49_152), Some(64));
        // SAM 3's vision encoder heads: 53 248 bytes with 64 keys
        assert_eq!(shared_bytes(64, 64, 64), 53_248);
        assert_eq!(shared_bytes(64, 64, 32), 28_160);
        assert_eq!(key_tile(64, 64, 49_152), Some(32));
        assert_eq!(key_tile(64, 64, 16_384), None);
    }
}
