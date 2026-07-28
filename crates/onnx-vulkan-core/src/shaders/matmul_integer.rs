//! Shared shaders and dispatch layouts for `MatMulInteger`.

pub const TILE_SIZE: u32 = 16;
pub const PACK_BINDINGS: u32 = 2;
pub const PACK_PUSH_BYTES: u32 = 12;
pub const MATMUL_BINDINGS: u32 = 5;
pub const MATMUL_PUSH_BYTES: u32 = 24;
pub const FLIP_BINDINGS: u32 = 2;
pub const FLIP_PUSH_BYTES: u32 = 4;
pub const FLIP_KEY: &str = "MMI_flip";
/// Mask that turns a signed byte into the unsigned byte `s + 128`, and a signed
/// zero point into the matching unsigned one. `u - zu == s - zs`, so the matmul
/// is unchanged — see [`FLIP_BYTES`].
pub const SIGN_FLIP_WORD: u32 = 0x8080_8080;
pub const SIGN_FLIP_BYTE: u32 = 0x80;

/// Copies a u8 tensor flipping the sign bit of every byte.
///
/// `int8` operands are brought to `uint8` once, ahead of the matmul, instead of
/// on every tile load: the cooperative matrix path cannot flip anything (there
/// is no bitwise op on a cooperative matrix, and the hardware combination is
/// `u8 × u8`), and for a constant weight the cost is paid once per session.
/// The identity is the same one the tiled kernel used inline: adding 128 to
/// both the byte and its zero point leaves `byte - zero_point` unchanged.
pub const FLIP_BYTES: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Push { words: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < pc.words) {
        dst[gid.x] = src[gid.x] ^ 0x80808080u;
    }
}
"#;

/// Transposes raw u8 B `[K, N]` into `[N, K/4]` packed u32, flipping the sign
/// bit of every byte when `flip` is `SIGN_FLIP_WORD` (see [`FLIP_BYTES`]).
pub const PACK_B: &str = r#"
@group(0) @binding(0) var<storage, read> braw: array<u32>;
@group(0) @binding(1) var<storage, read_write> bp: array<u32>;

struct Push { k: u32, n: u32, flip: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;      // column of B (0..n)
    let i = gid.y;        // K4 index (0..k/4)
    let k4 = pc.k / 4u;
    if (col >= pc.n || i >= k4) {
        return;
    }
    var w = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let idx = (i * 4u + j) * pc.n + col;
        let byte = (braw[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu;
        w = w | (byte << (j * 8u));
    }
    bp[col * k4 + i] = w ^ pc.flip;
}
"#;

/// Tiled int8 matmul: 16×16 block, shared memory reuse along K.
///
/// Each u32 word contains four bytes; zero point is loaded at boundaries so
/// contribution becomes zero after subtraction.
///
/// ONNX allows `uint8` and `int8` for A and B independently. The sign is not a
/// branch here: signed bytes are brought to unsigned by adding 128 (xor `0x80`,
/// see [`FLIP_BYTES`]), and the caller says where that flip still has to
/// happen. `a_byte_flip` is `SIGN_FLIP_WORD` when A is signed and reaches the
/// kernel as-is, `0` when the caller already flipped it; `a_zp_xor` /
/// `b_zp_xor` are `SIGN_FLIP_BYTE` whenever the corresponding operand is
/// signed, flipped ahead of time or not, because the zero point in the buffer
/// is always the original one. B's bytes are always flipped by the pack, so it
/// has no `byte_flip` field.
const MATMUL_TEMPLATE: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<u32>;    // [M, K/4] (A raw, K%4==0)
@group(0) @binding(1) var<storage, read> b: array<u32>;    // [N, K/4] packed
@group(0) @binding(2) var<storage, read> azp: array<u32>;  // byte 0 = a_zero_point
@group(0) @binding(3) var<storage, read> bzp: array<u32>;  // byte 0 = b_zero_point
@group(0) @binding(4) var<storage, read_write> out: array<i32>; // [M, N]

struct Push { m: u32, k4: u32, n: u32, a_byte_flip: u32, a_zp_xor: u32, b_zp_xor: u32 }
var<immediate> pc: Push;

const TILE = 16u;
var<workgroup> as_tile: array<u32, 256>; // [ty][tx] = 16*16
var<workgroup> bs_tile: array<u32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let ty = lid.y;
    let tx = lid.x;
    let row = wid.y * TILE + ty;
    let col = wid.x * TILE + tx;

    // Signed bytes become unsigned by adding 128 (= xor 0x80); the same
    // applies to the zero point: the difference `byte - zp` is unchanged. For
    // A the flip is applied at tile load, so the inner loop stays identical to
    // the unsigned case (no per-element branch); for B it is already done at
    // pack time.
    let a_flip = pc.a_byte_flip;
    let a_zp_u = (azp[0] & 0xffu) ^ pc.a_zp_xor;
    let b_zp_u = (bzp[0] & 0xffu) ^ pc.b_zp_xor;
    let a_zp_word = a_zp_u * 0x01010101u;
    let b_zp_word = b_zp_u * 0x01010101u;
    let a_zp4 = vec4<i32>(i32(a_zp_u));
    let b_zp4 = vec4<i32>(i32(b_zp_u));

    DECL
    let ntiles = (pc.k4 + TILE - 1u) / TILE;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let kw = t * TILE + tx;
        if (row < pc.m && kw < pc.k4) {
            as_tile[ty * TILE + tx] = a[row * pc.k4 + kw] ^ a_flip;
        } else {
            as_tile[ty * TILE + tx] = a_zp_word;
        }
        let bcol = wid.x * TILE + ty;
        if (bcol < pc.n && kw < pc.k4) {
            bs_tile[ty * TILE + tx] = b[bcol * pc.k4 + kw];
        } else {
            bs_tile[ty * TILE + tx] = b_zp_word;
        }
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            let aw = as_tile[ty * TILE + i];
            let bw = bs_tile[tx * TILE + i];
            ACC
        }
        workgroupBarrier();
    }
    if (row < pc.m && col < pc.n) {
        out[row * pc.n + col] = FIN;
    }
}
"#;

/// Inner loop on ALUs: unpack the four bytes, subtract the zero point,
/// `dot` on `vec4<i32>`. The portable path, and the only one on hardware
/// without `VK_KHR_shader_integer_dot_product`.
const VECTOR_DECL: &str = "var acc: i32 = 0;";
const VECTOR_ACC: &str = "acc = acc + dot(vec4<i32>(unpack4xU8(aw)) - a_zp4, \
                          vec4<i32>(unpack4xU8(bw)) - b_zp4);";
const VECTOR_FIN: &str = "acc";

/// Inner loop on the integer dot-product unit (`OpUDot`, one instruction for
/// four products plus their sum).
///
/// The zero point cannot be subtracted per byte before the dot, so the product
/// is decomposed:
///
/// ```text
/// Σ(a−za)(b−zb) = Σab − za·Σb − zb·Σa + n·za·zb
/// ```
///
/// Three `dot4U8Packed` per iteration instead of one `dot` — but no unpacking
/// and no `vec4` subtraction, and the two row/column sums come from the same
/// unit. `n` counts **every** byte iterated, padding included: there both
/// operands equal their zero point, so those terms are zero in the sum on the
/// left and the identity holds unchanged.
const PACKED_DECL: &str = "var acc: i32 = 0; var sa: i32 = 0; var sb: i32 = 0;";
const PACKED_ACC: &str = "acc = acc + i32(dot4U8Packed(aw, bw)); \
                          sa = sa + i32(dot4U8Packed(aw, 0x01010101u)); \
                          sb = sb + i32(dot4U8Packed(bw, 0x01010101u));";
const PACKED_FIN: &str = "acc - i32(a_zp_u) * sb - i32(b_zp_u) * sa \
                          + i32(ntiles * TILE * 4u) * i32(a_zp_u) * i32(b_zp_u)";

/// Variant cache key: two distinct pipelines for the same op.
pub const VECTOR_KEY: &str = "MMI_matmul";
pub const PACKED_KEY: &str = "MMI_matmul_dot4";

/// Bindings and push constants of the cooperative matrix variant. Same bindings
/// as the WGSL matmul; the push constants differ (`k` in bytes, no sign flags).
pub const COOP_BINDINGS: u32 = 5;
pub const COOP_PUSH_BYTES: u32 = 20;
/// Tile of the output block, fixed by the 16×16×K cooperative matrix shapes.
pub const COOP_TILE: u32 = 16;

/// A cooperative matrix build of the integer matmul: SPIR-V compiled offline
/// from `glsl/matmul_integer_coop.comp` (see `scripts/build-glsl.sh`), because
/// naga's WGSL frontend has neither 8-bit scalars nor non-square cooperative
/// matrices.
pub struct CoopVariant {
    /// Distinct cache key per variant, otherwise the pipelines collide.
    pub key: &'static str,
    /// K consumed by one cooperative multiply; the matrix K must be a multiple.
    pub k_tile: u32,
    /// Workgroup size: one subgroup, computing one 16×16 output tile.
    pub workgroup: u32,
    words: &'static [u8],
}

impl CoopVariant {
    /// SPIR-V words, decoded from the little-endian bytes embedded in the binary.
    pub fn spirv(&self) -> Vec<u32> {
        self.words
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// One entry per compiled `(k_tile, workgroup, acc_signed)` triple.
const COOP_VARIANTS: &[(CoopVariant, bool)] = &[
    (
        CoopVariant {
            key: "MMI_matmul_coop_k32",
            k_tile: 32,
            workgroup: 32,
            words: include_bytes!("spv/matmul_integer_coop_k32_sg32_u32.spv"),
        },
        false,
    ),
    (
        CoopVariant {
            key: "MMI_matmul_coop_k16",
            k_tile: 16,
            workgroup: 64,
            words: include_bytes!("spv/matmul_integer_coop_k16_sg64_i32.spv"),
        },
        true,
    ),
];

/// Picks the cooperative matrix variant matching what the driver advertises,
/// or `None` if no compiled shader fits. The largest `k_tile` wins: fewer
/// cooperative multiplies over the same K.
pub fn coop_variant(
    device_combos: &[vk_compute::CoopMatU8],
    subgroup_size: u32,
) -> Option<&'static CoopVariant> {
    COOP_VARIANTS
        .iter()
        .filter(|(v, acc_signed)| {
            v.workgroup == subgroup_size
                && device_combos
                    .iter()
                    .any(|c| c.k_tile == v.k_tile && c.acc_signed == *acc_signed)
        })
        .map(|(v, _)| v)
        .max_by_key(|v| v.k_tile)
}

/// Whether a given problem can run on the cooperative matrix variant.
///
/// Tiles are clamped rather than padded (an out-of-range `coopMatLoad` is
/// undefined without `cooperativeMatrixRobustBufferAccess`), so the matrix must
/// be at least one full tile in each direction, and K must be an exact multiple
/// of the cooperative K.
///
/// `a_signed_in_memory` rules out the case the shader cannot fix on its own:
/// the hardware combination is `u8 × u8` and a cooperative matrix has no
/// bitwise operations, so an operand still holding signed bytes has to be
/// flipped by an earlier pass ([`FLIP_BYTES`], or the flip built into
/// [`PACK_B`] for B) or fall back to the WGSL kernel.
pub fn coop_applies(
    v: &CoopVariant,
    m: usize,
    k: usize,
    n: usize,
    a_signed_in_memory: bool,
) -> bool {
    !a_signed_in_memory
        && m >= COOP_TILE as usize
        && n >= COOP_TILE as usize
        && k.is_multiple_of(v.k_tile as usize)
}

/// Integer matmul source. `packed_dot` requires
/// `VK_KHR_shader_integer_dot_product` on the device: without it, the SPIR-V
/// declares a missing capability and the pipeline fails to create.
pub fn matmul(packed_dot: bool) -> String {
    let (decl, acc, fin) = if packed_dot {
        (PACKED_DECL, PACKED_ACC, PACKED_FIN)
    } else {
        (VECTOR_DECL, VECTOR_ACC, VECTOR_FIN)
    };
    MATMUL_TEMPLATE
        .replace("DECL", decl)
        .replace("ACC", acc)
        // `FIN` must be replaced last: it appears inside the other two
        .replace("FIN", fin)
}

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for source in [
            super::PACK_B.to_string(),
            super::matmul(false),
            super::matmul(true),
        ] {
            vk_compute::compile_wgsl(&source).expect("shader MatMulInteger valido");
        }
    }
}
