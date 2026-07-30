//! `GroupQueryAttention` (com.microsoft): fused rotary + grouped-query
//! attention + KV-cache append.
//!
//! Five dispatches, deliberately unfused. The op is stateful in the runtime
//! sense — `past_*` in, `present_*` out — but nothing here owns state: the
//! cache is rebuilt as a fresh tensor on every call (`GQA_past` copies the
//! past, `GQA_pack` appends the new step). That wastes a full cache copy per
//! node per token and is the price of being testable: a pure function of its
//! inputs can be diffed against the CPU EP one node at a time. The in-place
//! append belongs to the generation runtime, not to the kernel.
//!
//! Layouts, with `nh` query heads, `kvh` key/value heads and head size `H`:
//!
//! | tensor | shape |
//! |---|---|
//! | `query` / `key` / `value` | `[b, s, n·H]` (n = `nh` or `kvh`) |
//! | `past_key` / `past_value` | `[b, kvh, past, H]` |
//! | `present_key` / `present_value` | `[b, kvh, total, H]`, `total = past + s` |
//! | scores / probabilities | `[b, nh, s, total]` |
//! | output | `[b, s, nh·H]` |
//!
//! The key/value kernels take the time extent of `present_*` **twice**: `total`
//! bounds the loop, `stride` addresses the buffer. They are equal here — the
//! cache is exactly as long as it is full — and the pair exists for the
//! resident cache that replaces this one. A buffer that survives across tokens
//! has to be laid out for the longest sequence it will ever hold, because the
//! row stride of `[b, kvh, T, H]` is `T`: let `T` grow with the cache and every
//! token already written changes address. Splitting the two now keeps the
//! stateless path bit-identical (`stride == total`) while the addressing is
//! already the one the resident path needs.
//!
//! The scores tensor is materialized: at `s = total = 512` and four heads it
//! is 4 MB, and having it lets the softmax be the existing row kernel instead
//! of an online rescan. A flash-attention formulation removes it and is the
//! obvious next kernel — after the arithmetic is proven.

pub const ROTARY_BINDINGS: u32 = 5;
pub const ROTARY_PUSH_BYTES: u32 = 32;
pub const PACK_BINDINGS: u32 = 4;
pub const PACK_PUSH_BYTES: u32 = 40;
pub const PAST_BINDINGS: u32 = 2;
pub const PAST_PUSH_BYTES: u32 = 20;
pub const SCORES_BINDINGS: u32 = 4;
pub const SCORES_PUSH_BYTES: u32 = 52;
pub const OUT_BINDINGS: u32 = 3;
pub const OUT_PUSH_BYTES: u32 = 36;

/// Threads per workgroup for every kernel in this module.
pub const WG: u32 = 256;

/// Masked-out score. `Softmax` subtracts the row max first, so this underflows
/// to a zero probability; a row is never entirely masked because the query
/// always attends to itself.
pub const NEG_INF: &str = "-3.4028235e38";

/// Standalone `RotaryEmbedding` (com.microsoft): rotate in place, positions
/// read from `position_ids` instead of derived from the cache length.
///
/// This is the same rotation `PACK` fuses, and it is a separate kernel because
/// the two differ in everything around the arithmetic: the layout is unchanged
/// (input shape is output shape, no `[b, s, n·H]` → `[b, n, t, H]` transpose),
/// and the position of a token is a *value* — qwen2.5-VL's decoder builds it
/// with a `Range`, and mRoPE means it is not `past + seq`. `pos` arrives as
/// i32 because WGSL has no 64-bit scalar; the conversion happens on the host,
/// where the tensor already lives (`Range` is a host op).
///
/// `bnsh` selects the input layout: `[b, n, s, H]` when set, `[b, s, n·H]`
/// otherwise — the schema admits both and the shape is what discriminates.
/// Channels at or past `rotary_embedding_dim` are copied, which is how a
/// partial rotation leaves the rest of the head alone.
pub const ROTARY: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> pos: array<i32>;
@group(0) @binding(2) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(3) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

struct Push {
    count: u32, h: u32, s: u32, n: u32,
    rot_half: u32, cache_half: u32, bnsh: u32, gx: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let d = i % pc.h;
    var rest = i / pc.h;
    var seq = 0u;
    var batch = 0u;
    if (pc.bnsh != 0u) {
        seq = rest % pc.s;
        batch = (rest / pc.s) / pc.n;
    } else {
        rest = rest / pc.n;
        seq = rest % pc.s;
        batch = rest / pc.s;
    }

    // past `rotary_embedding_dim` the channel is untouched
    if (d >= 2u * pc.rot_half) {
        out[i] = x[i];
        return;
    }
    let p = u32(max(pos[batch * pc.s + seq], 0));
    let j = select(d - pc.rot_half, d, d < pc.rot_half);
    let c = cos_cache[p * pc.cache_half + j];
    let sn = sin_cache[p * pc.cache_half + j];
    if (d < pc.rot_half) {
        out[i] = x[i] * c - x[i + pc.rot_half] * sn;
    } else {
        out[i] = x[i] * c + x[i - pc.rot_half] * sn;
    }
}
"#;

/// `[b, s, n·H]` → `[b, n, dst_len, H]` at time offset `dst_off`, applying
/// rotary embedding when `rotary != 0`.
///
/// Non-interleaved (half-split) layout only: channel `j` of the first half
/// pairs with `j + H/2`, not with its neighbour. The position of token `seq`
/// is `pos_off + seq`, and `pos_off` is the past length — for a decode step
/// (`s = 1`) that is the only thing distinguishing one token from the next.
pub const PACK: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

struct Push {
    count: u32, s: u32, n: u32, h: u32,
    dst_len: u32, dst_off: u32, pos_off: u32, rot_half: u32,
    rotary: u32, gx: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let d = i % pc.h;
    var rest = i / pc.h;
    let head = rest % pc.n;
    rest = rest / pc.n;
    let seq = rest % pc.s;
    let batch = rest / pc.s;

    let src = ((batch * pc.s + seq) * pc.n + head) * pc.h + d;
    var v = x[src];
    if (pc.rotary != 0u) {
        let pos = pc.pos_off + seq;
        // j indexes the rotation pair; the partner sits half a head away
        let j = select(d - pc.rot_half, d, d < pc.rot_half);
        let c = cos_cache[pos * pc.rot_half + j];
        let sn = sin_cache[pos * pc.rot_half + j];
        if (d < pc.rot_half) {
            v = v * c - x[src + pc.rot_half] * sn;
        } else {
            v = v * c + x[src - pc.rot_half] * sn;
        }
    }
    dst[((batch * pc.n + head) * pc.dst_len + pc.dst_off + seq) * pc.h + d] = v;
}
"#;

/// `[b, n, past_len, H]` → the first `past_len` time steps of a destination of
/// row stride `stride`. A plain copy, but a strided one: batch and head are
/// contiguous in both, the time axis is not.
///
/// The dispatch disappears entirely once the cache is resident, because source
/// and destination are then the same memory.
pub const PAST: &str = r#"
@group(0) @binding(0) var<storage, read> past: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Push { count: u32, past_len: u32, stride: u32, h: u32, gx: u32 }
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let d = i % pc.h;
    let rest = i / pc.h;
    let t = rest % pc.past_len;
    // batch and head share one index: both layouts order them the same way
    let bh = rest / pc.past_len;
    dst[(bh * pc.stride + t) * pc.h + d] = past[i];
}
"#;

/// `scores[b, nh, s, total] = scale · q·kᵀ + bias`, masked.
///
/// Two masks compose. Causal: query `sq` sits at absolute position
/// `past + sq` and cannot see beyond it. Sliding window: with
/// `window >= 0` it additionally cannot see further back than `window`
/// positions, which is what makes gemma3's 22 local layers differ from its 4
/// global ones. The bound is `pq - pk >= window`, i.e. exactly `window`
/// visible keys including the query's own position.
///
/// That last `=` is measured, not assumed. gemma3 agrees with the CPU EP to
/// 1e-5 at every cache length up to 511 and diverges by 5e-1 at 512 — the
/// first length where a `window = 512` layer has anything to mask — which
/// places the boundary on the key `window` positions back, not `window + 1`.
pub const SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

// `total` is this step's key count, `keys` the row extent of the score buffer.
// They differ against a resident cache, where the scores cover the cache's whole
// physical extent to keep the grid step-invariant — so `keys` indexes the score
// row and `total` still indexes the attention bias. The bias buffer is padded to
// the cache as well, so that its size stops moving with the token, but the
// padding is all at the end: the rows the graph wrote stay `total` apart.
struct Push {
    count: u32, nh: u32, kvh: u32, h: u32,
    s: u32, total: u32, past: u32, scale: f32,
    window: i32, has_bias: u32, stride: u32, keys: u32, gx: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let t = i % pc.keys;
    var rest = i / pc.keys;
    let sq = rest % pc.s;
    rest = rest / pc.s;
    let head = rest % pc.nh;
    let batch = rest / pc.nh;

    let pq = pc.past + sq;
    var masked = t > pq;
    if (pc.window >= 0 && i32(pq - t) >= pc.window) { masked = true; }
    if (masked) {
        scores[i] = -3.4028235e38;
        return;
    }

    // grouped query: several query heads share one key/value head
    let hkv = head / (pc.nh / pc.kvh);
    let qb = ((batch * pc.nh + head) * pc.s + sq) * pc.h;
    let kb = ((batch * pc.kvh + hkv) * pc.stride + t) * pc.h;
    var acc = 0.0;
    for (var d = 0u; d < pc.h; d = d + 1u) {
        acc = acc + q[qb + d] * k[kb + d];
    }
    acc = acc * pc.scale;
    if (pc.has_bias != 0u) {
        // attention bias is [b, 1, s, ·]: broadcast over heads
        acc = acc + bias[(batch * pc.s + sq) * pc.total + t];
    }
    scores[i] = acc;
}
"#;

/// `out[b, s, nh·H] = probs[b, nh, s, total] · v[b, kvh, total, H]`.
///
/// One thread per output element, so consecutive threads read consecutive `H`
/// of `v`; the probability is a broadcast read shared by the whole head.
pub const OUT: &str = r#"
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

// `total` is how many keys this step has, `keys` how many the probability
// buffer is laid out for: the two differ against a resident cache, where the
// score and probability scratch cover the cache's whole physical extent so the
// grid does not change from token to token. The padded probabilities are exactly
// zero, but the value cache behind them is not initialized, so the sum stops at
// `total` rather than multiplying zero by whatever is there.
struct Push {
    count: u32, nh: u32, kvh: u32, h: u32,
    s: u32, total: u32, stride: u32, keys: u32, gx: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let d = i % pc.h;
    var rest = i / pc.h;
    let sq = rest % pc.s;
    rest = rest / pc.s;
    let head = rest % pc.nh;
    let batch = rest / pc.nh;

    let hkv = head / (pc.nh / pc.kvh);
    let pb = ((batch * pc.nh + head) * pc.s + sq) * pc.keys;
    let vb = (batch * pc.kvh + hkv) * pc.stride * pc.h + d;
    var acc = 0.0;
    for (var t = 0u; t < pc.total; t = t + 1u) {
        acc = acc + probs[pb + t] * v[vb + t * pc.h];
    }
    out[((batch * pc.s + sq) * pc.nh + head) * pc.h + d] = acc;
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for source in [
            super::ROTARY,
            super::PACK,
            super::PAST,
            super::SCORES,
            super::OUT,
        ] {
            vk_compute::compile_wgsl(source).expect("valid attention shader");
        }
    }
}
