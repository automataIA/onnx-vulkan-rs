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
//! The scores tensor is materialized: at `s = total = 512` and four heads it
//! is 4 MB, and having it lets the softmax be the existing row kernel instead
//! of an online rescan. A flash-attention formulation removes it and is the
//! obvious next kernel — after the arithmetic is proven.

pub const PACK_BINDINGS: u32 = 4;
pub const PACK_PUSH_BYTES: u32 = 40;
pub const PAST_BINDINGS: u32 = 2;
pub const PAST_PUSH_BYTES: u32 = 20;
pub const SCORES_BINDINGS: u32 = 4;
pub const SCORES_PUSH_BYTES: u32 = 44;
pub const OUT_BINDINGS: u32 = 3;
pub const OUT_PUSH_BYTES: u32 = 28;

/// Threads per workgroup for every kernel in this module.
pub const WG: u32 = 256;

/// Masked-out score. `Softmax` subtracts the row max first, so this underflows
/// to a zero probability; a row is never entirely masked because the query
/// always attends to itself.
pub const NEG_INF: &str = "-3.4028235e38";

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

/// `[b, n, past_len, H]` → the first `past_len` time steps of
/// `[b, n, total, H]`. A plain copy, but a strided one: batch and head are
/// contiguous in both, the time axis is not.
pub const PAST: &str = r#"
@group(0) @binding(0) var<storage, read> past: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Push { count: u32, past_len: u32, total: u32, h: u32, gx: u32 }
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
    dst[(bh * pc.total + t) * pc.h + d] = past[i];
}
"#;

/// `scores[b, nh, s, total] = scale · q·kᵀ + bias`, masked.
///
/// Two masks compose. Causal: query `sq` sits at absolute position
/// `past + sq` and cannot see beyond it. Sliding window: with
/// `window >= 0` it additionally cannot see further back than `window`
/// positions, which is what makes gemma3's 22 local layers differ from its 4
/// global ones. The bound is `pq - pk > window`, i.e. `window + 1` visible
/// keys including the query's own position — the convention ONNX Runtime's
/// CPU kernel uses, and the reason this is validated against it.
pub const SCORES: &str = r#"
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

struct Push {
    count: u32, nh: u32, kvh: u32, h: u32,
    s: u32, total: u32, past: u32, scale: f32,
    window: i32, has_bias: u32, gx: u32,
}
var<immediate> pc: Push;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = (wid.y * pc.gx + wid.x) * 256u + lid.x;
    if (i >= pc.count) { return; }
    let t = i % pc.total;
    var rest = i / pc.total;
    let sq = rest % pc.s;
    rest = rest / pc.s;
    let head = rest % pc.nh;
    let batch = rest / pc.nh;

    let pq = pc.past + sq;
    var masked = t > pq;
    if (pc.window >= 0 && i32(pq - t) > pc.window) { masked = true; }
    if (masked) {
        scores[i] = -3.4028235e38;
        return;
    }

    // grouped query: several query heads share one key/value head
    let hkv = head / (pc.nh / pc.kvh);
    let qb = ((batch * pc.nh + head) * pc.s + sq) * pc.h;
    let kb = ((batch * pc.kvh + hkv) * pc.total + t) * pc.h;
    var acc = 0.0;
    for (var d = 0u; d < pc.h; d = d + 1u) {
        acc = acc + q[qb + d] * k[kb + d];
    }
    acc = acc * pc.scale;
    if (pc.has_bias != 0u) {
        // attention bias is [b, 1, s, total]: broadcast over heads
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

struct Push { count: u32, nh: u32, kvh: u32, h: u32, s: u32, total: u32, gx: u32 }
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
    let pb = ((batch * pc.nh + head) * pc.s + sq) * pc.total;
    let vb = (batch * pc.kvh + hkv) * pc.total * pc.h + d;
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
        for source in [super::PACK, super::PAST, super::SCORES, super::OUT] {
            vk_compute::compile_wgsl(source).expect("valid attention shader");
        }
    }
}
