//! Shared shaders and dispatch layouts for row normalizations
//! (`Softmax`, `LayerNormalization`).

pub const SOFTMAX_BINDINGS: u32 = 2;
pub const SOFTMAX_PUSH_BYTES: u32 = 16;
pub const LAYERNORM_BINDINGS: u32 = 4;
pub const LAYERNORM_PUSH_BYTES: u32 = 12;

/// Numerically stable f32 Softmax on **any axis**: one workgroup per
/// row, max → sum of exps → normalization, with shared memory reductions.
///
/// Row is non-contiguous in general: `c` elements separated by `inner`
/// (product of dimensions past axis). With last axis `inner = 1` falling back
/// to contiguous case. Rows are indexed on a 2D grid
/// (`gx` per grid row) because `rows` easily exceeds the limit of
/// 65535 workgroups per dimension.
pub const SOFTMAX: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

struct Push { c: u32, inner: u32, rows: u32, gx: u32 }
var<immediate> pc: Push;

var<workgroup> sred: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * pc.gx + wid.x;
    if (row >= pc.rows) { return; }
    // row r → (outer, inner_idx): base = outer * c * inner + inner_idx
    let base = (row / pc.inner) * pc.c * pc.inner + (row % pc.inner);

    // 1) max di riga
    var m = -3.4028235e38;
    var i = lid.x;
    while (i < pc.c) {
        m = max(m, x[base + i * pc.inner]);
        i = i + 256u;
    }
    sred[lid.x] = m;
    workgroupBarrier();
    var s = 128u;
    while (s > 0u) {
        if (lid.x < s) {
            sred[lid.x] = max(sred[lid.x], sred[lid.x + s]);
        }
        workgroupBarrier();
        s = s / 2u;
    }
    let row_max = sred[0];
    workgroupBarrier();

    // 2) somma exp
    var sum = 0.0;
    i = lid.x;
    while (i < pc.c) {
        sum = sum + exp(x[base + i * pc.inner] - row_max);
        i = i + 256u;
    }
    sred[lid.x] = sum;
    workgroupBarrier();
    s = 128u;
    while (s > 0u) {
        if (lid.x < s) {
            sred[lid.x] = sred[lid.x] + sred[lid.x + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    let inv_sum = 1.0 / sred[0];

    // 3) scrittura
    i = lid.x;
    while (i < pc.c) {
        out[base + i * pc.inner] = exp(x[base + i * pc.inner] - row_max) * inv_sum;
        i = i + 256u;
    }
}
"#;

/// f32 LayerNormalization: one workgroup per row, sum/sum-of-squares reduction
/// in shared memory → mean and variance → optional scale and bias (`has_bias`).
pub const LAYERNORM: &str = r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scale: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

struct Push { c: u32, eps: f32, has_bias: u32 }
var<immediate> pc: Push;

var<workgroup> ssum: array<f32, 256>;
var<workgroup> ssq: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let base = row * pc.c;
    var sum = 0.0;
    var sq = 0.0;
    var i = lid.x;
    while (i < pc.c) {
        let v = x[base + i];
        sum = sum + v;
        sq = sq + v * v;
        i = i + 256u;
    }
    ssum[lid.x] = sum;
    ssq[lid.x] = sq;
    workgroupBarrier();
    var s = 128u;
    while (s > 0u) {
        if (lid.x < s) {
            ssum[lid.x] = ssum[lid.x] + ssum[lid.x + s];
            ssq[lid.x] = ssq[lid.x] + ssq[lid.x + s];
        }
        workgroupBarrier();
        s = s / 2u;
    }
    let mean = ssum[0] / f32(pc.c);
    let variance = ssq[0] / f32(pc.c) - mean * mean;
    let inv_std = inverseSqrt(variance + pc.eps);
    i = lid.x;
    while (i < pc.c) {
        var v = (x[base + i] - mean) * inv_std * scale[i];
        if (pc.has_bias != 0u) {
            v = v + bias[i];
        }
        out[base + i] = v;
        i = i + 256u;
    }
}
"#;

#[cfg(test)]
mod tests {
    #[test]
    fn sources_compile() {
        for source in [super::SOFTMAX, super::LAYERNORM] {
            vk_compute::compile_wgsl(source).expect("shader di normalizzazione valido");
        }
    }
}
