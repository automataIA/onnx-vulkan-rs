//! Measure f32 matmul tilings on the shapes that actually run.
//!
//! The four families below are the `MatMul` nodes ORT leaves in
//! `sam3_image_encoder.onnx` after it fuses every biased one into `Gemm`: the
//! attention products. Two of them have N=64, a single tile column for any tile
//! wider than 64, which is the reason not to reuse the `Gemm` geometry blindly.
//!
//! Each variant keeps a 256-thread workgroup and changes only the shape of the
//! per-thread micro-tile, so the comparison isolates the register blocking.
//! `MM`×`MN` outputs per thread cost `MM+MN` shared reads per `MM*MN` FMAs.
//!
//! Run: `cargo run --release -p vk-compute --example matmul_tiling`

use std::time::Instant;
use vk_compute::{VkContext, compile_wgsl};

/// (M, K, N, batch, how many nodes of this shape the model runs)
const SHAPES: &[(u32, u32, u32, u32, u32, &str)] = &[
    (576, 64, 576, 144, 28, "win QK^T"),
    (576, 576, 64, 144, 28, "win AV  "),
    (5184, 64, 5184, 16, 4, "glob QK^T"),
    (5184, 5184, 64, 16, 4, "glob AV  "),
];

/// (rows/thread, cols/thread, stage B as vec4). Tile is 16*MM × 16*MN.
const VARIANTS: &[(u32, u32, bool)] = &[
    (4, 4, false),
    (8, 4, false),
    (4, 4, true),
    (8, 4, true),
    (2, 4, true),
];

fn source(mm: u32, mn: u32, vec4_b: bool) -> String {
    let (tm, tn, ks) = (16 * mm, 16 * mn, 16u32);
    let mut decl = String::new();
    let mut inner = String::new();
    let mut store = String::new();
    for i in 0..mm {
        for j in 0..mn {
            decl += &format!("    var acc_{i}_{j} = 0.0;\n");
        }
    }
    for i in 0..mm {
        inner += &format!("            let av{i} = as_tile[(arow + {i}u) * {ks}u + kk];\n");
    }
    if vec4_b {
        inner += &format!(
            "            let bvec = bs_tile[kk * {}u + lid.x];\n",
            tn / 4
        );
        for j in 0..mn {
            inner += &format!("            let bv{j} = bvec[{j}u];\n");
        }
    } else {
        for j in 0..mn {
            inner += &format!("            let bv{j} = bs_tile[kk * {tn}u + bcol + {j}u];\n");
        }
    }
    for i in 0..mm {
        for j in 0..mn {
            inner += &format!("            acc_{i}_{j} = fma(av{i}, bv{j}, acc_{i}_{j});\n");
        }
    }
    for i in 0..mm {
        store += &format!("    let r{i} = row0 + lid.y * {mm}u + {i}u;\n");
        for j in 0..mn {
            store += &format!(
                "    {{ let c = col0 + lid.x * {mn}u + {j}u;\n\
                 \u{20}     if (r{i} < pc.m && c < pc.n) {{ out[o_base + r{i} * pc.n + c] = acc_{i}_{j}; }} }}\n"
            );
        }
    }
    format!(
        r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
struct Push {{ m: u32, k: u32, n: u32, batch: u32 }}
var<immediate> pc: Push;

var<workgroup> as_tile: array<f32, {a_len}>;
var<workgroup> bs_tile: array<{btype}, {b_len}>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.y * {tm}u;
    let col0 = wid.x * {tn}u;
    let a_base = wid.z * pc.m * pc.k;
    let b_base = wid.z * pc.k * pc.n;
    let o_base = wid.z * pc.m * pc.n;
{decl}
    let ntiles = (pc.k + {ks}u - 1u) / {ks}u;
    for (var t = 0u; t < ntiles; t = t + 1u) {{
        let k0 = t * {ks}u;
        for (var s = 0u; s < {mm}u; s = s + 1u) {{
            let l = tid + s * 256u;
            let gr = row0 + l / {ks}u;
            let gk = k0 + l % {ks}u;
            var v = 0.0;
            if (gr < pc.m && gk < pc.k) {{ v = a[a_base + gr * pc.k + gk]; }}
            as_tile[l] = v;
        }}
        for (var s = 0u; s < {mn}u; s = s + 1u) {{
            let l = tid + s * 256u;
            let gk = k0 + l / {tn}u;
            let gc = col0 + l % {tn}u;
            var v = 0.0;
            if (gk < pc.k && gc < pc.n) {{ v = b[b_base + gk * pc.n + gc]; }}
{bstore}
        }}
        workgroupBarrier();
        let arow = lid.y * {mm}u;
        let bcol = lid.x * {mn}u;
        for (var kk = 0u; kk < {ks}u; kk = kk + 1u) {{
{inner}        }}
        workgroupBarrier();
    }}
{store}}}
"#,
        a_len = tm * ks,
        b_len = if vec4_b { ks * tn / 4 } else { ks * tn },
        btype = if vec4_b { "vec4<f32>" } else { "f32" },
        bstore = if vec4_b {
            "            bs_tile[l / 4u][l % 4u] = v;"
        } else {
            "            bs_tile[l] = v;"
        },
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = VkContext::new()?;
    println!(
        "shape                     flops   {}",
        VARIANTS
            .iter()
            .map(|(a, b, v)| format!("{a}x{b}{:<6}", if *v { "v" } else { "" }))
            .collect::<String>()
    );

    let mut totals = vec![0.0f64; VARIANTS.len()];
    for &(m, k, n, batch, count, name) in SHAPES {
        let (mb, kb, nb) = (m as u64, k as u64, n as u64);
        let a_buf = ctx.create_storage_buffer(4 * batch as u64 * mb * kb)?;
        let b_buf = ctx.create_storage_buffer(4 * batch as u64 * kb * nb)?;
        let o_buf = ctx.create_storage_buffer(4 * batch as u64 * mb * nb)?;
        let flops = 2.0 * batch as f64 * mb as f64 * kb as f64 * nb as f64;
        let mut push = Vec::new();
        for v in [m, k, n, batch] {
            push.extend_from_slice(&v.to_le_bytes());
        }

        print!("{name} {m}x{k}x{n}x{batch}  ");
        for (vi, &(mm, mn, v4)) in VARIANTS.iter().enumerate() {
            let pipe = ctx.create_pipeline(&compile_wgsl(&source(mm, mn, v4))?, 3, 16)?;
            let groups = [n.div_ceil(16 * mn), m.div_ceil(16 * mm), batch];
            let bufs = [&a_buf, &b_buf, &o_buf];
            let run = |reps: u32| -> Result<f64, Box<dyn std::error::Error>> {
                let t = Instant::now();
                for _ in 0..reps {
                    ctx.stream_dispatch(&pipe, &bufs, &push, groups)?;
                }
                ctx.flush()?;
                Ok(t.elapsed().as_secs_f64() / reps as f64)
            };
            run(2)?;
            let best = (0..3).try_fold(f64::MAX, |acc, _| run(8).map(|s| acc.min(s)))?;
            totals[vi] += best * count as f64 * 1e3;
            print!("{:8.2}", flops / best / 1e12);
        }
        println!("   TFLOP/s");
    }

    println!("\nmodel MatMul total (ms), all 64 nodes:");
    for (vi, &(mm, mn, v4)) in VARIANTS.iter().enumerate() {
        println!(
            "  {mm}x{mn}{}  {:8.1}",
            if v4 { "v" } else { " " },
            totals[vi]
        );
    }
    Ok(())
}
