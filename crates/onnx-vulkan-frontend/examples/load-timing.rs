//! Where load time goes: parse, IR clone, and each load-time rewrite, timed
//! separately.
//!
//! It exists because the obvious suspect was the wrong one. On qwen2.5-VL's q4
//! decoder the load took 767 s and `GraphIr::clone` — the deep copy of ~2 GB of
//! weights that looked like the culprit — was **192 ms** of it. The cost was
//! `read_external` reading the whole external-data file once per initializer,
//! 959 times. Without a per-phase number that is a two-line fix nobody finds.
//!
//!     cargo run --release -p onnx-vulkan-frontend --example load-timing -- model.onnx

use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let t = Instant::now();
    let bytes = std::fs::read(&path).expect("read");
    println!("read {} MB in {:?}", bytes.len() / 1_000_000, t.elapsed());

    let t = Instant::now();
    let model = onnx_vulkan_frontend::load(&path).expect("load");
    println!("frontend::load in {:?}", t.elapsed());

    let ir = &model.graph;
    let weights: usize = ir.initializers.values().map(|i| i.data.len()).sum();
    println!(
        "{} nodes, {} initializers, {} MB of weights",
        ir.nodes.len(),
        ir.initializers.len(),
        weights / 1_000_000
    );

    let t = Instant::now();
    let mut rewritten = ir.clone();
    println!("GraphIr::clone in {:?}", t.elapsed());

    for (label, count) in [
        ("fuse_layernorm", {
            let t = Instant::now();
            let n = onnx_vulkan_core::fuse_layernorm(&mut rewritten);
            println!("fuse_layernorm in {:?}", t.elapsed());
            n
        }),
        ("fold_constants", {
            let t = Instant::now();
            let n = onnx_vulkan_core::fold_constants(&mut rewritten);
            println!("fold_constants in {:?}", t.elapsed());
            n
        }),
        ("prune_dead_nodes", {
            let t = Instant::now();
            let n = onnx_vulkan_core::prune_dead_nodes(&mut rewritten);
            println!("prune_dead_nodes in {:?}", t.elapsed());
            n
        }),
    ] {
        println!("  {label}: {count}");
    }
}
