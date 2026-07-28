//! Loads models and reports what the frontend understood of them.
//!
//! The number that matters is the shape-inference coverage: how many of the
//! values the graph produces come out with a fully known shape. It is the
//! metric that says whether load-time memory planning is possible.

fn main() {
    for path in std::env::args().skip(1) {
        match onnx_vulkan_frontend::load(&path) {
            Err(e) => println!("{path}\n  ERROR: {e}"),
            Ok(model) => {
                let ir = &model.graph;
                let produced: Vec<&str> = ir
                    .nodes
                    .iter()
                    .flat_map(|n| n.outputs.iter())
                    .filter(|o| !o.is_empty())
                    .map(String::as_str)
                    .collect();
                let concrete = model.types.concrete_count(produced.iter().copied());
                let resolved = model.types.resolved_count(produced.iter().copied());
                let typed = produced
                    .iter()
                    .filter(|n| model.types.get(n).is_some_and(|t| t.dtype.is_some()))
                    .count();
                let bytes: usize = ir.initializers.values().map(|i| i.data.len()).sum();
                let unsupported = ir
                    .nodes
                    .iter()
                    .filter(|n| !onnx_vulkan_core::is_implemented_node(n))
                    .count();

                // Same rewrites `Executor::new` applies, measured here because
                // they need no GPU and the count is the point.
                let mut rewritten = ir.clone();
                let fused = onnx_vulkan_core::fuse_layernorm(&mut rewritten);
                let folded = onnx_vulkan_core::fold_constants(&mut rewritten);
                let pruned = onnx_vulkan_core::prune_dead_nodes(&mut rewritten);
                let released = onnx_vulkan_core::prune_dead_initializers(&mut rewritten);

                println!("{path}");
                println!(
                    "  nodes {} · initializers {} ({:.1} MB) · not implemented {unsupported}",
                    ir.nodes.len(),
                    ir.initializers.len(),
                    bytes as f64 / 1e6
                );
                println!(
                    "  rewrite: {fused} LayerNormalization fused · {folded} constants folded · \
                     {pruned} dead nodes pruned · {:.1} MB of initializers released · {} nodes left",
                    released as f64 / 1e6,
                    rewritten.nodes.len()
                );
                let pct = |n: usize| 100.0 * n as f64 / produced.len().max(1) as f64;
                println!(
                    "  values produced {} · known dtype {typed} ({:.0}%) · resolved shape {resolved} ({:.0}%) · of which fixed {concrete} ({:.0}%)",
                    produced.len(),
                    pct(typed),
                    pct(resolved),
                    pct(concrete)
                );
                for name in ir.inputs.iter().chain(&ir.outputs) {
                    let kind = if ir.inputs.contains(name) {
                        "input "
                    } else {
                        "output"
                    };
                    match model.types.get(name) {
                        Some(t) => println!("  {kind} {name}: {t}"),
                        None => println!("  {kind} {name}: unknown"),
                    }
                }
                for conflict in model.conflicts.iter().take(5) {
                    println!("  ! {conflict}");
                }
                if model.conflicts.len() > 5 {
                    println!("  ! …and {} more", model.conflicts.len() - 5);
                }
            }
        }
    }
}
