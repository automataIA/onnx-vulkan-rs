# onnx-vulkan-core

The reusable core of the `onnx-vulkan-rs` engine: an ONNX-shaped graph IR,
a Vulkan device interpreter, the shared WGSL shader sources, convex-block graph
fusion, and load-time rewrites (decomposed LayerNorm fusion, constant folding,
dead-node pruning).

This crate has **no dependency on ONNX Runtime** — it depends only on
[`vk-compute`](https://crates.io/crates/vk-compute). The same IR is produced by
[`onnx-vulkan-frontend`](https://crates.io/crates/onnx-vulkan-frontend) (from a
`.onnx` file) and by the ORT plugin, so the standalone path and the plugin
cannot end up executing different graphs.

## What's in here

- `GraphIr`, `NodeIr`, `ElementType` — the IR the engine runs.
- `execute` / `is_implemented` — the device interpreter.
- `KernelCache` — session-owned pipelines, packed weights, zero buffer.
- `Executor` — the one entry point every host goes through; runs the load-time
  rewrites so the executed graph is canonical.
- `shaders` — one WGSL module per algorithm, each exporting its bindings and
  push-constant sizes.

## Usage

This crate is the foundation; most users want the facade
[`onnx-vulkan`](https://crates.io/crates/onnx-vulkan) instead. Reach for it
directly only if you are building a different host or running the IR from your
own frontend.

```rust
use onnx_vulkan_core::{Executor, ExecutionEnv, Tensor};
// graph: GraphIr built by your own frontend or by onnx-vulkan-frontend
// let mut env = ExecutionEnv::new(&graph);
// let exe = Executor::new(&vk_ctx, &graph, &mut env)?;
// let out: Tensor = exe.run(&mut env, inputs)?;
```

## License

Dual-licensed under MIT or Apache-2.0, at your option ([LICENSE-MIT],
[LICENSE-APACHE]).

[LICENSE-MIT]: ../../LICENSE-MIT
[LICENSE-APACHE]: ../../LICENSE-APACHE

## Trademark

Vulkan and the Vulkan logo are registered trademarks of the Khronos Group Inc.
This project is not affiliated with or endorsed by the Khronos Group.
