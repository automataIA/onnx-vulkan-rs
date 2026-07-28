# vk-compute

Pure-Rust Vulkan compute runtime. No ONNX, no framework — just the GPU plumbing:
a Vulkan device via [`ash`], buffers through `gpu-allocator`, WGSL→SPIR-V
compilation via `naga`, a deferred command stream, a persistent descriptor
arena, and a GPU timestamp profiler.

This is the low-level layer that [`onnx-vulkan`](https://crates.io/crates/onnx-vulkan)
builds on, but it carries no ONNX dependency and is usable on its own for any
Vulkan compute workload.

## Usage

```rust
use vk_compute::{VkContext, compile_wgsl};

let ctx = VkContext::new()?;          // picks the first suitable device
let spirv = compile_wgsl(MY_WGSL)?;   // WGSL source → SPIR-V
// ... build a ComputePipeline, dispatch, flush, readback
# Ok::<(), Box<dyn std::error::Error>>(())
```

`VkContext` is created once per process (the entry point is cheap after the
first call). Dispatches are recorded into a deferred stream and submitted as a
single command buffer on `flush()`.

## Runtime requirements

A Vulkan 1.2+ loader must be present at runtime (system `libvulkan.so.1` on
Linux, `vulkan-1.dll` on Windows). NVIDIA / AMD GPUs are supported; lavapipe
works as a CPU fallback but is **not** representative for performance.

## License

Dual-licensed under MIT or Apache-2.0, at your option ([LICENSE-MIT],
[LICENSE-APACHE]).

[LICENSE-MIT]: ../../LICENSE-MIT
[LICENSE-APACHE]: ../../LICENSE-APACHE

## Trademark

Vulkan and the Vulkan logo are registered trademarks of the Khronos Group Inc.
This project is not affiliated with or endorsed by the Khronos Group.
