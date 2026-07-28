# onnx-vulkan

Run ONNX models on a Vulkan GPU, in pure Rust, with **no ONNX Runtime**.

```rust
let session = onnx_vulkan::Session::load("model.onnx")?;
let run = session.run([("data", my_input)])?;
let output = run.get("output")?;
run.finish();
# Ok::<(), onnx_vulkan::Error>(())
```

Three properties are the point of this crate, and each is a contract rather than
a default:

- **Nothing native ships with it.** The only shared library the process opens is
  the system Vulkan loader, the way a program opens `libc`. `ldd` on a binary
  built against this crate lists no ONNX Runtime.
- **All or nothing.** `Session::load` refuses a model containing any node the
  engine cannot run, and the error names every one of them. There is no per-op
  fallback to the CPU, silent or otherwise: a session that exists runs entirely
  on the GPU.
- **The model stays on the device.** Weights are uploaded and pipelines compiled
  on the first `Session::run` and reused by every later one, for as long as the
  session lives. Loading is the expensive call; running is not.

The Vulkan device is created once per process, on first use.

## Layering

This is the public facade. Underneath it:

- [`vk-compute`](https://crates.io/crates/vk-compute) — pure Vulkan compute.
- [`onnx-vulkan-core`](https://crates.io/crates/onnx-vulkan-core) — graph IR,
  device interpreter, shaders, fusion.
- [`onnx-vulkan-frontend`](https://crates.io/crates/onnx-vulkan-frontend) —
  `.onnx` loader.

For the ONNX Runtime **plugin** (Vulkan Execution Provider) and the reference
STT app, see the repository.

## Runtime requirements

A Vulkan 1.2+ loader must be present at runtime. NVIDIA / AMD GPUs are
supported; lavapipe works as a CPU fallback but is **not** representative for
performance.

## License

Dual-licensed under MIT or Apache-2.0, at your option ([LICENSE-MIT],
[LICENSE-APACHE]).

[LICENSE-MIT]: ../../LICENSE-MIT
[LICENSE-APACHE]: ../../LICENSE-APACHE

## Trademark

Vulkan and the Vulkan logo are registered trademarks of the Khronos Group Inc.
This project is not affiliated with or endorsed by the Khronos Group.
