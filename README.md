<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-onnx-vulkan-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-onnx-vulkan-light.svg">
    <img alt="onnx-vulkan-rs logo" src="assets/logo-onnx-vulkan-light.svg" width="200">
  </picture>
</p>

# onnx-vulkan-rs

A Vulkan Execution Provider **plugin** for ONNX Runtime, entirely in Rust, plus
a **standalone pure-Rust engine** (own ONNX parser, no ORT in the process) that
shares the same kernels. Reference app: STT with **Parakeet TDT 0.6B v3 (int8
ONNX)**. Linux and Windows, NVIDIA/AMD GPUs (lavapipe as a CPU fallback —
correctness only, never performance).

ONNX Runtime has no official Vulkan EP: this project implements one out-of-tree
using the [Plugin EP API](https://onnxruntime.ai/docs/execution-providers/plugin-ep-libraries/usage.html)
(ORT ≥ 1.23, pinned here to **1.27.1**). The long-term target is the standalone
library (`plan.md`); the plugin is how the kernels get validated against a real
runtime on real models.

## Architecture

```
                         onnx-vulkan-core
                    GraphIr, shaders, interpreter,
                    fusion, KernelCache, host ops
                           │        │
   ┌───────────────────────┘        └──────────────────────┐
   │                                                       │
onnx-vulkan-frontend                              vulkan-ep (cdylib)
 .onnx → prost/protox → GraphIr                    ORT Plugin EP API
 static shape inference                            OrtGraph → GraphIr
 no ORT in the process                             GetCapability / Compile
   │                                                       │
   └──────────────────► vk-compute ◄──────────────────────┘
                 ash + gpu-allocator + naga (WGSL→SPIR-V)
                 deferred stream, descriptor arena, profiler
```

- **`crates/onnx-vulkan-core`** — owned `GraphIr`, all WGSL shaders, the device
  interpreter, convex fusion, host shape ops, `Executor`, session-owned
  `KernelCache`. Depends only on `vk-compute`, `anyhow`, `log` — **no ORT**.
- **`crates/vk-compute`** — pure Vulkan compute: `VkContext`, buffers,
  WGSL→SPIR-V, deferred command stream, persistent descriptor arena, GPU
  timestamp profiler. Own GPU tests.
- **`crates/onnx-vulkan-frontend`** — `.onnx` parser (vendored ONNX schema,
  `protox` in pure Rust, no `protoc`) + static shape inference → `GraphIr`.
- **`crates/vulkan-ep`** — the ORT plugin, `onnxruntime_ep_vulkan.{so,dll}`.
- **`crates/ort-ep-sys`** — bindgen on the ORT C headers.
- **`crates/stt-app`** — wav → log-mel → encoder → TDT greedy decode → text.
- **`crates/model-runner`** — runs any ONNX model on the CPU EP and the Vulkan
  EP and diffs the outputs. This is what the test suite drives.

### Two execution paths

**Compiling EP** (`VULKAN_EP_COMPILE=1`, the path that is measured):
`GetCapability` returns **convex blocks**, `Compile` builds one
`OrtNodeComputeInfo` per block, each block runs as **a single command buffer**
with pipelines and packed weights living as long as the session. The whole test
matrix runs at **1 convex block per model** — the graph does not return to the
CPU mid-run.

**Standalone** (`cargo run -p onnx-vulkan-frontend --example run-standalone`):
same `GraphIr`, same kernels, no ONNX Runtime loaded at all — `ldd` on the
binary lists only libc/libm/libgcc, Vulkan arrives via `dlopen`.

## Setup

Prerequisites: Rust ≥1.85, clang (for bindgen), Vulkan driver/loader
(`libvulkan1`; on Linux without a GPU: `mesa-vulkan-drivers` for lavapipe).

```bash
./scripts/fetch-deps.sh   # ORT 1.27.1 (linux+win) + Parakeet model (~700MB)
cargo build --release
cargo test                        # Vulkan kernel + core tests (run on lavapipe too)
cargo clippy --workspace -- -D warnings
```

## Usage

```bash
RUST_LOG=info ./target/release/stt-app models/en-sample.wav [model_dir]
cargo run -p model-runner --release -- model.onnx --dim height=560 --dim width=560
scripts/testsuite.sh --baseline runs/baseline.json    # the regression gate
```

The plugin is loaded if present next to the executable
(override: `VULKAN_EP_PATH`; disable: `STT_NO_VULKAN=1`;
alternative ORT runtime: `ORT_DYLIB_PATH`; profiler: `VULKAN_EP_STATS=1`).
The wav must be 16 kHz.

### Windows (cross-build from Linux/WSL2)

```bash
rustup target add x86_64-pc-windows-msvc
cargo xwin build --release --target x86_64-pc-windows-msvc -p vulkan-ep -p stt-app
# from WSL2 you can run directly on the Windows host:
export RUST_LOG=info ORT_DYLIB_PATH='third_party\onnxruntime\win-x64\lib\onnxruntime.dll'
WSLENV=RUST_LOG:ORT_DYLIB_PATH ./target/x86_64-pc-windows-msvc/release/stt-app.exe models/en-sample.wav
```

## Status

- [x] EP plugin loaded by ORT, Vulkan device enumeration, CPU fallback
- [x] GPU-resident tensors (device `OrtAllocator` + DataTransfer + Memcpy)
- [x] Deferred command stream, persistent descriptor arena, GPU profiler
- [x] **Compiling EP**: convex blocks, one command buffer per block —
      **1 block on every model in the suite**
- [x] Core extracted from ORT (`onnx-vulkan-core`), synthetic-graph tests
- [x] **Standalone frontend**: own `.onnx` parser + static shape inference,
      runs the Parakeet encoder with no ONNX Runtime in the process
- [x] Op coverage 100% on the whole vision/speech matrix (parakeet int8, rfdetr
      fp32/int8, sam3 vision int8, SAM3 ViT-H fp32, yolov4/v8n, mobilenetv2,
      resnet50-qdq, roberta)
- [x] `VK_KHR_cooperative_matrix` on `MatMulInteger` (GLSL, compiled offline)
- [x] Register-blocked `MatMul` / `Gemm` / `Conv` behind occupancy predicates
- [x] Liveness-based intermediate release + buffer pool (sam3: OOM → 5.6 GB peak)
- [ ] Public `onnx-vulkan` facade crate (`load` → `run`)
- [ ] Pre-recorded command buffer, full load-time memory planning
- [ ] fp16, static-quant Q/DQ ops, AMD validation
- [ ] `MatMulNBits` + attention ops + generation runtime (int4 LLM)

See `plan.md` for the ordered roadmap and `cronologia.md` for the work log.

## Performance

RTX 4070, driver 610.74, batch 1, native Windows ORT, `runs/conv-blocked-1`.
Ratio is against the ORT **CPU EP (MLAS)** on the same graph.

| model | wall | CPU EP | ratio | blocks | flush | GPU Pareto head |
|---|---|---|---|---|---|---|
| rfdetr | 41.1 ms | 367.1 ms | **8.93×** | 1 | 9 | `MatMul` 55% |
| parakeet (encoder) | 46.6 ms | 294.8 ms | **6.32×** | 1 | 7 | `MMI_matmul_coop_k32` 58% |
| yolov4 | 42.0 ms | 168.6 ms | **4.01×** | 1 | 4 | `Conv` 85% |
| yolov8n | 9.2 ms | 25.6 ms | **2.78×** | 1 | 2 | `Conv` 62% |
| mobilenetv2 | 1.8 ms | 2.0 ms | 1.11× | 1 | 2 | `Conv16` 47% |
| roberta (seq 1) | 10.2 ms | 7.8 ms | **0.76×** | 1 | 4 | `MatMul16` 64% |
| resnet50-qdq | 9.3 ms | 6.9 ms | **0.74×** | 1 | 2 | `Conv16` 49% |

Read honestly:

- **Structure is solved.** Every model is at 1 convex block with sync under
  1 ms (yolov4: 4.5 ms). Boundary count is no longer a lever — the suite is
  kernel-bound end to end.
- **The two models below 1× lose for the same reason**, and it is not a missing
  kernel: **their output tensors are smaller than the GPU**. A 4070 holds 70,656
  resident threads; no `Conv` geometry in resnet50-qdq fills it, and roberta at
  `seq_len = 1` is a GEMV — 768 useful threads out of 70,656. Every tile
  enlargement buys arithmetic intensity by giving back grid width, measured at
  par on both. Full attribution in `docs/resnet50-gap.md`. The same roberta graph
  at `seq_len = 128` runs at **1.96×** (43.2 ms CPU EP against 22.0).
- **mobilenetv2 at 1.11× is overhead-dominated** (1.8 ms total) — not a target.
- **lavapipe numbers mean nothing for performance**; the suite marks those runs
  `perf_valid: false`.

The regression gate (`scripts/testsuite.sh --baseline runs/baseline.json`) fails
on accuracy outside tolerance, median wall past `perf_tol`, more flushes or MB
transferred, or **more convex blocks / fewer claimed nodes** — the last two are
deterministic and lead the wall clock.

## Technical notes

- WGSL→SPIR-V at runtime with naga 30: push-constant parameters use the
  `immediate` address space (renamed from `push_constant`).
- `dot4U8Packed` compiles to a native `OpUDot` whenever naga is allowed the
  `DotProduct` capabilities — which it is here. The SPIR-V version is not the
  discriminator; the polyfill appears only if those capabilities are denied.
  The device must enable `VK_KHR_shader_integer_dot_product`.
- Zero-point correction per block of 4:
  `Σ(aᵢ−az)(bᵢ−bz) = Σaᵢbᵢ − az·Σbᵢ − bz·Σaᵢ + 4·az·bz`.
- Shaders are WGSL, with one exception: the cooperative-matrix (tensor core)
  kernels, which naga cannot express. They are GLSL, compiled offline by
  `scripts/build-glsl.sh`, and their SPIR-V is committed under `shaders/spv/`.
- **Bit-exactness with MLAS is not a goal and is not reachable.** fp32
  summation order differs, and on dynamically quantized graphs a 1-ulp move of a
  tensor's extreme shifts its whole scale. The correctness contract is per-node,
  ±1 LSB on the first quantized tensor, plus the expected transcript / argmax.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

**Models are not included.** The test suite downloads pre-trained models on
demand; each model has its own license. See [NOTICE](NOTICE) for the full
attribution table.

