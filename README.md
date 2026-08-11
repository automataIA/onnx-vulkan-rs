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
shares the same kernels. Linux, NVIDIA/AMD GPUs (lavapipe as a CPU
fallback — correctness only, never performance).

Runs int4 LLMs at **~160 tok/s** (gemma3-1b) and **~71 tok/s** (qwen2.5-VL),
puts static int8 convolutions on the **tensor cores**, and reaches 7.98× the
ORT CPU EP on the Parakeet encoder — while refusing at load time, loudly and completely, any
graph it cannot run end to end on the GPU. Reference app: STT with **Parakeet
TDT 0.6B v3 (int8 ONNX)**.

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
./scripts/fetch-deps.sh   # ORT 1.27.1 (linux-x64) + Parakeet model (~700MB)
cargo build --release
cargo test -- --test-threads=1    # Vulkan kernel + core tests (run on lavapipe too)
cargo clippy --workspace -- -D warnings
```

`--test-threads=1` is not optional on a real GPU: every test builds its own
`VkContext`, and creating a dozen Vulkan devices concurrently wedges the NVIDIA
Linux driver — the same suite that takes 2 s serialized did not finish in 25
minutes in parallel. On lavapipe the default threading works.

## Usage

```bash
RUST_LOG=info ./target/release/stt-app models/en-sample.wav [model_dir]
cargo run -p model-runner --release -- model.onnx --dim height=560 --dim width=560
scripts/testsuite.sh --baseline runs/baseline.json    # the regression gate
```

To tune the supported workloads of a concrete ONNX model into an exact,
device-specific tactic artifact, see
[`docs/autotune-model.md`](docs/autotune-model.md). The one-command entry point
is `uv run scripts/autotune/model.py model.onnx --output tactics.json`.

The plugin is loaded if present next to the executable
(override: `VULKAN_EP_PATH`; disable: `STT_NO_VULKAN=1`;
alternative ORT runtime: `ORT_DYLIB_PATH`; profiler: `VULKAN_EP_STATS=1`).
The wav must be 16 kHz.

## Status

- [x] EP plugin loaded by ORT, Vulkan device enumeration, CPU fallback
- [x] GPU-resident tensors (device `OrtAllocator` + DataTransfer + Memcpy)
- [x] Deferred command stream, persistent descriptor arena, GPU profiler
- [x] **Compiling EP**: convex blocks, one command buffer per block —
      **1 block on every model in the suite**
- [x] Core extracted from ORT (`onnx-vulkan-core`), synthetic-graph tests
- [x] **Standalone frontend**: own `.onnx` parser + static shape inference,
      runs the Parakeet encoder with no ONNX Runtime in the process
- [x] Op coverage 100% on the whole matrix: vision, speech and LLM (parakeet
      int8, rfdetr fp32/int8, sam3 vision int8, SAM3 ViT-H fp32, yolov4/v8n,
      mobilenetv2 fp32/int8, resnet50 qdq/int8, roberta, gemma3-1b int4,
      qwen2.5-VL int4)
- [x] `VK_KHR_cooperative_matrix` on `MatMulInteger` (GLSL, compiled offline)
- [x] Register-blocked `MatMul` / `Gemm` / `Conv` behind occupancy predicates
- [x] Liveness-based intermediate release + buffer pool (sam3: OOM → 5.6 GB peak)
- [x] Public `onnx-vulkan` facade crate (`Session::load` → `run` → `get`)
- [x] **int4 LLM**: `MatMulNBits`, `GroupQueryAttention`, `RotaryEmbedding`,
      `GatherBlockQuantized`, resident KV cache, replayed decode loop
- [x] **Static int8 (QOperator)**: `QLinearConv`/`MatMul`/`Add`/`GlobalAveragePool`,
      `ConvInteger` on the tensor cores via im2col + `cooperative_matrix`
- [x] **Load-time validation**: opset bounds and operand dtypes, so an
      unsupported node is refused before it can answer wrongly
- [ ] fp16, AMD validation (parked for want of hardware)
- [ ] Full arena allocation (archived: no model is near OOM — `plan.md` §5)

See `plan.md` for the ordered roadmap and `cronologia.md` for the work log.

## Fail loud, by construction

The engine runs a graph **entirely** on the GPU or refuses it at load time,
naming every offender. There is no silent per-node fallback, and that is an
architectural choice, not a missing feature: a backend that quietly drops nodes
to the CPU turns a coverage gap into a performance mystery, and a backend that
claims a node it cannot honour turns it into a wrong number.

Three independent checks must agree before a node is claimed, all of them
before any memory is allocated:

| check | reads | example refusal |
|---|---|---|
| `is_implemented_node` | op, **opset**, attributes, arity | `Pad` below opset 11, where `pads` is an attribute and the input the kernel reads does not exist |
| `unsupported_quantization` | the operand **values** | a per-channel weight zero point that is not identically zero |
| `unsupported_dtype` | the operand **types** | a non-float tensor reaching a kernel that reads `f32` and has no integer path |

Two of the three exist because the first was not enough: an int64-mask
`ReduceSum` claimed by a float kernel crashed a run, and a `uint8` `MaxPool`
read packed bytes as floats — `max|Δ| = 8.086`, argmax 489 → 611. A **silently
wrong answer**, caught only because that model ships golden reference data. The
failure mode this design refuses is not "unsupported"; it is "plausible".

Above `MAX_OPSET` a model is refused whole, and the message says why: the
kernels have not been read against that revision of the spec. Re-exporting at a
supported opset is one line in every export tool, and it beats a number nobody
can trust.

## Performance

RTX 4070, driver 595.84, Pop!_OS 24.04, batch 1, `runs/popos-3`.
Ratio is against the ORT **CPU EP (MLAS)** on the same graph. `GPU` is the
profiler's compute time, i.e. the part of the wall that is actually shaders.

| model | wall | CPU EP | ratio | GPU | blocks | flush | GPU Pareto head |
|---|---|---|---|---|---|---|---|
| parakeet (encoder) | 39.4 ms | 314.3 ms | **7.98×** | 31.0 | 1 | 7 | `MMI_matmul_coop_k32` 58% |
| rfdetr | 38.7 ms | 260.1 ms | **6.72×** | 34.0 | 1 | 9 | `MatMul` 60% |
| yolov4 | 33.7 ms | 123.4 ms | **3.66×** | 19.6 | 1 | 4 | `Conv_split` 55% |
| yolov8n | 6.5 ms | 22.1 ms | **3.40×** | 4.9 | 1 | 2 | `Conv_split` 41% |
| roberta seq 1 | 3.3 ms | 9.4 ms | **2.85×** | 2.2 | 1 | 4 | `GEMV` 46% |
| roberta seq 128 | 17.7 ms | 40.9 ms | **2.31×** | 16.2 | 1 | 4 | `MatMul` 88% |
| qwen2.5-VL decoder (int4) | 28.5 ms | 57.6 ms | **2.02×** | 12.1 | 1 | 77 | `MatMulNBits_wide` 37% |
| gemma3-1b (int4) | 17.3 ms | 18.9 ms | **1.09×** | 6.0 | 1 | 57 | `MatMulNBits` 29% |
| resnet50-int8 | 3.4 ms | 3.6 ms | **1.06×** | 2.4 | 1 | 2 | `ConvInteger_coop` 40% |
| resnet50-qdq | 4.7 ms | 4.9 ms | **1.04×** | 3.9 | 1 | 2 | `Conv_split` 61% |
| mobilenetv2 | 1.5 ms | 1.2 ms | 0.80× | 1.1 | 1 | 2 | `Conv` 29% |
| mobilenetv2-int8 | 1.8 ms | 0.7 ms | 0.39× | 0.9 | 1 | 2 | `Requantize` 29% |

Correctness in the same run: `sync-check.sh` reports **12/12 models clean**
under the Khronos synchronization-validation layer, both int8 classifiers are
bit-exact against their golden data (`max|Δ| = 0.000e0`), and every golden
argmax is reproduced.

Generation, measured on the replayed decode loop rather than on a single
forward: **gemma3-1b 6.24 ms/token (~160 tok/s)** and **qwen2.5-VL 14.09
ms/token (~71 tok/s)**, zero device allocations per step, one flush. These two
were taken on the previous Windows host and have **not** been re-measured since
the move; at one flush per step they are the least exposed numbers here to the
per-flush cost described below — which also means the fence fix should barely
move them — but they are not confirmed either way.

Read honestly:

- **The biggest single win here was one Vulkan call, not a kernel.** Every
  submit+fence used to create and destroy its fence. `cargo run --release -p
  vk-compute --example flush_cost` times the round trip with nothing to compute:
  0.496 ms through `flush()`, 0.136 ms with a pre-recorded command buffer and a
  reused fence, and 0.494 ms again the moment the fence is created and destroyed
  — so **0.358 ms of every flush was one fence allocation**, which the NVIDIA
  driver charges dearly and lavapipe does not charge at all. Creating the fence
  once and resetting it per submit took **qwen2.5-VL 69.4 → 28.5 ms** and
  **gemma3-1b 41.2 → 17.3**, and moved all twelve models by 0.42–0.73 ms per
  flush — a constant that now shows up identically from 2 flushes to 77.
- **Structure is solved and the submit boundary is now cheap.** Every model is
  at 1 convex block, so boundary *count* was already not a lever; each remaining
  flush costs ~0.14 ms, which is the driver's own floor.
- **The kernels are unchanged across all of this.** GPU compute matches the
  previous Windows-host baseline within ±10% on every model (gemma3-1b 6.4 →
  6.0 ms, rfdetr 34.9 → 34.0, yolov8n 4.83 → 4.86). Everything above and below
  moved the wall around the shaders, never the shaders.
- **Two models are below the CPU EP, and no kernel will move them.**
  `mobilenetv2-int8` spends 0.9 of its 1.8 ms on the GPU; the rest is ORT and
  plugin overhead across 73 nodes. Its `ConvInteger` work is 17 depthwise
  convolutions with `K = 9` — not a GEMM, nothing to tile, nothing to split.
  They are overhead items, and they are labelled as such instead of tuned.
- **The two models that used to be below 1× no longer are**, and neither was
  fixed by a bigger tile. roberta at `seq_len = 1` (0.76×) is a GEMV: 768 useful
  threads on a card that holds 70,656, so the fix was splitting `K` to
  manufacture workgroups — now 2.85×, at 500–515 GB/s against the card's ~504,
  which means that lever is spent rather than merely pulled. resnet50-qdq
  (0.74×) was the same diagnosis with split-K on `Conv` — now 1.04× against a
  reference that reads 4.9 ms today and 7.0 ms a run ago. Full attribution in
  `docs/resnet50-gap.md`.
- **Ratios move because the reference moves, and this table is mostly ratios.**
  The MLAS baseline drifted 4.5 → 6.2 ms on the same binary across consecutive
  runs; between the two runs behind this table it moved up to 20% (gemma 21.8 →
  18.9, roberta 7.8 → 9.4) while our own walls moved by tenths. Compare
  milliseconds, and treat `blocks` / `flushes` / `MB transferred` — which are
  deterministic — as the primary metric.
- **lavapipe numbers mean nothing for performance**; the suite marks those runs
  `perf_valid: false`. They are still a valid *correctness* gate, because
  integer arithmetic is exact on any device.

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
