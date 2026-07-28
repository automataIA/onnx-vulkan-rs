# onnx-vulkan-frontend

A standalone `.onnx` file → `GraphIr` loader. Reads the protobuf with `prost`
(the `.proto` is vendored and compiled by `protox`, so no `protoc` to install)
and runs static shape inference that keeps symbolic dimensions as symbols.

This is what makes the engine standalone: the IR that the runtime runs no
longer has to come from an `OrtGraph`, so running a model does not require
ONNX Runtime in the process. The same canonicalization the EP applies
(`fold_constant_params`) is applied here, so coverage does not depend on which
frontend built the IR.

## Usage

```rust
use onnx_vulkan_frontend::load;

let model = load("model.onnx")?;
let graph = &model.graph;             // onnx_vulkan_core::GraphIr
// model.types   — inferred type of every value
// model.conflicts — places where file-declared and inferred shapes disagree
# Ok::<(), onnx_vulkan_frontend::Error>(())
```

External weights are resolved relative to the model's own directory, as the
ONNX spec prescribes — this keeps a downloaded model self-contained.

## License

Dual-licensed under MIT or Apache-2.0, at your option ([LICENSE-MIT],
[LICENSE-APACHE]).

[LICENSE-MIT]: ../../LICENSE-MIT
[LICENSE-APACHE]: ../../LICENSE-APACHE
