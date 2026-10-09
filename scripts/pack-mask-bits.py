#!/usr/bin/env -S uv run --quiet --with onnx --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx"]
# ///
"""Returns a mask output as `numpy.packbits(mask > 0, axis=-1)` instead of f32.

A mask decoder's logits are usually only thresholded at zero by the caller
(SAM 3: `pred_masks [1, 200, 288, 288]`, 66 MB of f32 per prompt, of which the
application keeps one bit each). This appends, in standard operators,

    Reshape(x, [..., W/8, 8]) -> Greater(., 0) -> Cast(FLOAT)
      -> Mul(., [128, 64, ..., 1]) -> ReduceSum(-1, keepdims=0) -> Cast(UINT8)

and replaces the output OUTPUT with `OUTPUT_bits`, uint8 `[..., W/8]`: byte j of
a row packs elements 8j..8j+7, the first in the most significant bit, exactly
as `numpy.packbits(x > 0, axis=-1)`. Every step is exact, so any ONNX runtime
gives the same bytes; the Vulkan runtime fuses the tail into one kernel
(`rewrite::fuse_pack_positive_bits`). W must be a multiple of 8.

    ./scripts/pack-mask-bits.py in.onnx out.onnx OUTPUT

External data files of the input are copied next to the output when it lives
elsewhere: ONNX Runtime refuses external data behind a symlink that leaves the
model directory, and the checker refuses symlinks and hard links.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def main(source: Path, target: Path, output: str) -> None:
    model = onnx.load(source, load_external_data=False)
    graph = model.graph
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)
    if opset < 18:
        sys.exit(f"opset {opset}: the ReduceSum written here takes its axes as an input (>= 18)")
    out = next((o for o in graph.output if o.name == output), None)
    if out is None:
        sys.exit(f"{source}: no graph output named {output!r}")
    if out.type.tensor_type.elem_type != TensorProto.FLOAT:
        sys.exit(f"{output}: not f32")
    dims = [d.dim_value if d.HasField("dim_value") else None for d in out.type.tensor_type.shape.dim]
    if not dims or None in dims or dims[-1] % 8:
        sys.exit(f"{output}: shape {dims} must be static with a last axis multiple of 8")

    p = f"{output}_pack"
    grouped = dims[:-1] + [dims[-1] // 8, 8]
    graph.initializer.extend([
        numpy_helper.from_array(np.array(grouped, np.int64), f"{p}_shape"),
        numpy_helper.from_array(np.array(0, np.float32), f"{p}_zero"),
        numpy_helper.from_array(np.array([128, 64, 32, 16, 8, 4, 2, 1], np.float32), f"{p}_weights"),
        numpy_helper.from_array(np.array([-1], np.int64), f"{p}_axes"),
    ])
    graph.node.extend([
        helper.make_node("Reshape", [output, f"{p}_shape"], [f"{p}_groups"], name=f"{p}_reshape"),
        helper.make_node("Greater", [f"{p}_groups", f"{p}_zero"], [f"{p}_positive"], name=f"{p}_greater"),
        helper.make_node("Cast", [f"{p}_positive"], [f"{p}_ones"], name=f"{p}_cast", to=TensorProto.FLOAT),
        helper.make_node("Mul", [f"{p}_ones", f"{p}_weights"], [f"{p}_weighted"], name=f"{p}_mul"),
        helper.make_node("ReduceSum", [f"{p}_weighted", f"{p}_axes"], [f"{p}_sum"], name=f"{p}_sum",
                         keepdims=0),
        helper.make_node("Cast", [f"{p}_sum"], [f"{output}_bits"], name=f"{p}_bits", to=TensorProto.UINT8),
    ])
    index = list(graph.output).index(out)
    graph.output.remove(out)
    graph.output.insert(index, helper.make_tensor_value_info(
        f"{output}_bits", TensorProto.UINT8, dims[:-1] + [dims[-1] // 8]))
    # an in-memory check resolves external data against the working directory
    cwd = Path.cwd()
    os.chdir(source.parent)
    try:
        onnx.checker.check_model(model)
    finally:
        os.chdir(cwd)

    target.parent.mkdir(parents=True, exist_ok=True)
    locations = {entry.value for tensor in graph.initializer
                 for entry in tensor.external_data if entry.key == "location"}
    for location in sorted(locations):
        if (source.parent / location).resolve() != (target.parent / location).resolve():
            shutil.copyfile(source.parent / location, target.parent / location)
    onnx.save(model, target)
    print(f"{target}: {output} {dims} f32 -> {output}_bits {dims[:-1] + [dims[-1] // 8]} u8")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
