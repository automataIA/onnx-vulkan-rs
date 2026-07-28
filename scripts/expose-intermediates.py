#!/usr/bin/env -S uv run --quiet --with onnx --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx"]
# ///
"""Exposes a model's intermediate tensors as graph outputs.

Used to localize **where** two backends start to diverge: on dynamic-
quantization graphs, comparing the final outputs says nothing (a minimal
perturbation shifts the scales in cascade), whereas the first intermediate
that diverges points to the responsible node.

    ./scripts/expose-intermediates.py in.onnx out.onnx [--from N] [--every N] [--limit N]
    ./target/release/model-runner out.onnx --dump 3

Note: adding outputs prevents ONNX Runtime from fusing those nodes, so the
instrumented graph is no longer identical to the original. Fine for localizing
a divergence, not for measuring performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        sys.exit(__doc__)
    src, dst = Path(argv[0]), Path(argv[1])
    every = 1
    limit = 0
    start = 0
    rest = argv[2:]
    while rest:
        flag, value, rest = rest[0], rest[1], rest[2:]
        if flag == "--from":
            start = int(value)
        elif flag == "--every":
            every = int(value)
        elif flag == "--limit":
            limit = int(value)
        else:
            sys.exit(f"unknown flag: {flag}")

    model = onnx.load(str(src))
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    known = {vi.name: vi for vi in model.graph.value_info}
    existing = {o.name for o in model.graph.output}

    exposed: list[tuple[str, str]] = []
    added = 0
    for index, node in enumerate(model.graph.node):
        if index < start or index % every:
            continue
        for name in node.output:
            if not name or name in existing:
                continue
            vi = known.get(name)
            # without an inferred type the output cannot be declared
            if vi is None or vi.type.tensor_type.elem_type == TensorProto.UNDEFINED:
                continue
            model.graph.output.append(helper.ValueInfoProto())
            model.graph.output[-1].CopyFrom(vi)
            existing.add(name)
            exposed.append((name, TensorProto.DataType.Name(vi.type.tensor_type.elem_type)))
            added += 1
            break
        if limit and added >= limit:
            break

    onnx.save(model, str(dst), save_as_external_data=False)
    # name and dtype, in graph order: whoever compares per-node needs to know
    # which is the first **quantized** intermediate, which in the model outputs
    # comes after the original ones and is not necessarily the first exposed
    names = dst.with_suffix(".outputs.txt")
    names.write_text("".join(f"{n}\t{t}\n" for n, t in exposed), encoding="utf-8")
    print(f"{added} intermediates exposed out of {len(model.graph.node)} nodes → {dst}")
    print(f"names in {names}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
