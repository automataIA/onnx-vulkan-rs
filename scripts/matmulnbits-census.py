#!/usr/bin/env -S uv run --quiet --with onnx --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx"]
# ///
"""Census of the `MatMulNBits` geometries a model actually runs.

The tier-1 bench of a kernel must run the shapes of a real graph, never
synthetic ones (`CLAUDE.md`, "the suite is the gate, not the search"). This
prints the distinct `(N, K, bits, block_size, zero-point, bias)` tuples with
their node counts, so `example matmulnbits` can be written from measured
geometries instead of from what the architecture suggests.

    ./scripts/matmulnbits-census.py models/zoo/gemma3-1b/model_q4.onnx

`M` is deliberately not part of the tuple: it comes from the activation, which
is symbolic in a decoder (prefill vs decode are the same node at two very
different shapes). It is reported separately, as the symbol names.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import onnx

OP = "MatMulNBits"


def attrs(node) -> dict:
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = a.f
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
    return out


def dims(value_info) -> str:
    shape = value_info.type.tensor_type.shape
    return "×".join(d.dim_param or str(d.dim_value) for d in shape.dim) or "scalar"


def census(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    shapes = {v.name: v for v in list(graph.value_info) + list(graph.input)}

    nodes = [n for n in graph.node if n.op_type == OP]
    print(f"\n{path}")
    print(f"  {len(graph.node)} nodes, {len(nodes)} {OP}")
    if not nodes:
        return

    geometries: Counter = Counter()
    activations: defaultdict[str, int] = defaultdict(int)
    dtypes: Counter = Counter()
    for node in nodes:
        a = attrs(node)
        # inputs: A, B, scales, [zero_points], [g_idx], [bias]
        named = list(node.input) + [""] * 6
        has_zp = bool(named[3])
        has_gidx = bool(named[4])
        has_bias = bool(named[5])
        # per-block or per-column zero point: a scalar per column would make the
        # dequantization a different kernel, so the shape is the discriminator
        zp_shape = ""
        if has_zp:
            zp = inits.get(named[3])
            zp_shape = "×".join(str(d) for d in zp.dims) if zp else "?"
        scales = inits.get(named[2])
        if scales is not None:
            dtypes[onnx.TensorProto.DataType.Name(scales.data_type)] += 1
        geometries[
            (
                a.get("N"),
                a.get("K"),
                a.get("bits"),
                a.get("block_size"),
                a.get("accuracy_level", 0),
                zp_shape if has_zp else "none",
                has_gidx,
                has_bias,
            )
        ] += 1
        if named[0] in shapes:
            activations[dims(shapes[named[0]])] += 1

    print(f"  scale dtype: {dict(dtypes)}")
    print(f"  {len(geometries)} distinct geometries")
    print(
        f"  {'count':>5}  {'N':>6} {'K':>6}  bits  block  acc  "
        f"{'zero-point':<14} g_idx bias   MB(B)"
    )
    for geom, count in sorted(geometries.items(), key=lambda kv: -kv[1]):
        n, k, bits, block, acc, zp, gidx, bias = geom
        # packed weight bytes: K·N·bits/8, plus one scale per block
        packed = (n * k * bits / 8) / 1e6 if n and k and bits else 0
        print(
            f"  {count:>5}  {n:>6} {k:>6}  {bits:>4}  {block:>5}  {acc:>3}  "
            f"{zp:<14} {str(gidx):<5} {str(bias):<5} {packed:>7.2f}"
        )
    if activations:
        print("  activation shapes (A):")
        for shape, count in sorted(activations.items(), key=lambda kv: -kv[1]):
            print(f"    {count:>5}  {shape}")
    total = sum(
        (g[0] * g[1] * g[2] / 8) * c for g, c in geometries.items() if all(g[:3])
    )
    print(f"  packed weights across all {OP}: {total / 1e6:.1f} MB")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    for p in paths:
        census(Path(p))
