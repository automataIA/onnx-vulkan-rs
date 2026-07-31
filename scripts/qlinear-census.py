#!/usr/bin/env -S uv run --quiet --with onnx --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx"]
# ///
"""Census of the QOperator (static int8) nodes a model actually runs.

Same purpose as `matmulnbits-census.py`, for the other quantization format:
`is_implemented_node` must be written on the constraints a real export uses,
not on the ones the schema allows. The schema of `QLinearConv` permits things
no model in the zoo emits (per-channel `x_zero_point`, `i8` and `u8` mixed
across operands, `channels_last`), and refusing them costs a line while
supporting them costs a kernel.

    ./scripts/qlinear-census.py models/zoo/golden/resnet50-v1-12-int8/*.onnx

For every op it prints, per distinct signature: which optional inputs are
present, whether each scale/zero-point is a **per-tensor scalar or a
per-channel vector** (the discriminator that decides the kernel epilogue),
the element type of every quantized operand, and — for `QLinearConv` — the
convolution geometry the tier-1 bench will need.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import onnx

# input order per op, from the ONNX / contrib schemas. `None` marks a slot that
# is optional in the schema; the census reports whether the export uses it.
SPECS: dict[str, list[str]] = {
    "QLinearConv": [
        "x", "x_scale", "x_zero_point",
        "w", "w_scale", "w_zero_point",
        "y_scale", "y_zero_point", "B",
    ],
    "QLinearMatMul": [
        "a", "a_scale", "a_zero_point",
        "b", "b_scale", "b_zero_point",
        "y_scale", "y_zero_point",
    ],
    "QLinearAdd": [
        "A", "A_scale", "A_zero_point",
        "B", "B_scale", "B_zero_point",
        "C_scale", "C_zero_point",
    ],
    "QLinearGlobalAveragePool": [
        "X", "x_scale", "x_zero_point", "y_scale", "y_zero_point",
    ],
}

# the data operands, as opposed to the quantization parameters. Per op, because
# `B` is the bias of a `QLinearConv` and the second addend of a `QLinearAdd`.
BULK: dict[str, set[str]] = {
    "QLinearConv": {"x", "w"},
    "QLinearMatMul": {"a", "b"},
    "QLinearAdd": {"A", "B"},
    "QLinearGlobalAveragePool": {"X"},
}


def attrs(node) -> dict:
    out: dict = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.INTS:
            out[a.name] = tuple(a.ints)
        elif a.type == onnx.AttributeProto.STRING:
            out[a.name] = a.s.decode()
    return out


def dtype_name(proto: int) -> str:
    return onnx.TensorProto.DataType.Name(proto).lower()


def describe(name: str, inits: dict, shapes: dict, *, bulk: bool = False) -> str:
    """One operand, as the kernel will see it.

    An initializer is folded at load time and its *shape* is what matters —
    `scalar` and `per-channel` are two different epilogues. A runtime value is
    an activation and only its type is known here.

    `bulk` marks the data operands (the weights, the activations): their length
    is geometry, reported separately, and folding it into the signature would
    print one signature per layer instead of one per *kind* of layer.
    """
    if not name:
        return "-"
    init = inits.get(name)
    if init is not None:
        if bulk:
            return f"{dtype_name(init.data_type)}/init"
        count = 1
        for d in init.dims:
            count *= d
        kind = "scalar" if count == 1 else "per-channel"
        return f"{dtype_name(init.data_type)}/{kind}"
    info = shapes.get(name)
    if info is not None and info.type.tensor_type.elem_type:
        return f"{dtype_name(info.type.tensor_type.elem_type)}/dyn"
    return "dyn"


def conv_geometry(node, inits: dict) -> tuple:
    a = attrs(node)
    w = inits.get(node.input[3])
    if w is None:
        return ("?",)
    c_out, c_in_g, *kernel = list(w.dims)
    group = a.get("group", 1)
    return (
        c_in_g * group,
        c_out,
        "×".join(str(k) for k in kernel),
        group,
        a.get("strides", ()),
        a.get("pads", ()),
        a.get("dilations", ()),
        a.get("auto_pad", "NOTSET"),
    )


def census(path: Path) -> None:
    model = onnx.load(str(path), load_external_data=False)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    shapes = {
        v.name: v
        for v in list(graph.value_info) + list(graph.input) + list(graph.output)
    }

    print(f"\n=== {path}")
    print(f"    {len(graph.node)} nodes")

    for op, spec in SPECS.items():
        nodes = [n for n in graph.node if n.op_type == op]
        if not nodes:
            continue
        domains = Counter(n.domain or "ai.onnx" for n in nodes)
        print(f"\n  -- {op}  ({len(nodes)} nodes, domain {dict(domains)})")

        signatures: Counter = Counter()
        other_attrs: Counter = Counter()
        for node in nodes:
            named = list(node.input) + [""] * len(spec)
            signatures[
                tuple(
                    describe(named[i], inits, shapes, bulk=spec[i] in BULK[op])
                    for i in range(len(spec))
                )
            ] += 1
            if op == "QLinearConv":
                a = attrs(node)
                other_attrs[tuple(sorted(k for k in a if k not in
                                         ("group", "strides", "pads", "dilations",
                                          "kernel_shape", "auto_pad")))] += 1
            else:
                other_attrs[tuple(sorted(attrs(node).items()))] += 1

        width = max(len(s) for s in spec)
        for sig, count in sorted(signatures.items(), key=lambda kv: -kv[1]):
            print(f"     {count:>4} node(s):")
            for name, value in zip(spec, sig):
                mark = "  (absent)" if value == "-" else ""
                print(f"           {name:<{width}}  {value}{mark}")
        for extra, count in other_attrs.items():
            print(f"     attributes beyond the geometry: {list(extra) or 'none'} ({count})")

        # a zero-point that is always zero is symmetric quantization, and it
        # removes a whole correction term from the kernel — worth knowing before
        # writing the general form
        zeros: Counter = Counter()
        for node in nodes:
            named = list(node.input) + [""] * len(spec)
            for slot, name in zip(spec, named):
                if not name.endswith("zero_point") or not name:
                    continue
                init = inits.get(name)
                if init is None:
                    zeros[f"{slot}: runtime"] += 1
                    continue
                values = onnx.numpy_helper.to_array(init)
                zeros[f"{slot}: {'all zero' if not values.any() else 'nonzero'}"] += 1
        if zeros:
            print(f"     zero-point values: {dict(zeros)}")

        if op == "QLinearConv":
            geoms: Counter = Counter()
            for node in nodes:
                geoms[conv_geometry(node, inits)] += 1
            print(f"     {len(geoms)} distinct geometries "
                  f"(C_in, C_out, kernel, group, strides, pads, dilations, auto_pad):")
            for geom, count in sorted(geoms.items(), key=lambda kv: -kv[1]):
                print(f"       {count:>4}  {geom}")

    # what else is in the graph: all-or-nothing means these matter as much as
    # the QLinear ops themselves
    rest = Counter(
        f"{n.domain}::{n.op_type}" if n.domain else n.op_type
        for n in graph.node
        if n.op_type not in SPECS
    )
    if rest:
        print(f"\n  -- everything else: {dict(rest)}")

    io = defaultdict(list)
    for v in list(graph.input) + list(graph.output):
        io["input" if v in graph.input else "output"].append(
            f"{v.name}:{dtype_name(v.type.tensor_type.elem_type)}"
        )
    print(f"  -- graph io: {dict(io)}")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(2)
    for p in paths:
        census(Path(p))
