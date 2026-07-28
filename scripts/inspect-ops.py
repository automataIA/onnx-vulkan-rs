#!/usr/bin/env -S uv run --quiet --with onnx --with numpy --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx", "numpy"]
# ///
"""Engine op coverage on real ONNX models.

For each model it counts the graph's operators and diffs them against what the
interpreter declares it can run. The check is **per node**, not by op name:
`is_implemented_node` also looks at attributes, so a `ReduceSum` with axes as
input or a `MaxPool` with `ceil_mode = 1` is not claimed even when the name is
in the list.

What is read from the Rust source (`crates/onnx-vulkan-core/src/interp.rs`),
so it cannot diverge:

- the list of names, from `is_implemented`;
- the **set of constrained ops**, from the match arms of
  `is_implemented_node`.

The actual constraints are rewritten in `NODE_RULES` below — Python does not
run Rust. The duplication is kept honest by comparing the two sets: if someone
adds or removes an arm in the Rust without updating `NODE_RULES`, the script
**exits with an error** instead of reporting optimistic numbers.

    ./scripts/inspect-ops.py models/zoo/*/*.onnx
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent.parent
INTERP = ROOT / "crates/onnx-vulkan-core/src/interp.rs"


def implemented_ops() -> set[str]:
    """Ops listed by `is_implemented`, read from the Rust source."""
    src = INTERP.read_text()
    body = re.search(r"pub fn is_implemented\b.*?matches!\((.*?)\)\n\}", src, re.S)
    if not body:
        sys.exit(f"cannot read is_implemented from {INTERP}")
    return set(re.findall(r'"([A-Za-z0-9_]+)"', body.group(1)))


def constrained_ops() -> set[str]:
    """Ops that have a dedicated arm in `is_implemented_node`."""
    src = INTERP.read_text()
    body = re.search(r"pub fn is_implemented_node\b.*?\n\}\n", src, re.S)
    if not body:
        sys.exit(f"cannot read is_implemented_node from {INTERP}")
    arms = re.findall(
        r'^\s+((?:"[A-Za-z0-9_]+"\s*\|\s*)*"[A-Za-z0-9_]+")\s*=>',
        body.group(0),
        re.M,
    )
    return {name for arm in arms for name in re.findall(r'"([A-Za-z0-9_]+)"', arm)}


def _attr(node, name: str, default):
    for a in node.attribute:
        if a.name == name:
            return helper.get_attribute_value(a)
    return default


def _one_axis(node, consts) -> bool:
    axes = _attr(node, "axes", None)
    if axes is None and len(node.input) > 1 and node.input[1]:
        # `fold_constant_params` canonicalization: axes passed as a constant
        # input are promoted to an attribute before the check
        axes = consts.get(node.input[1])
    return axes is not None and len(axes) == 1


def _pool_ok(node, _consts) -> bool:
    return _attr(node, "ceil_mode", 0) == 0 and len(node.output) == 1


def _resize_ok(node, _consts) -> bool:
    def s(name: str, default: str) -> str:
        v = _attr(node, name, default)
        return v.decode() if isinstance(v, bytes) else v

    return (
        _attr(node, "exclude_outside", 0) == 0
        and s("mode", "nearest") in {"nearest", "linear", "cubic"}
        and s("coordinate_transformation_mode", "half_pixel")
        in {"half_pixel", "asymmetric", "align_corners", "pytorch_half_pixel"}
        and s("nearest_mode", "round_prefer_floor")
        in {"round_prefer_floor", "round_prefer_ceil", "floor", "ceil"}
    )


def _conv_ok(node, _consts) -> bool:
    v = _attr(node, "auto_pad", "NOTSET")
    v = v.decode() if isinstance(v, bytes) else v
    return v in {"NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"}


def _conv_transpose_ok(node, _consts) -> bool:
    v = _attr(node, "auto_pad", "NOTSET")
    v = v.decode() if isinstance(v, bytes) else v
    has_output_shape = any(a.name == "output_shape" for a in node.attribute)
    return v in {"NOTSET", "VALID"} and not has_output_shape


def _grid_sample_ok(node, _consts) -> bool:
    def s(name: str, default: str) -> str:
        v = _attr(node, name, default)
        return v.decode() if isinstance(v, bytes) else v

    return s("mode", "bilinear") == "bilinear" and s("padding_mode", "zeros") in {
        "zeros",
        "border",
    }


def _scatter_nd_ok(node, _consts) -> bool:
    v = _attr(node, "reduction", "none")
    return (v.decode() if isinstance(v, bytes) else v) == "none"


#: Per-node constraints, mirroring the arms of `is_implemented_node`.
NODE_RULES = {
    "Resize": _resize_ok,
    "MaxPool": _pool_ok,
    "AveragePool": _pool_ok,
    "Conv": _conv_ok,
    "ConvInteger": _conv_ok,
    "ConvTranspose": _conv_transpose_ok,
    "ReduceMean": _one_axis,
    "ReduceSum": _one_axis,
    "ReduceMax": _one_axis,
    "ReduceMin": _one_axis,
    "GridSample": _grid_sample_ok,
    "ScatterND": _scatter_nd_ok,
}


def check_rules_in_sync() -> None:
    """Fails if the Rust arms and `NODE_RULES` do not coincide."""
    rust, here = constrained_ops(), set(NODE_RULES)
    if rust == here:
        return
    lines = [f"NODE_RULES out of sync with is_implemented_node in {INTERP}:"]
    if rust - here:
        lines.append(f"  constraints in Rust but not here: {sorted(rust - here)}")
    if here - rust:
        lines.append(f"  constraints here but not in Rust: {sorted(here - rust)}")
    sys.exit("\n".join(lines))


def int_constants(graph) -> dict[str, list[int]]:
    """Integer values known at load-time: initializers and outputs of `Constant` nodes.

    This is what `fold_constant_params` can resolve on the Rust side.
    """
    out: dict[str, list[int]] = {}
    for init in graph.initializer:
        if init.data_type in (TensorProto.INT64, TensorProto.INT32):
            out[init.name] = numpy_helper.to_array(init).ravel().tolist()
    for node in graph.node:
        if node.op_type != "Constant" or node.domain:
            continue
        for a in node.attribute:
            if a.name == "value" and a.t.data_type in (
                TensorProto.INT64,
                TensorProto.INT32,
            ):
                out[node.output[0]] = numpy_helper.to_array(a.t).ravel().tolist()
    return out


def graph_ops(path: Path, known: set[str]) -> tuple[Counter[str], Counter[str]]:
    """Histogram of the graph's ops and of only the **non-claimed** nodes.

    The non-standard domain is kept in the name (`com.microsoft::GQA`): contrib
    ops and same-named standard ops are different things. External weights are
    not loaded.
    """
    model = onnx.load(str(path), load_external_data=False)
    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    stack = [model.graph]
    while stack:
        graph = stack.pop()
        consts = int_constants(graph)
        for node in graph.node:
            name = node.op_type if not node.domain else f"{node.domain}::{node.op_type}"
            counts[name] += 1
            rule = NODE_RULES.get(name) if not node.domain else None
            if name not in known:
                missing[name] += 1
            elif rule is not None and not rule(node, consts):
                # known name but non-claimable node: counts as a gap, and is
                # the distinction that the by-name count was missing
                missing[f"{name} (unsupported shape)"] += 1
            for attr in node.attribute:
                if attr.HasField("g"):
                    stack.append(attr.g)
                stack.extend(attr.graphs)
    return counts, missing


def opset(path: Path) -> str:
    model = onnx.load(str(path), load_external_data=False)
    return ", ".join(
        f"{o.domain or 'ai.onnx'}={o.version}" for o in model.opset_import
    )


def main(paths: list[str]) -> int:
    check_rules_in_sync()
    known = implemented_ops()
    missing_total: Counter[str] = Counter()

    for p in paths:
        path = Path(p)
        counts, missing = graph_ops(path, known)
        total = sum(counts.values())
        covered = total - sum(missing.values())

        print(f"\n=== {path.relative_to(ROOT) if path.is_absolute() else path}")
        print(f"    opset: {opset(path)}")
        print(f"    nodes: {total} | covered: {covered} ({100 * covered / max(total, 1):.1f}%)")
        if missing:
            print(f"    missing ops ({len(missing)} types, {sum(missing.values())} nodes):")
            for op, n in missing.most_common():
                print(f"      {n:6d}  {op}")
            missing_total.update(missing)
        else:
            print("    full coverage")

    if missing_total:
        print("\n=== aggregate missing ops (by number of nodes)")
        for op, n in missing_total.most_common():
            print(f"  {n:6d}  {op}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [str(p) for p in sorted((ROOT / "models/zoo").rglob("*.onnx"))]
    sys.exit(main(args))
