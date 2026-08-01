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


def _gelu_ok(node, _consts) -> bool:
    v = _attr(node, "approximate", "none")
    return (v.decode() if isinstance(v, bytes) else v) in {"none", "tanh"}


def _simplified_layernorm_ok(node, _consts) -> bool:
    # the optional second output `inv_std_var` is not produced
    return len([o for o in node.output if o]) == 1


def _skip_simplified_layernorm_ok(node, _consts) -> bool:
    present = lambda i: len(node.input) > i and node.input[i] != ""  # noqa: E731
    out = lambda i: len(node.output) > i and node.output[i] != ""  # noqa: E731
    # `beta`/`bias` unimplemented; `mean`/`inv_std_var` not produced, the
    # residual sum (output 3) is
    return (
        all(present(i) for i in (0, 1, 2))
        and not present(3)
        and not present(4)
        and out(0)
        and not out(1)
        and not out(2)
    )


def _rotary_ok(node, _consts) -> bool:
    return (
        _attr(node, "interleaved", 0) == 0
        and _attr(node, "is_packed_batching", 0) == 0
        and _attr(node, "scale", 1.0) == 1.0
        and len(node.input) == 4
        and all(i != "" for i in node.input)
        and len(node.output) == 1
    )


def _argmax_ok(node, _consts) -> bool:
    # the kernel keeps the first occurrence, so the reversed tie-break is refused
    return _attr(node, "select_last_index", 0) == 0


def _gqa_ok(node, _consts) -> bool:
    present = lambda i: len(node.input) > i and node.input[i] != ""  # noqa: E731
    rotary = _attr(node, "do_rotary", 0) != 0
    return (
        _attr(node, "softcap", 0.0) == 0.0
        and _attr(node, "rotary_interleaved", 0) == 0
        and _attr(node, "smooth_softmax", 0) <= 0
        and all(present(i) for i in (1, 2, 3, 4))
        and (not rotary or (present(7) and present(8)))
        and len(node.output) == 3
        and all(o != "" for o in node.output)
    )


def _matmul_nbits_ok(node, _consts) -> bool:
    present = lambda i: len(node.input) > i and node.input[i] != ""  # noqa: E731
    block_size = _attr(node, "block_size", 0)
    return (
        _attr(node, "bits", 0) == 4
        and block_size > 0
        and block_size % 8 == 0
        and _attr(node, "K", 0) > 0
        and _attr(node, "N", 0) > 0
        and present(2)
        and present(3)
        and not present(4)
        and not present(5)
    )


def _qlinear_params_ok(node, consts, pairs, weights=None, bias=None) -> bool:
    """The QOperator constraints, mirroring `unsupported_quantization`.

    Unlike the other rules this one reads *values*, which is why the Rust half
    of it does not live in `is_implemented_node`: a `NodeIr` has no
    initializers. Here they are one attribute away, so both halves are checked
    together and the coverage number stays the truth.
    """
    quant = consts.params
    for scale, zero, _slot in pairs:
        s, z = quant.get(_input(node, scale)), quant.get(_input(node, zero))
        if s is None or z is None:
            return False
        if s[0] != TensorProto.FLOAT or len(s[1]) != 1 or len(z[1]) != 1:
            return False
    if weights is not None:
        scale, zero = weights
        s, z = quant.get(_input(node, scale)), quant.get(_input(node, zero))
        if s is None or z is None or s[0] != TensorProto.FLOAT:
            return False
        if len(s[1]) != len(z[1]):
            return False
        # per-channel weights must be symmetric: no `w_zp[c]·Σx` correction
        if len(z[1]) > 1 and any(v != 0 for v in z[1]):
            return False
    if bias is not None:
        b = quant.get(_input(node, bias))
        if b is not None and b[0] != TensorProto.INT32:
            return False
    return True


def _input(node, index: int) -> str:
    return node.input[index] if len(node.input) > index else ""


def _qlinear_conv_ok(node, consts) -> bool:
    v = _attr(node, "auto_pad", "NOTSET")
    v = v.decode() if isinstance(v, bytes) else v
    return (
        v in {"NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"}
        and 8 <= len(node.input) <= 9
        and all(node.input[i] for i in range(8))
        and len(node.output) == 1
        and _qlinear_params_ok(
            node, consts, [(1, 2, "x"), (6, 7, "y")], weights=(4, 5), bias=8
        )
    )


def _qlinear_matmul_ok(node, consts) -> bool:
    return (
        len(node.input) == 8
        and all(node.input)
        and len(node.output) == 1
        and _qlinear_params_ok(node, consts, [(1, 2, "a"), (6, 7, "y")], weights=(4, 5))
    )


def _qlinear_add_ok(node, consts) -> bool:
    return (
        len(node.input) == 8
        and all(node.input)
        and len(node.output) == 1
        and _qlinear_params_ok(
            node, consts, [(1, 2, "A"), (4, 5, "B"), (6, 7, "C")]
        )
    )


def _qlinear_global_average_pool_ok(node, consts) -> bool:
    return (
        _attr(node, "channels_last", 0) == 0
        and len(node.input) == 5
        and all(node.input)
        and len(node.output) == 1
        and _qlinear_params_ok(node, consts, [(1, 2, "x"), (3, 4, "y")])
    )


def _gather_block_quantized_ok(node, _consts) -> bool:
    present = lambda i: len(node.input) > i and node.input[i] != ""  # noqa: E731
    block_size = _attr(node, "block_size", 0)
    return (
        _attr(node, "bits", 4) == 4
        and block_size > 0
        and block_size % 2 == 0
        and _attr(node, "gather_axis", 0) == 0
        and _attr(node, "quantize_axis", 1) == 1
        and present(2)
        and present(3)
    )


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
    "ArgMax": _argmax_ok,
    "GridSample": _grid_sample_ok,
    "ScatterND": _scatter_nd_ok,
    "Gelu": _gelu_ok,
    "SimplifiedLayerNormalization": _simplified_layernorm_ok,
    "SkipSimplifiedLayerNormalization": _skip_simplified_layernorm_ok,
    "RotaryEmbedding": _rotary_ok,
    "GroupQueryAttention": _gqa_ok,
    "MatMulNBits": _matmul_nbits_ok,
    "GatherBlockQuantized": _gather_block_quantized_ok,
    "QLinearConv": _qlinear_conv_ok,
    "QLinearMatMul": _qlinear_matmul_ok,
    "QLinearAdd": _qlinear_add_ok,
    "QLinearGlobalAveragePool": _qlinear_global_average_pool_ok,
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


class Constants(dict):
    """Integer constants, plus the quantization parameters keyed separately.

    A plain `dict` for the rules that resolve an `axes` input, with `.params`
    carrying `(dtype, values)` for the QOperator rules — those need the element
    type and the length, not just the numbers.
    """

    def __init__(self, ints: dict, params: dict):
        super().__init__(ints)
        self.params = params


def quant_params(graph) -> dict[str, tuple[int, list]]:
    """Every small numeric initializer, as `(dtype, values)`.

    "Small" is the point: scales and zero points are one value or one per
    output channel, and reading their contents is what tells a symmetric
    quantization from an asymmetric one. The weight tensor itself is skipped.
    """
    out: dict[str, tuple[int, list]] = {}
    for init in graph.initializer:
        # the model is loaded without external data, so a tensor stored outside
        # the file has no values to read here; quantization parameters never are
        if init.data_location == TensorProto.EXTERNAL:
            continue
        size = 1
        for d in init.dims:
            size *= d
        if size > 1 << 16:
            continue
        out[init.name] = (init.data_type, numpy_helper.to_array(init).ravel().tolist())
    return out


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


def opset_bounds() -> tuple[int, dict[str, int]]:
    """`MAX_OPSET` and `MIN_OPSET`, read from the Rust.

    Unlike `NODE_RULES` these are *data*, not logic, so they are parsed rather
    than transcribed: there is nothing to keep in sync by hand and therefore
    nothing to drift.
    """
    src = INTERP.read_text()
    ceiling = re.search(r"pub const MAX_OPSET: i32 = (\d+);", src)
    if not ceiling:
        sys.exit(f"cannot read MAX_OPSET from {INTERP}")
    table = re.search(r"const MIN_OPSET: &\[\(&str, i32\)\] = &\[(.*?)\];", src, re.S)
    if not table:
        sys.exit(f"cannot read MIN_OPSET from {INTERP}")
    mins = {op: int(v) for op, v in re.findall(r'\("([A-Za-z0-9_]+)",\s*(\d+)\)', table.group(1))}
    return int(ceiling.group(1)), mins


def float_only_ops() -> set[str]:
    """`FLOAT_ONLY_INPUT0`, read from the Rust for the same reason."""
    src = INTERP.read_text()
    body = re.search(r"const FLOAT_ONLY_INPUT0: &\[&str\] = &\[(.*?)\];", src, re.S)
    if not body:
        sys.exit(f"cannot read FLOAT_ONLY_INPUT0 from {INTERP}")
    return set(re.findall(r'"([A-Za-z0-9_]+)"', body.group(1)))


FLOAT = TensorProto.FLOAT


def value_dtypes(graph) -> dict[str, int]:
    """Element type of every value the graph declares or produces.

    The engine gets these from its frontend's inference; here the declared
    types are enough, because what the check refuses is a value whose type the
    file *states* and the kernel cannot read.
    """
    out: dict[str, int] = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.type.HasField("tensor_type"):
            out[vi.name] = vi.type.tensor_type.elem_type
    for init in graph.initializer:
        out[init.name] = init.data_type
    return out


def graph_ops(path: Path, known: set[str]) -> tuple[Counter[str], Counter[str]]:
    """Histogram of the graph's ops and of only the **non-claimed** nodes.

    The non-standard domain is kept in the name (`com.microsoft::GQA`): contrib
    ops and same-named standard ops are different things. External weights are
    not loaded.
    """
    model = onnx.load(str(path), load_external_data=False)
    max_opset, min_opset = opset_bounds()
    float_only = float_only_ops()
    opsets = {o.domain: o.version for o in model.opset_import}
    counts: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    stack = [model.graph]
    while stack:
        graph = stack.pop()
        consts = Constants(int_constants(graph), quant_params(graph))
        dtypes = value_dtypes(graph)
        for node in graph.node:
            name = node.op_type if not node.domain else f"{node.domain}::{node.op_type}"
            counts[name] += 1
            # coverage is decided on the bare op name, domain included in the
            # display only: `is_implemented` matches `node.op` and never looks
            # at the domain, so a contrib op we implement (`GroupQueryAttention`)
            # must count as covered here too
            rule = NODE_RULES.get(node.op_type)
            # `ai.onnx` only: contrib domains are versioned on their own axis
            version = opsets.get(node.domain, 0) if node.domain in ("", None) else 0
            in0 = node.input[0] if node.input and node.input[0] else None
            if node.op_type not in known:
                missing[name] += 1
            elif version and not (
                min_opset.get(node.op_type, 1) <= version <= max_opset
            ):
                missing[f"{name} (opset {version})"] += 1
            elif (
                node.op_type in float_only
                and in0 in dtypes
                and dtypes[in0] != FLOAT
            ):
                missing[f"{name} (element type {dtypes[in0]})"] += 1
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
