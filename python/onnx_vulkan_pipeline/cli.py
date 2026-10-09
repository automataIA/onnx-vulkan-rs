"""Command-line entry point; all work is implemented in the public services."""
import argparse
import json
from pathlib import Path
import collections
import re
import statistics
import sys
import subprocess

from .pipeline import (inspect_model, input_symbols, kv_cache_example, prepare_bundle, optimize_bundle, run_bundle,
                       unsupported_operators)
from .research import research
from .process import failure_reason
from .search import search_quantization

DESCRIPTIONS = {
    "inspect": "Report whether the Rust Vulkan pipeline supports every node of an ONNX model.",
    "prepare": "Verify an ONNX model on the CPU and seal it into a bundle directory with reference outputs.",
    "run": "Run a bundle on the Vulkan GPU and compare its outputs with the CPU reference.",
    "optimize": "Tune Vulkan kernels for a bundle and keep the result only if it is measurably faster.",
    "research": "Bounded automatic EP code experiments in git worktrees; results are review proposals. "
                "Needs a prepared bundle, a git checkout and a problem JSON (python/README.md, EP research).",
    "quantize-search": "Try several quantization settings and optimize the fastest one that keeps quality. "
                       "Needs the --quantization config that prepare takes (python/README.md, quantization).",
}
EPILOG = """\
Typical flow: inspect → prepare → run → optimize. See python/README.md.

The JSON result goes to stdout when it is redirected or piped (> result.json, | jq);
on a terminal only the verdict lines (onnx-vulkan: ...) are printed, on stderr.

Exit status: 0 success; 1 failure (bad input, a worker error, outputs beyond
tolerance); 2 inspect/prepare worked but the model cannot run on Vulkan.
`optimize` exits 0 when it keeps the untuned model (compatible_no_measured_gain).
A failed prepare/optimize keeps its evidence in a hidden .OUTPUT.failed or
.OUTPUT.incomplete directory beside --output, named in the message."""
SOURCES = {
    "inspect": "ONNX model file",
    "prepare": "ONNX model file",
    "run": "bundle directory written by prepare or optimize",
    "optimize": "bundle directory written by prepare",
    "research": "problem JSON naming a prepared bundle, a proposer and four gates (see python/README.md)",
    "quantize-search": "ONNX model file",
}
# Pattern in a failure → what the user can do about it.
HINTS = (
    (r"beyond tolerance: [1-9]",
     "Vulkan and CPU outputs differ by more than the bundle tolerance, which `prepare` fixed "
     "(--atol/--rtol). If these differences are acceptable for your application, prepare again "
     "with explicit, larger tolerances."),
    ("GPU lane is busy", "another onnx-vulkan process is using the GPU; wait for it or raise --timeout."),
)
# Above this relative error a divergence is a wrong result, not rounding.
GROSS_RELATIVE = 1e-2
GROSS_HINT = ("differences this large are wrong results, not rounding: do not loosen the tolerances. "
              "If this happened while tuning, `optimize --no-tune` keeps the default kernels; "
              "please report the model and the failing output.")
ARGMAX_HINT = ("every value is within tolerance, but the largest element of an output moved to another "
               "index whose reference value is not tied with the top one: for a classifier that is a "
               "different answer, which larger tolerances would hide rather than fix.")


def failure_summary(error):
    """A short ending for a failure whose full worker output was printed above it."""
    text = getattr(error, "output", None) or str(error)
    offending = [line.strip() for line in text.splitlines() if re.search(r"beyond tolerance: [1-9]", line)]
    argmax = [line.strip() for line in text.splitlines() if line.rstrip().endswith("(different answer)")]
    lines = [failure_reason(error)] + ["  " + line for line in (offending or argmax)[:8]]
    relative = [float(value) for line in offending for value in re.findall(r"relative=([0-9.eE+-]+)", line)]
    if relative and max(relative) > GROSS_RELATIVE:
        return lines + [f"hint: {GROSS_HINT}"]
    if argmax and not offending:
        return lines + [f"hint: {ARGMAX_HINT}"]
    lines += [f"hint: {hint}" for pattern, hint in HINTS if re.search(pattern, f"{error}\n{text}")]
    return lines


def _signature(source):
    """One line per graph input: name, element type and dimensions. Inputs that
    differ only by numbers, with the same type (a KV cache's 52
    `past_key_values.N.key`/`.value`), share one line."""
    import onnx

    graph = onnx.load(str(source), load_external_data=False).graph
    initializers = {init.name for init in graph.initializer}
    groups = {}
    for value in graph.input:
        if value.name in initializers:
            continue
        kind = (f"{onnx.TensorProto.DataType.Name(value.type.tensor_type.elem_type).lower()} ["
                + ", ".join(str(dim.dim_value) if dim.HasField("dim_value") else dim.dim_param or "?"
                            for dim in value.type.tensor_type.shape.dim) + "]")
        numbers = [int(number) for number in re.findall(r"\d+", value.name)]
        groups.setdefault((re.sub(r"\d+", "#", value.name), kind), []).append((value.name, numbers))
    lines = []
    for (template, kind), names in groups.items():
        if len(names) == 1:
            lines.append(f"  input {names[0][0]}: {kind}")
        else:
            numbers = sorted({tuple(found) for _, found in names})
            lines.append(f"  {len(names)} inputs {template}: {kind} (# from {'.'.join(map(str, numbers[0]))} "
                         f"to {'.'.join(map(str, numbers[-1]))})")
    return lines


def _without_class(line):
    """A failure line without this package's exception class path: the
    message is for a person, and the class is in the full output."""
    return re.sub(r"^onnx_vulkan_pipeline\.[\w.]*?(\w+Error): ", "", line)


def _inspect_json(result):
    """`inspect`'s printed result: each issue once with its nodes, and the shapes
    known only at run time counted. Per node, FasterRCNN printed 6048 issues
    (1.3 MB), almost all `shape` diagnostics; `inspect_model()` and a bundle's
    `compatibility.json` keep them per node."""
    issues = result.get("issues", [])
    runtime = [issue for issue in issues if issue.get("kind") == "shape" and not issue.get("blocking")]
    groups = {}
    for issue in issues:
        if issue not in runtime:
            key = tuple(issue.get(field) for field in ("kind", "op", "reason", "blocking"))
            groups.setdefault(key, []).append(issue.get("node"))
    return {**result,
            "issues": [{"kind": kind, "op": op, "reason": reason, "blocking": blocking, "nodes": nodes}
                       for (kind, op, reason, blocking), nodes in groups.items()],
            "runtime_shape_nodes": len({issue.get("node") for issue in runtime})}


# Operators that a quantization pass writes, counted in a quantized model.
QUANTIZED_OPS = re.compile(r"^(QLinear\w+|QGemm|QAttention|\w+Integer|MatMulNBits|DynamicQuantizeLinear|"
                           r"QuantizeLinear|DequantizeLinear)$")


def _quantized_operators(model):
    """`QLinearConv ×57, QuantizeLinear ×3`: what quantization put in a model."""
    import onnx

    graph = onnx.load(str(model), load_external_data=False).graph
    counts = collections.Counter(node.op_type for node in graph.node if QUANTIZED_OPS.match(node.op_type))
    return ", ".join(f"{op} ×{count}" for op, count in counts.most_common())


def _quantization(bundle, profile):
    """`static QOperator quantization`, or None for a model prepare did not quantize."""
    path = Path(bundle) / profile / "preparation.json"
    report = json.loads(path.read_text()).get("quantization") if path.is_file() else None
    return report and " ".join(filter(None, [report.get("mode"), report.get("format"), "quantization"]))


def _metric(report, backend):
    """The quality verdict: the metric before and after, its limit and direction."""
    better = "higher" if report.get("higher_is_better", True) else "lower"
    within = report["drop"] <= report["max_drop"]
    return (f"application metric {report.get('metric')} {report['before']:.4g} → {report['after']:.4g} on "
            f"{backend} ({better} is better): drop {report['drop']:.4g}, "
            f"{'within' if within else 'beyond'} the limit max_drop={report['max_drop']:g}")


def _median_ms(values):
    """Median of timed iterations; the first one includes pipeline creation."""
    values = values[1:] or values
    return f"{statistics.median(values):.2f} ms" if values else "n/a"


def success_summary(command, args, result):
    if command == "inspect":
        tuning, execution = result.get("tuning", {}), result.get("execution", {})
        # Only input symbols take a --dim; the issues also name symbols that
        # shape inference derived from them (`floor(height/14)`).
        symbols = input_symbols(args.source, {item.split("=", 1)[0] for item in args.dim})
        # Tunable workloads need concrete shapes: ResNet50 counts 52 with
        # --dim N=1 and 0 without.
        counted = (f"{tuning.get('tunable_workloads')} tunable workloads" if not symbols
                   else "tunable workloads counted only with every --dim given")
        lines = [f"compatible: {'yes' if result.get('compatible') else 'no'} "
                 f"({execution.get('supported_nodes')}/{execution.get('total_nodes')} nodes supported, {counted})"]
        lines += _signature(args.source)
        others = result.get("issues", [])
        if symbols:
            example = kv_cache_example(symbols)
            lines.append(f"  input dimensions without a value: {', '.join(symbols)}; prepare needs the size of your "
                         f"real inputs for each, as {' '.join(f'--dim {s}=<size>' for s in symbols)}")
            if example:
                lines.append(f"  {example}")
        runtime = sorted({issue.get("node", "?") for issue in others if issue.get("kind") == "shape"})
        if runtime:
            lines.append(f"  {len(runtime)} node(s) have shapes known only at run time "
                         f"({', '.join(runtime[:5])}{', …' if len(runtime) > 5 else ''}): "
                         "informational, prepare checks them on real inputs")
            others = [issue for issue in others if issue.get("kind") != "shape"]
        operators = unsupported_operators(others)
        if operators:
            lines.append(f"  not implemented on Vulkan, as operators or in these forms: {operators}")
            lines.append("  so the model cannot run on Vulkan end to end: there is no CPU fallback, and `run` "
                         "and `optimize` refuse it")
            others = [issue for issue in others if not issue.get("blocking")]
        lines += [f"  {issue.get('node')} ({issue.get('op')}): {issue.get('reason')}" for issue in others[:10]]
        if len(others) > 10:
            lines.append(f"  … {len(others) - 10} more issues in the JSON above")
        return lines
    if command == "prepare":
        compatible = result.get("compatibility_obtained")
        rejected = sorted({(step["name"], profile) for profile in result.get("profiles", {})
                           for step in json.loads((Path(args.output) / profile / "preparation.json").read_text())
                           .get("transformations", []) if step.get("status") == "rejected"})
        quantized = []
        for profile in result.get("profiles", {}):
            report = json.loads((Path(args.output) / profile / "preparation.json").read_text()).get("quantization")
            if report:
                quantized += [f"{_quantization(args.output, profile)} ({profile}): "
                              f"{_quantized_operators(Path(args.output) / profile / 'model.onnx') or 'no operator'}",
                              f"  {_metric(report, 'the CPU EP')}"]
        # Hardlinked weights count once, as they take disk space once.
        size = sum({(stat.st_dev, stat.st_ino): stat.st_size for path in Path(args.output).rglob("*")
                    if path.is_file() for stat in [path.stat()]}.values())
        return [f"bundle written to {args.output} ({size / 1e6:.1f} MB); Vulkan compatible: "
                f"{'yes' if compatible else 'no'}",
                *quantized,
                "reference inputs: " + ("from the adapter's example_inputs()" if args.adapter
                                        else "deterministic generated values; pass --adapter for real ones"),
                *(f"{name} rejected ({profile}): it changed the CPU outputs beyond atol/rtol, so the bundle "
                  "keeps the model without it; see preparation.json" for name, profile in rejected),
                *([f"next: uv run onnx-vulkan run {args.output}"] if compatible else
                  [f"`run` and `optimize` refuse profile {profile}: Vulkan does not implement "
                   f"{unsupported_operators(json.loads(path.read_text()).get('issues', [])) or 'every node'}; "
                   f"details in {path}"
                   for profile in result.get("profiles", {})
                   for path in [Path(args.output) / profile / "compatibility.json"]
                   if path.is_file() and not json.loads(path.read_text()).get("compatible")])]
    if command == "run":
        # The comparison reference runs without graph optimizations; when the
        # optimized CPU EP was timed too, that is the fair CPU baseline.
        cpu_times = result.get("cpu_optimized_ms") or result.get("cpu_ms", [])
        cpu = (f"CPU EP {_median_ms(cpu_times)}" if result.get("cpu_optimized_ms")
               else f"CPU EP without graph optimizations {_median_ms(cpu_times)}")
        if 1 < len(cpu_times) < len(result.get("candidate_ms", [])):
            # model-runner's --cpu-budget ended the CPU sessions early.
            cpu += f" ({len(cpu_times) - 1} timed run(s) within the CPU time budget)"
        backends = [name for name in result.get("executed_backends", []) if name != "ort-cpu"]
        if result.get("gate") == "parity":
            largest = max((output["max_abs"] for output in result.get("outputs", [])), default=0.0)
            gate = ("the application metric below is" if result.get("dynamic_quantization") == "pipeline"
                    else "the quantized tensors below are")
            agreement = (f"{' and '.join(backends)} outputs are identical; the CPU EP differs by up to "
                         f"max|Δ|={largest:.3g}, expected for dynamic quantization, "
                         f"so {gate} the gate")
        else:
            ratios = [output.get("max_tolerance_ratio") for output in
                      result.get("outputs", []) + result.get("standalone_outputs", [])]
            worst = max((ratio for ratio in ratios if ratio is not None), default=None)
            agreement = (f"{' and '.join(backends)} outputs match the CPU reference within "
                         f"atol={result['atol']:g}, rtol={result['rtol']:g}: every element has "
                         "|Δ| ≤ atol + rtol·|reference|"
                         + (f", the worst at {worst:.0%} of that allowance" if worst is not None else ""))
        quality = result.get("quality_evaluation") or {}
        lines = [agreement]
        tensors = result.get("quantized_tensors")
        if tensors:
            lines.append(f"quantized tensors, in graph order: {tensors['identical']} of {tensors['total']} identical "
                         "to the CPU EP" + (f"; the first that differs, {tensors['tensor']}, is within "
                                            f"±1 code (max|Δ|={tensors['max_abs']:g})" if tensors["tensor"] else ""))
        if "drop" not in quality and result.get("gate") == "parity":
            lines.append("application quality not measured: the source was already quantized and has no "
                         "quality metric; check it on your own data")
        elif "drop" not in quality:
            # An elementwise match proves the GPU computes what the CPU
            # computes, not that the model is right for the application.
            adapter = json.loads((Path(args.source) / args.profile / "preparation.json").read_text()).get("adapter")
            lines.append("on " + ("the adapter's example inputs" if adapter else "deterministic generated inputs")
                         + "; this is GPU-vs-CPU agreement, not application quality")
        # Single runs on a shared host, with logging and profiling on: not
        # benchmark numbers; optimize measures that.
        lines.append(f"validation timings, not a benchmark: {cpu}, "
                     f"Vulkan EP {_median_ms(result.get('candidate_ms', []))}"
                     + (f", standalone {_median_ms(result['standalone_ms'])}" if result.get("standalone_ms") else ""))
        if "drop" in quality:
            quantization = json.loads((Path(args.source) / args.profile / "preparation.json").read_text()
                                      ).get("quantization") or {}
            lines.append(_metric({**quantization, **quality}, "Vulkan"))
        return lines
    if command == "optimize":
        report = json.loads((Path(args.output) / "report.json").read_text())
        lines = []
        for name, outcome in report.items():
            baseline, candidate = outcome.get("baseline", {}), outcome.get("candidate", {})

            def ms(row, key):
                return "n/a" if row.get(key) is None else f"{row[key]:.3f} ms"
            skipped = "--no-tune" if args.no_tune else "the model has no tunable workloads"
            lines.append(f"profile {name}: {outcome['status']}"
                         + ("" if outcome.get("tuned", True) else f" (no tuning: {skipped})"))
            if outcome.get("tactics_rejected"):
                lines.append("  tuned kernels rejected: on the bundle's reference inputs they miss its tolerance "
                             f"({outcome['tactics_rejected']}); default kernels kept")
            if outcome.get("stopped") == "identical":
                lines.append("  not measured: the prepared model is the original one, with default kernels")
            if outcome.get("stopped") == "futility":
                lines.append(f"  stopped after {outcome['pairs']} of {outcome['samples']} pairs (one original and "
                             "one prepared sample each): even two standard deviations in the prepared model's favor "
                             "would miss the required gain")
            quantization = _quantization(args.output, name)
            if baseline.get("median_wall_ms") is not None:
                kernels = "tuned" if outcome.get("tuned", True) and not outcome.get("tactics_rejected") else "default"
                prepared = f"prepared model ({quantization})" if quantization else "prepared model"
                lines.append(f"  original model → {prepared} with {kernels} kernels, both on Vulkan: "
                             f"wall {ms(baseline, 'median_wall_ms')} → {ms(candidate, 'median_wall_ms')}, "
                             f"GPU {ms(baseline, 'median_gpu_ms')} → {ms(candidate, 'median_gpu_ms')} "
                             f"(promotion needs ≥{outcome['required_gain']:.0%} on both, beyond noise)")
                # Each sample runs in a new process, after one warm-up iteration.
                lines.append("  each sample is a separate process, so walls exceed `run`'s in-process median; "
                             "`run` compares Vulkan with the CPU")
            if outcome["status"] == "compatible_no_measured_gain" and outcome.get("baseline_model", "").endswith(
                    "/original.onnx"):
                lines.append("  the output bundle runs the original model"
                             + (f", not the quantized one ({quantization}), which was not faster" if quantization
                                else ", since the prepared one was not faster"))
        if report:
            lines.append(f"next: uv run onnx-vulkan run {args.output}")
        return lines
    if command == "quantize-search":
        optimized = result.get("optimized") or {}
        rows = []
        for item in result.get("candidates", []):
            detail = (f"standalone {item['median_wall_ms']:.2f} ms, Vulkan metric drop {item.get('vulkan_drop', 0):g}"
                      if item.get("status") == "passed" else item.get("reason", ""))
            # Full reasons, with paths, stay in search.json.
            detail = detail if len(detail) <= 160 else detail[:159] + "…"
            rows.append(f"  {item['name']}: {item.get('status')} — {detail}")
        return rows + [f"selected candidate: {result.get('selected') or 'none passed'}"
                + (f"; acceleration demonstrated: {'yes' if optimized.get('acceleration_demonstrated') else 'no'}"
                   if optimized else ""),
                f"details: {Path(args.output) / 'search.json'}"]
    return []


def main():
    parser = argparse.ArgumentParser(
        prog="onnx-vulkan",
        description="Prepare, verify, tune and run ONNX models on the Vulkan GPU runtime.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name in DESCRIPTIONS:
        sub = commands.add_parser(name, help=DESCRIPTIONS[name].split(". ")[0], description=DESCRIPTIONS[name],
                                  epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
        sub.add_argument("source", type=Path, help=SOURCES[name])
        sub.add_argument("--timeout", type=float, default=300, help="seconds allowed per worker process (default 300)")
        if name in ("prepare", "optimize", "research", "quantize-search"):
            sub.add_argument("--output", type=Path, required=True, help="new directory to create; must not exist")
        if name in ("inspect", "prepare", "quantize-search"):
            sub.add_argument("--dim", action="append", default=[], metavar="SYMBOL=N",
                             help="value of a symbolic input dimension, e.g. --dim batch_size=1; repeatable")
        if name in ("prepare", "quantize-search"):
            sub.add_argument("--adapter", type=Path,
                             help="Python file with load_model()/example_inputs() for export or real inputs; "
                                  "the quality metric is the quantization config's own 'adapter' key")
            sub.add_argument("--quantization", type=Path, required=name == "quantize-search",
                             help="quantization config JSON: datasets, quality adapter, metric and threshold "
                                  "(see python/README.md)")
            sub.add_argument("--profiles", type=Path, help="JSON object mapping profile names to symbol/value objects")
            sub.add_argument("--atol", type=float, default=1e-5,
                             help="absolute tolerance of every later CPU/Vulkan comparison (default 1e-5)")
            sub.add_argument("--rtol", type=float, default=1e-4,
                             help="relative tolerance of every later CPU/Vulkan comparison (default 1e-4)")
        if name == "quantize-search":
            sub.add_argument("--grid", type=Path, help="JSON list of option overrides; default covers static, dynamic and int4")
        if name in ("optimize", "run", "quantize-search"):
            sub.add_argument("--samples", type=int, default=20, help="timed iterations (default 20)")
        if name in ("optimize", "quantize-search"):
            sub.add_argument("--no-tune", action="store_true", help="skip kernel tuning; only measure the bundle")
        if name == "run":
            sub.add_argument("--backend", choices=("ep", "standalone"), default="standalone",
                             help="ep: ONNX Runtime CPU and the Vulkan EP; standalone (default): those and the "
                                  "pure Rust runtime, which needs no ONNX Runtime")
            sub.add_argument("--profile", default="default", help="bundle profile to run (default: default)")
        if name == "research":
            sub.add_argument("--attempts", type=int, default=3,
                             help="proposals to try, each in its own autoresearch/* worktree (default 3)")
            sub.add_argument("--budget-seconds", type=float, default=1800,
                             help="wall-clock limit for the whole experiment; no attempt starts after it "
                                  "(default 1800)")
    args = parser.parse_args()
    try:
        if args.command in ("inspect", "prepare", "quantize-search"):
            dims = {key: int(value) for key, value in (item.split("=", 1) for item in args.dim)}
        if args.command == "inspect":
            result = inspect_model(args.source, profile=dims, timeout=args.timeout)
        elif args.command == "prepare":
            profiles = json.loads(args.profiles.read_text()) if args.profiles else {"default": dims}
            result = prepare_bundle(args.source, args.output, adapter=args.adapter, profiles=profiles,
                                    atol=args.atol, rtol=args.rtol, timeout=args.timeout,
                                    quantization=json.loads(args.quantization.read_text()) if args.quantization else None)
        elif args.command == "quantize-search":
            profiles = json.loads(args.profiles.read_text()) if args.profiles else {"default": dims}
            result = search_quantization(args.source, args.output, json.loads(args.quantization.read_text()),
                                         grid=json.loads(args.grid.read_text()) if args.grid else None,
                                         adapter=args.adapter, profiles=profiles, atol=args.atol, rtol=args.rtol,
                                         samples=args.samples, timeout=args.timeout, tune=not args.no_tune)
        elif args.command == "optimize":
            result = optimize_bundle(args.source, args.output, samples=args.samples, timeout=args.timeout, tune=not args.no_tune)
        elif args.command == "run":
            result = run_bundle(args.source, backend=args.backend, profile=args.profile, samples=args.samples, timeout=args.timeout)
        else:
            result = research(args.source, args.output, attempts=args.attempts, budget_seconds=args.budget_seconds, timeout=args.timeout)
        text = json.dumps(_inspect_json(result) if args.command == "inspect" else result, indent=2, allow_nan=False)
        if sys.stdout.isatty():
            # Hundreds of lines a person scrolls past; scripts redirect stdout.
            print("onnx-vulkan: JSON result not shown on a terminal; redirect stdout to keep it "
                  "(> result.json)", file=sys.stderr)
        else:
            print(text)
        # The JSON above is for scripts; the verdict goes last, where a person looks.
        for line in success_summary(args.command, args, result):
            print(f"onnx-vulkan: {line}", file=sys.stderr)
        # The command worked but the model cannot run on Vulkan: 2, so that a
        # script chaining inspect → prepare → run stops; 1 means it failed.
        unrunnable = ((args.command == "inspect" and not result.get("compatible"))
                      or (args.command == "prepare" and not result.get("compatibility_obtained")))
        return 2 if unrunnable else 0
    except FileExistsError as error:
        print(f"onnx-vulkan: {error.filename or error} already exists; choose a new --output directory",
              file=sys.stderr)
        return 1
    except (ValueError, OSError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        summary = failure_summary(error)
        notes = getattr(error, "__notes__", [])
        evidence = [note.rsplit(" at ", 1)[1] for note in notes if " evidence retained at " in note]
        failed = f"{args.command} failed" + "".join(f" during {note.removeprefix('Stage: ')}"
                                                    for note in notes[-1:] if note.startswith("Stage: "))
        if isinstance(error, subprocess.TimeoutExpired):
            # The exception's own text is the whole command line.
            print(f"onnx-vulkan: {failed}: {Path(str(error.cmd[0])).name} did not finish within "
                  f"{error.timeout:g} s", file=sys.stderr)
            print("onnx-vulkan: hint: raise --timeout, the limit for each worker process; on a large model the "
                  "CPU reference alone can take seconds per sample"
                  + (", and fewer --samples shorten `run`" if args.command == "run" else ""), file=sys.stderr)
            for path in evidence[-1:]:
                print(f"onnx-vulkan: evidence: {path}", file=sys.stderr)
            return 1
        if evidence and "\n" in str(error):
            # The worker's whole output is in the evidence; the terminal keeps the verdict.
            print(f"onnx-vulkan: {failed}: {_without_class(summary[0])}", file=sys.stderr)
            for line in summary[1:]:
                print(f"onnx-vulkan: {line}", file=sys.stderr)
            print(f"onnx-vulkan: full output: {Path(evidence[-1]) / 'error.log'}", file=sys.stderr)
            return 1
        print(f"onnx-vulkan: {error}", file=sys.stderr)
        if "\n" in str(error) or summary[1:]:
            print(f"\nonnx-vulkan: {failed}: {_without_class(summary[0])}", file=sys.stderr)
            for line in summary[1:]:
                print(f"onnx-vulkan: {line}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
