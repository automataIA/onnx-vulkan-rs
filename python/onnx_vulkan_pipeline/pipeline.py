"""Content-addressed model bundles; Rust remains the compatibility authority."""
from __future__ import annotations

from contextlib import contextmanager
import collections
import functools
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

from . import policy
from .process import command, failure_reason, gpu_lock

ROOT = Path(os.environ.get("ONNX_VULKAN_ROOT", Path(__file__).resolve().parents[2]))


def _retain_evidence(error, source, failure, kind, record):
    """Replace `failure` with the files of `source` and the full error.

    One directory per destination, holding the latest failure: repeated
    attempts do not pile up directories. The worker's whole output goes to
    `error.log`; the note tells the CLI where, so the terminal keeps the verdict.
    """
    shutil.rmtree(failure, ignore_errors=True)
    shutil.copytree(source, failure)
    (failure / "error.log").write_text(f"{error}\n\nComplete worker output:\n{error.output}"
                                       if hasattr(error, "output") else str(error))
    write_json(failure / "failure.json", {"reason": failure_reason(error), "error": str(error),
                                          "type": type(error).__name__, **record})
    if hasattr(error, "add_note"):
        error.add_note(f"{kind} evidence retained at {failure}")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def binary(name):
    path = ROOT / "target/release" / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run cargo build --release --workspace --lib --bins --examples")
    return path


def dimensions(profile):
    result = []
    for name, value in sorted(profile.items()):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"--dim {name}={value}: sizes must be positive integers; "
                             "the runtime has no zero-size tensors")
        result += ["--dim", f"{name}={value}"]
    return result


def input_symbols(source, given=()):
    """Symbolic dimensions of the graph inputs, in order, without a value in `given`."""
    import onnx

    graph = onnx.load(str(source), load_external_data=False).graph
    initializers = {init.name for init in graph.initializer}
    symbols = [dim.dim_param for value in graph.input if value.name not in initializers
               for dim in value.type.tensor_type.shape.dim if dim.HasField("dim_param")]
    return [symbol for symbol in dict.fromkeys(symbols) if symbol not in given]


def kv_cache_example(symbols):
    """Worked `--dim` values for a decoder with a KV cache, or None.

    Its total length is past + new tokens, which no name makes obvious, and an
    empty cache (past 0) is a zero-size tensor."""
    past = [symbol for symbol in symbols if "past" in symbol]
    total = [symbol for symbol in symbols if "total" in symbol]
    batch = [symbol for symbol in symbols if "batch" in symbol]
    new = [symbol for symbol in symbols if symbol not in past + total + batch]
    if len(past) != 1 or len(total) != 1 or len(new) != 1:
        return None
    values = {**{symbol: 1 for symbol in batch}, past[0]: 1, new[0]: 4, total[0]: 5}
    return (f"a decoder with a KV cache: {total[0]} = {past[0]} + {new[0]}, all ≥ 1 (a past of 0 is a "
            f"zero-size tensor), e.g. " + " ".join(f"--dim {symbol}={values[symbol]}" for symbol in symbols))


def inspect_model(source, *, profile=None, timeout=300):
    result = command([binary("onnx-vulkan-tune"), "model-check", "--model", Path(source).resolve(),
                      *dimensions(profile or {})], timeout=timeout)
    return json.loads(result.stdout)


def seal(bundle, manifest):
    manifest["files"] = {
        str(p.relative_to(bundle)): digest(p)
        for p in sorted(bundle.rglob("*")) if p.is_file() and p != bundle / "manifest.json"
    }
    manifest["identity"] = _manifest_identity(manifest)
    write_json(bundle / "manifest.json", manifest)


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported bundle schema")
    for name, expected in manifest["files"].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Stale or modified bundle file: {name}")
    expected = _manifest_identity(manifest)
    if manifest.get("identity") != expected:
        raise ValueError("Bundle identity mismatch")
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob("*")
              if p.is_file() and p != bundle / "manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Bundle contains unsealed files")
    for name, profile in manifest.get("profiles", {}).items():
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid bundle profile name")
        for key in ("model", "tactics"):
            if profile.get(key) is not None and profile[key] not in manifest["files"]:
                raise ValueError(f"Unsealed profile {key}: {profile[key]}")
    return manifest


def _manifest_identity(manifest):
    return hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "identity"},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def tool_identity():
    versions = {}
    for name in ("onnx-vulkan-pipeline", "onnx", "onnxruntime", "numpy", "onnxscript", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    versions["binaries"] = {name: digest(ROOT / "target/release" / name) for name in
        ("model-runner", "onnx-vulkan-tune", "libonnxruntime_ep_vulkan.so", "examples/conv_blocked", "examples/matrix_tune")
        if (ROOT / "target/release" / name).is_file()}
    versions["runtime_libraries"] = {}
    for name, default in (("ORT_DYLIB_PATH", ROOT / "third_party/onnxruntime/linux-x64/lib/libonnxruntime.so"),
                          ("VULKAN_EP_PATH", ROOT / "target/release/libonnxruntime_ep_vulkan.so")):
        path = Path(os.environ.get(name) or default)
        if path.is_file():
            versions["runtime_libraries"][name] = digest(path)
    versions["pipeline_sources"] = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    versions["tuner_sources"] = {p.name: digest(p) for p in sorted((ROOT / "scripts/autotune").glob("*.py"))
                                 if not p.name.startswith("test_")}
    return versions


def _link_duplicates(directory, minimum=1 << 20):
    """Hardlink identical large files of one profile. A source with external
    data whose candidate keeps every weight (Gemma 3 1B: 859 MB) otherwise
    holds the same bytes twice, as `original.onnx`'s data and `candidate.data`."""
    seen = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.stat().st_size >= minimum:
            key = (path.stat().st_size, digest(path))
            if key in seen:
                linked = path.with_name(path.name + ".link")
                os.link(seen[key], linked)
                linked.replace(path)
            else:
                seen[key] = path


@contextmanager
def _preparation_workspace(destination):
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        try:
            yield Path(temporary)
        except BaseException as error:
            _retain_evidence(error, temporary, destination.with_name(f".{destination.name}.failed"),
                             "Preparation", {})
            raise


def prepare_bundle(source, destination, *, adapter=None, profiles=None, atol=1e-5, rtol=1e-4, timeout=300, quantization=None, quality=None):
    if not all(math.isfinite(t) and t >= 0 for t in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    if quality is not None:
        from . import quality as quality_module
        quality = quality_module.validate_config(quality)
        if quantization is not None:
            quality_module.check_quantization_compatible(quality, quantization)
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    profiles = profiles or {"default": {}}
    # Checked before any work, so that a typo leaves no failed-run directory.
    for name, profile in profiles.items():
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Profile names must contain only letters, digits, underscores or hyphens")
        dimensions(profile)
        # Without an adapter, prepare generates the inputs and needs every size.
        missing = input_symbols(source, profile) if adapter is None and Path(source).suffix == ".onnx" else []
        if missing:
            example = kv_cache_example(missing)
            raise ValueError(f"input dimensions without a value: {', '.join(missing)}; pass the size of your "
                             "real inputs for each, " + (example or "e.g. " + " ".join(
                                 f"--dim {symbol}=1" for symbol in missing)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _preparation_workspace(destination) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        manifest = {"schema_version": 1, "tools": tool_identity(), "atol": atol, "rtol": rtol,
                    "profiles": {}, "status": "prepared", "acceleration_demonstrated": False}
        for name, profile in profiles.items():
            directory = bundle / name
            config_path = Path(temporary) / "prepare-config.json"
            write_json(config_path, {"source": str(Path(source).resolve()), "destination": str(directory),
                                    "adapter": str(Path(adapter).resolve()) if adapter else None,
                                    "profiles": profile, "atol": atol, "rtol": rtol, "quantization": quantization,
                                    "quality": quality})
            command([sys.executable, "-m", "onnx_vulkan_pipeline._prepare_worker", config_path], timeout=timeout)
            _link_duplicates(directory)
            result = json.loads((directory / "preparation.json").read_text())
            check = inspect_model(directory / "model.onnx", profile=profile, timeout=timeout)
            if "quantization" in result:
                result["quantization"]["rust_pipeline_check"] = {"status": "passed" if check.get("compatible") else "incompatible", "verified": bool(check.get("compatible"))}
            write_json(directory / "compatibility.json", check)
            write_json(directory / "preparation.json", result)
            manifest["profiles"][name] = {"dimensions": profile, "model": f"{name}/model.onnx",
                                          "compatible": check.get("compatible", False), "tactics": None}
        manifest["compatibility_obtained"] = all(p["compatible"] for p in manifest["profiles"].values())
        seal(bundle, manifest)
        bundle.rename(destination)
    return verify_bundle(destination)


def _gpu_metrics(text):
    spec = importlib.util.spec_from_file_location("vulkan_parse_run", ROOT / "scripts/testsuite/parse_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.parse_gpu(text.splitlines())
    if result is not None:
        result["blocks"] = module.parse_blocks(text.splitlines())
    return result


def _expected_instrumentation(run_policy):
    """What the runtime must report as enabled under `run_policy`: the profiler
    and the span probe as the policy says, and no other probe at all."""
    measurement = run_policy["measurement"]
    return {"stats": measurement["stats"], "gpu_span": "VULKAN_GPU_SPAN" in measurement["probes"],
            "node_probe": False, "alloc_probe": False, "pool_probe": False}


def _require_instrumentation(report, run_policy):
    """The runner reports which instrumentation was active in its process. A
    timing taken with an ambient probe, or without the one it relies on, is not
    the timing of its mode."""
    observed = (report.get("measurement") or {}).get("instrumentation")
    if observed is None:
        raise RuntimeError("model-runner reported no instrumentation: rebuild it before measuring")
    expected = _expected_instrumentation(run_policy)
    if observed != expected:
        raise RuntimeError(f"Instrumentation of the run differs from its measurement mode: ran with {observed}, "
                           f"expected {expected}")


def _valid_span(span):
    """A GPU span usable as evidence: complete, covering the inference and finite."""
    if not isinstance(span, dict) or span.get("complete") is not True:
        return None
    busy, extent = span.get("busy_ms"), span.get("span_ms")
    if not span.get("flushes") or not span.get("command_buffers"):
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (busy, extent)):
        return None
    return span if 0 < busy <= extent else None


def _timing_row(report):
    """One interleaved timing: the backend's wall clock and GPU busy time of the
    same last inference. Without a valid span `gpu_ms` is None and the pair
    cannot promote anything."""
    spans = (report.get("measurement") or {}).get("standalone_gpu_span") or [None]
    span = spans[-1]
    valid = _valid_span(span)
    return {"wall_ms": report["standalone_ms"][-1], "gpu_ms": valid["busy_ms"] if valid else None,
            "gpu_span": span, "storage_peak_bytes": report.get("standalone_storage_peak_bytes"),
            "measurement_mode": "timing"}


def _default_reference(bundle, profile, model):
    """fp32 tensors check original.onnx; a quantized model.onnx has its own."""
    quantized = bundle / profile / "quantized-reference"
    if model == f"{profile}/model.onnx" and quantized.is_dir():
        return quantized
    return bundle / profile / "reference"


@functools.lru_cache(maxsize=8)
def _dynamic_quantizations(path):
    """Quantized outputs of the `DynamicQuantizeLinear` nodes, in graph order."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    return tuple(node.output[0] for node in graph.node
                 if node.op_type == "DynamicQuantizeLinear" and node.domain in ("", "ai.onnx"))


def _dynamic_quantization(bundle, profile, model):
    """`pipeline` if `prepare` quantized the run model dynamically, `source` if
    the model already held `DynamicQuantizeLinear`, else None.

    Its scales follow each tensor's extremes, so a one-ulp fp32 difference from
    the CPU grows layer by layer and `allclose` against the CPU cannot hold
    (docs/decisions.md §1). Such a model is gated by standalone/EP parity, by
    the per-node contract on its first quantized tensor and, when `prepare`
    quantized it with a quality config, by the application metric instead.
    """
    preparation = json.loads((bundle / profile / "preparation.json").read_text())
    if model == f"{profile}/model.onnx" and (preparation.get("quantization") or {}).get("mode") == "dynamic":
        return "pipeline"
    return "source" if _dynamic_quantizations(str(bundle / model)) else None


# Exposed tensors cannot share memory with the rest of the graph, and
# model-runner keeps a copy of each per backend (1 byte per uint8 element): all
# 137 quantized tensors of SAM 3 (997M elements) in one run peaked at 6.6 GB.
# Batches of this many elements bound larger models.
PER_NODE_ELEMENTS = 1_000_000_000


def _elements(value, dimensions):
    """Element count of an inferred tensor; None when its shape is unknown."""
    if value is None or not value.type.tensor_type.HasField("shape"):
        return None
    sizes = [d.dim_value if d.HasField("dim_value") else dimensions.get(d.dim_param)
             for d in value.type.tensor_type.shape.dim]
    return None if None in sizes else math.prod(sizes)


def _verify_quantized_tensors(bundle, manifest, profile, backend, timeout):
    """The per-node contract of dynamic quantization (docs/decisions.md §1).

    Every `DynamicQuantizeLinear` output is exposed as a graph output and
    compared with the CPU EP on the reference inputs, in graph order. While
    the quantized tensors upstream match exactly, only the fp32 summation
    order differs, so the first one that differs must be within one code;
    below it the moved scales cascade and nothing more is gated. The first
    quantized tensor alone is often the quantized input image, identical by
    construction. Tensors are exposed in batches of PER_NODE_ELEMENTS, one
    run each, until a batch holds the first that differs.
    """
    import onnx

    selected = manifest["profiles"][profile]
    path = bundle / selected["model"]
    tensors = _dynamic_quantizations(str(path))
    if not tensors:
        return None
    model = onnx.load(str(path), load_external_data=False)
    inferred = {value.name: value for value in onnx.shape_inference.infer_shapes(model).graph.value_info}
    onnx.load_external_data_for_model(model, str(path.parent))
    outputs = len(model.graph.output)
    existing = {output.name for output in model.graph.output}
    batches, size = [[]], 0
    for tensor in tensors:
        count = 0 if tensor in existing else _elements(inferred.get(tensor), selected["dimensions"])
        count = PER_NODE_ELEMENTS if count is None else count
        if batches[-1] and size + count > PER_NODE_ELEMENTS:
            batches, size = batches + [[]], 0
        batches[-1].append(tensor)
        size += count
    identical = 0
    with tempfile.TemporaryDirectory(prefix="vulkan-per-node-") as temporary:
        instrumented = Path(temporary) / "model.onnx"
        # The first save moves the weights to the sidecar and clears them from
        # `model`; later saves rewrite only the graph that points at it.
        onnx.save(model, str(instrumented), save_as_external_data=True, all_tensors_to_one_file=True,
                  location="model.onnx.data", size_threshold=1024)
        for batch in batches:
            del model.graph.output[outputs:]
            model.graph.output.extend(inferred.get(tensor)
                                      or onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.UINT8, None)
                                      for tensor in batch if tensor not in existing)
            onnx.save(model, str(instrumented))
            try:
                report = _run(bundle, manifest, profile, backend, 1, timeout, model=instrumented,
                              tactics=selected["tactics"],
                              reference=_default_reference(bundle, profile, selected["model"]))
            except Exception as error:
                error.add_note("Stage: the per-node check, which exposes the quantized tensors as outputs")
                raise
            results = {item["name"]: item for item in report["outputs"]}
            for tensor in batch:
                output = results[tensor]
                if output["dtype"] != output["candidate_dtype"] or output["max_abs"] > 0:
                    return {"total": len(tensors), "identical": identical, "tensor": tensor,
                            "max_abs": output["max_abs"], "runs": batches.index(batch) + 1,
                            "passed": output["dtype"] == output["candidate_dtype"] and output["max_abs"] <= 1}
                identical += 1
    return {"total": len(tensors), "identical": identical, "tensor": None, "max_abs": 0.0, "runs": len(batches),
            "passed": True}


def _require_metric(report):
    if report.get("gate") != "parity":
        return
    tensors = report.get("quantized_tensors")
    if tensors is not None and not tensors["passed"]:
        raise RuntimeError(f"Quantized tensor `{tensors['tensor']}` differs from the CPU EP by "
                           f"{tensors['max_abs']:g} codes while every quantized tensor before it matches; dynamic "
                           "quantization allows at most 1 (docs/decisions.md §1)")
    if report.get("dynamic_quantization") == "pipeline" and not report["quality_evaluation"].get("vulkan_verified"):
        raise RuntimeError("A dynamically quantized model needs its application metric verified on Vulkan "
                           "outputs, and it was not")


# The CPU EP is the reference, not the subject: past this many seconds per CPU
# session its timing has enough samples (SAM 3: ~10 s a run, 21 of them in each
# of two sessions exceeded the default timeout).
CPU_BUDGET_SECONDS = 30


def _run(bundle, manifest, profile, backend, samples, timeout, *, model=None, tactics=None, reference=None, output_data=False,
         cpu_timing=False, mode="diagnostic"):
    """One model-runner validation. `mode` is the measurement it runs under (`policy.MODES`):
    only `diagnostic` carries the profiler, whose timings explain a run but are not a timing."""
    selected = manifest["profiles"][profile]
    with tempfile.TemporaryDirectory(prefix="vulkan-validation-") as temporary:
        report_path = Path(temporary) / "report.json"
        args = [binary("model-runner"), bundle / (model or selected["model"]),
                "--iters", str(samples + 1), "--cpu-budget", str(CPU_BUDGET_SECONDS),
                "--no-opt", "--require-vulkan", "--tol", str(manifest["atol"]),
                "--rtol", str(manifest["rtol"]), "--report-json", report_path,
                *dimensions(selected["dimensions"])]
        if cpu_timing:
            # The --no-opt reference is for comparison; time ORT's optimized
            # CPU EP separately so the speed baseline is a fair one.
            args += ["--time-optimized-cpu"]
        if output_data:
            args += ["--output-data-json", Path(temporary) / "outputs.json"]
        reference = reference or _default_reference(bundle, profile, model or selected["model"])
        if reference.is_dir():
            args += ["--reference", reference]
        if backend == "standalone":
            args += ["--standalone"]
        dynamic = _dynamic_quantization(bundle, profile, model or selected["model"])
        if dynamic:
            if backend != "standalone":
                raise ValueError("A dynamically quantized model is verified by standalone/EP parity; "
                                 "run it with --backend standalone")
            args += ["--parity-gate"]
        run_policy = policy.resolve(tactics=tactics, ep_compile=True, mode=mode)
        env = {**policy.environment(run_policy, bundle=bundle), "RUST_LOG": "info" if mode == "diagnostic" else "warn"}
        try:
            result = command(args, cwd=ROOT, timeout=timeout, env=env)
        except BaseException as error:
            _retain_evidence(error, temporary, bundle.with_name(f".{bundle.name}.run-failed"), "Run", {
                "profile": profile, "backend": backend, "bundle": str(bundle), "identity": manifest.get("identity"),
                "command": [str(arg) for arg in args], "environment": env, "execution_policy": run_policy,
            })
            raise
        report = json.loads(report_path.read_text())
        within_tolerance = report.get("gate") == "parity" or all(o["passed"] for o in report["outputs"])
        if not report.get("passed") or not report.get("cpu_fallback_disabled") or not report["outputs"] or not within_tolerance:
            raise RuntimeError("Full-model validation did not pass")
        if backend == "standalone" and not report.get("standalone_outputs"):
            raise RuntimeError("Standalone validation missing")
        _require_instrumentation(report, run_policy)
        report["measurement_mode"] = mode
        report["dynamic_quantization"] = dynamic
        report["requested_backend"] = backend
        report["executed_backends"] = ["ort-cpu", "vulkan-ep"] + (["standalone"] if backend == "standalone" else [])
        report["tools"] = tool_identity()
        report["execution_policy"] = run_policy
        # Only the profiler prints the Pareto block these are read from.
        report["gpu"] = _gpu_metrics(result.stderr + "\n" + result.stdout) if mode == "diagnostic" else None
        if output_data:
            report["output_data"] = json.loads((Path(temporary) / "outputs.json").read_text())
        report["log"] = result.stdout + result.stderr
        return report


def unsupported_operators(issues):
    """Blocking compatibility issues by operator, most frequent first: `NonZero ×98, RoiAlign ×4`."""
    counts = collections.Counter(issue.get("op", "?") for issue in issues
                                 if issue.get("blocking") and issue.get("kind") != "shape")
    return ", ".join(f"{op} ×{count}" for op, count in counts.most_common())


def _device(timeout):
    """Identity of the device the runtime picks under the caller's device selection."""
    return json.loads(command([binary("examples/conv_blocked"), "--device-json"], timeout=timeout,
                              env=policy.environment(policy.resolve())).stdout)


def _require_compatible_policy(manifest):
    """A tuned artifact was measured and validated under the policy its manifest
    records (numerics, allocator, EP and device selection; its device is bound
    separately). One without a record predates policy identity and cannot be
    shown to match."""
    recorded = manifest.get("execution_policy")
    if recorded is None:
        raise ValueError("Tuned bundle records no execution policy (made before policy identity); "
                         "optimize again in a new directory")
    changes = policy.differences(recorded, policy.pipeline_identity(), policy.COMPATIBILITY)
    if changes:
        raise ValueError("Tuned bundle was optimized under a different execution policy ("
                         + "; ".join(changes) + "); restore it or optimize again in a new directory")


def _bundle_policy(manifest):
    """How the policy of this run compares with the one the bundle was sealed under.
    Untuned bundles run under any policy; the report says which one it was."""
    recorded = manifest.get("execution_policy")
    if recorded is None:
        return {"status": "unrecorded", "differences": []}
    changes = policy.differences(recorded, policy.pipeline_identity())
    return {"status": "different" if changes else "match", "differences": changes}


def run_bundle(bundle, *, profile="default", backend="standalone", samples=20, timeout=300):
    if backend not in ("standalone", "ep") or samples < 1:
        raise ValueError("Expected backend ep/standalone and positive samples")
    bundle = Path(bundle).resolve()
    manifest = verify_bundle(bundle)
    selected = manifest["profiles"][profile]
    if not selected["compatible"]:
        compatibility = bundle / profile / "compatibility.json"
        issues = json.loads(compatibility.read_text()).get("issues", []) if compatibility.is_file() else []
        operators = unsupported_operators(issues)
        raise ValueError(f"Profile `{profile}` cannot run: Vulkan does not implement "
                         + (f"these operators, or these forms of them: {operators}" if operators else "every node")
                         + f", and there is no CPU fallback; details in {compatibility}")
    with gpu_lock(timeout):
        if selected["tactics"]:
            if manifest.get("device") != _device(timeout) or manifest.get("tools") != tool_identity():
                raise ValueError("Tuned bundle targets different tools or device; optimize again in a new directory")
            _require_compatible_policy(manifest)
        result = _run(bundle, manifest, profile, backend, samples, timeout, tactics=selected["tactics"],
                      cpu_timing=True)
        result["bundle_execution_policy"] = _bundle_policy(manifest)
        result["quality_evaluation"] = _verify_quality(bundle, manifest, profile, backend, timeout)
        if result.get("gate") == "parity":
            result["quantized_tensors"] = _verify_quantized_tensors(bundle, manifest, profile, backend, timeout)
        _require_metric(result)
        return result


def _verify_quality(bundle, manifest, profile, backend, timeout):
    from .quality import verify_quality
    return verify_quality(bundle, manifest, profile, backend, timeout)


@contextmanager
def _optimization_workspace(source, destination, manifest, samples, tune):
    staging = destination.with_name(f".{destination.name}.incomplete")
    identity = {"source": manifest["identity"], "tools": tool_identity(), "samples": samples, "tune": tune,
                "profiles": manifest["profiles"], "atol": manifest["atol"], "rtol": manifest["rtol"],
                "execution_policy": policy.pipeline_identity(),
                # How the timings are taken; rows measured under another plan are never mixed in.
                "measurement": policy.measurement_plan(),
                # The tuner subprocess runs under the caller's settings, which the pipeline's own
                # runs override; without tuning there is no such launch.
                "tuner_execution_policy": policy.resolve() if tune else None}
    if staging.exists():
        recorded = json.loads((staging / "resume.json").read_text())
        if recorded != identity:
            if "execution_policy" not in recorded or "tuner_execution_policy" not in recorded:
                reason = " (it predates execution-policy identity)"
            elif "measurement" not in recorded:
                reason = " (it predates measurement-mode identity)"
            elif recorded["execution_policy"] != identity["execution_policy"]:
                reason = " (execution policy changed: " + "; ".join(
                    policy.differences(recorded["execution_policy"], identity["execution_policy"])) + ")"
            elif recorded.get("tune") == tune and recorded["tuner_execution_policy"] != identity["tuner_execution_policy"]:
                reason = " (tuner execution policy changed: " + "; ".join(
                    policy.differences(recorded["tuner_execution_policy"], identity["tuner_execution_policy"])) + ")"
            elif recorded["measurement"] != identity["measurement"]:
                reason = " (measurement modes changed: " + "; ".join(
                    policy.differences(recorded["measurement"], identity["measurement"])) + ")"
            else:
                reason = ""
            raise ValueError(f"Stale experiment at {staging}{reason}; choose a new output directory")
    else:
        staging.mkdir()
        write_json(staging / "resume.json", identity)
        shutil.copytree(source, staging / "bundle")
        for name in manifest["profiles"]:
            (staging / "bundle" / name / "benchmark.json").unlink(missing_ok=True)
            # A copy does not keep the source bundle's hardlinks.
            _link_duplicates(staging / "bundle" / name)
    try:
        yield staging
    except BaseException as error:
        write_json(staging / "failure.json", {"reason": failure_reason(error), "error": str(error),
                                              "type": type(error).__name__})
        raise
    else:
        shutil.rmtree(staging)


REQUIRED_GAIN = 0.10


MIN_PAIRS = 20


def _gain(baseline, candidate, key, threshold):
    """Whether `key` fell by more than the required gain and than twice the
    noise of either side. A missing or invalid value never gains."""
    a, b = [x.get(key) for x in baseline], [x.get(key) for x in candidate]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in a + b):
        return False
    noise = max(statistics.stdev(a), statistics.stdev(b)) * 2
    return statistics.median(a) - statistics.median(b) > max(statistics.median(a) * threshold, noise)


def _improvement(baseline, candidate, *, threshold=REQUIRED_GAIN):
    """The timing verdict of interleaved pairs: GPU evidence and wall must both
    gain; wall alone cannot promote a candidate. Whether the candidate also keeps
    its coverage, transfers and partitioning is `_diagnostics_hold`'s question."""
    if len(baseline) < MIN_PAIRS or len(candidate) < MIN_PAIRS:
        return False
    return all(_gain(baseline, candidate, key, threshold) for key in ("gpu_ms", "wall_ms"))


def _confirmed(baseline, candidate, *, threshold=REQUIRED_GAIN):
    """The uninstrumented confirmation: no probe, no profiler, so only the wall
    clock a user sees can speak, and it must show the same gain."""
    if len(baseline) < MIN_PAIRS or len(candidate) < MIN_PAIRS:
        return False
    return _gain(baseline, candidate, "wall_ms", threshold)


def _graph_evidence(bundle, model, inspection):
    """The inspection of one graph under the profile's dimensions, with the
    digest of the file inspected so that two sides are told apart by content."""
    execution = inspection.get("execution") or {}
    return {"model": model, "digest": digest(bundle / model), "compatible": bool(inspection.get("compatible")),
            **{key: execution.get(key) for key in ("total_nodes", "supported_nodes", "unsupported_nodes")}}


def _diagnostic_evidence(report, graph, tactics):
    """What one profiled run of a graph says about its coverage, transfers and
    partitioning. Deterministic, so one run per side is enough."""
    gpu = report.get("gpu") or {}
    blocks = gpu.get("blocks") or {}
    return {"graph": graph, "tactics": tactics, "passed": bool(report.get("passed")),
            "cpu_fallback_disabled": bool(report.get("cpu_fallback_disabled")),
            "flushes": gpu.get("flushes"), "upload_mb": gpu.get("upload_mb"), "download_mb": gpu.get("download_mb"),
            "convex_blocks": blocks.get("convex_blocks"), "nodes_claimed": blocks.get("nodes_claimed")}


def _coverage(evidence):
    """(supported, total) nodes of a graph that ran verified on Vulkan with the
    CPU fallback disabled, else None. The counts belong to the graph itself:
    a rewrite may be smaller, so its claimed node count is not comparable."""
    graph = evidence.get("graph") or {}
    total, supported = graph.get("total_nodes"), graph.get("supported_nodes")
    claimed = evidence.get("nodes_claimed")
    if not (graph.get("compatible") and evidence.get("passed") and evidence.get("cpu_fallback_disabled")):
        return None
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (total, supported, claimed)):
        return None
    if graph.get("unsupported_nodes") != total - supported:
        return None
    return (supported, total) if 0 < claimed and 0 < total and 0 <= supported <= total else None


TRANSFER_KEYS = ("flushes", "upload_mb", "download_mb", "convex_blocks")


def _diagnostics_hold(source, candidate):
    """Why a faster candidate still must not replace the source, else None.

    Fails closed: evidence that is absent or inconsistent is a reason. Coverage
    is the fraction of each graph the Vulkan runtime supports, so a smaller
    rewritten graph passes while one that leaves nodes to a fallback does not;
    transfers and partitions may not grow.
    """
    reasons = []
    covered = {name: _coverage(evidence) for name, evidence in (("source", source), ("candidate", candidate))}
    for name, value in covered.items():
        if value is None:
            reasons.append(f"{name} graph has no verified Vulkan coverage evidence")
    if all(covered.values()):
        (s_sup, s_tot), (c_sup, c_tot) = covered["source"], covered["candidate"]
        if c_sup * s_tot < s_sup * c_tot:
            reasons.append(f"candidate covers {c_sup}/{c_tot} nodes, source {s_sup}/{s_tot}")
    for key in TRANSFER_KEYS:
        before, after = source.get(key), candidate.get(key)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (before, after)):
            reasons.append(f"{key} not measured")
        elif after > before:
            reasons.append(f"{key} grew from {before:g} to {after:g}")
    return reasons or None


# Pairs measured before a hopeless candidate may stop early.
FUTILITY_PAIRS = 5


def _futile(baseline, candidate, *, threshold=REQUIRED_GAIN):
    """Whether no continuation of the interleaved runs can promote the candidate.

    A futility bound of a sequential test: it only ends a run early with
    `compatible_no_measured_gain`; promotion still needs every sample and
    `_improvement`. A missing or invalid time is final, since `_improvement`
    refuses it. Otherwise the gain is projected two per-sample standard
    deviations in the candidate's favor, more than twice the standard error of
    the difference of the medians at FUTILITY_PAIRS pairs, and the run stops
    only if even that misses the required gain.
    """
    pairs = min(len(baseline), len(candidate))
    if pairs < FUTILITY_PAIRS:
        return False
    for key in ("gpu_ms", "wall_ms"):
        a, b = [x[key] for x in baseline[:pairs]], [x[key] for x in candidate[:pairs]]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in a + b):
            return True
        optimistic = statistics.median(a) - statistics.median(b) + 2 * max(statistics.stdev(a), statistics.stdev(b))
        if optimistic < statistics.median(a) * threshold:
            return True
    return False


def _medians(rows, keys=("wall_ms", "gpu_ms")):
    """Median wall and GPU time of interleaved runs, for the human report only."""
    result = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        result[f"median_{key}"] = (statistics.median(values)
                                   if values and all(isinstance(v, (int, float)) for v in values) else None)
    return result


def _tuning_priority(report):
    """Aggregate measured dispatch time by tunable ONNX operator family."""
    priorities = {"Conv": 0.0, "MatMul": 0.0, "Gemm": 0.0}
    for row in (report.get("gpu") or {}).get("pareto", []):
        op, milliseconds = row["op"], row["ms"]
        if not math.isfinite(milliseconds) or milliseconds < 0:
            raise ValueError("Invalid GPU profile time")
        family = ("Conv" if op in {"Conv", "Conv16", "Conv32", "Conv_grouped", "Conv_split", "Conv_split_reduce"}
                  else "MatMul" if op in {"MatMul", "MatMul16"}
                  else "Gemm" if op in {"Gemm", "Gemm16"} else None)
        if family:
            priorities[family] += milliseconds
    return priorities


# What scripts/autotune/model.py prints when the tuned model misses the tolerance.
TUNING_REJECTED = "model autotune rejected"


def _divergence(error):
    """The first output model-runner found beyond tolerance, else the failure's reason.

    The error keeps only the end of the runner's output, which may have lost
    the EP lines but keeps the standalone and reference comparisons.
    """
    match = re.search(r"^\s*(?:output|standalone vs CPU|reference \w+)\s+(\S+)\s.*?max\|Δ\|=(\S+)"
                      r".*?beyond tolerance: ([1-9]\d*)", str(error), re.M)
    if not match:
        return failure_reason(error)
    return f"output {match[1]}: max|Δ|={match[2]}, {match[3]} value(s) beyond tolerance"


def _load_rows(path, mode):
    """Rows of interleaved runs kept for a resume, trimmed to the completed pairs.
    Rows taken under another measurement mode are an error, never mixed in."""
    rows = json.loads(path.read_text()) if path.exists() else {"baseline": [], "candidate": []}
    for label, values in rows.items():
        stray = [row.get("measurement_mode") for row in values if row.get("measurement_mode") != mode]
        if stray:
            raise ValueError(f"{path} holds {label} rows measured as {stray[0]!r}, not {mode!r}; "
                             "choose a new output directory")
    completed = min(len(rows["baseline"]), len(rows["candidate"]))
    return {label: values[:completed] for label, values in rows.items()}


def _interleave(rows, path, samples, measure, *, futile=None):
    """Alternates baseline/candidate (AB, then BA, ...) until `samples` pairs exist,
    appending `measure(label)` to `rows` and saving after each run. Returns
    "futility" when `futile` ends the run early."""
    for index in range(samples):
        if futile and futile(rows["baseline"], rows["candidate"]):
            return "futility"
        for label in (("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")):
            if len(rows[label]) > index:
                continue
            rows[label].append(measure(label))
            write_json(path, rows)
    return None


def _diagnose(bundle, manifest, name, timeout, baseline_model, baseline_check, candidate_model, candidate_check,
              tactics):
    """Profiled runs of the source and the candidate graph, each inspected on its
    own file: a rewrite is judged against its own coverage, not the source's."""
    path = bundle / name / "diagnostics.json"
    if path.exists():
        return json.loads(path.read_text())
    sides = {}
    for label, model, inspection, side_tactics in (("baseline", baseline_model, baseline_check, None),
                                                   ("candidate", candidate_model, candidate_check, tactics)):
        report = _run(bundle, manifest, name, "standalone", 1, timeout, mode="diagnostic",
                      model=model if label == "baseline" else None, tactics=side_tactics)
        sides[label] = _diagnostic_evidence(report, _graph_evidence(bundle, model, inspection), side_tactics)
    write_json(path, sides)
    return sides


def optimize_bundle(source, destination, *, samples=20, timeout=300, tune=True):
    if samples < 20:
        raise ValueError("Optimization requires at least 20 interleaved samples")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    manifest = verify_bundle(source)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _optimization_workspace(source, destination, manifest, samples, tune) as temporary:
        bundle = Path(temporary) / "bundle"
        experiment = json.loads((Path(temporary) / "resume.json").read_text())
        experiment_tools, experiment_policy = experiment["tools"], experiment["execution_policy"]
        tuner_policy = experiment["tuner_execution_policy"]
        outcomes = {}
        with gpu_lock(timeout):
            device = _device(timeout)
            device_path = Path(temporary) / "device.json"
            if device_path.exists() and json.loads(device_path.read_text()) != device:
                raise ValueError("Device or implementation changed since interrupted experiment")
            write_json(device_path, device)
            manifest["device"] = device
            for name, profile in manifest["profiles"].items():
                if not profile["compatible"]:
                    outcomes[name] = {"status": "incompatible"}
                    continue
                check = inspect_model(bundle / profile["model"], profile=profile["dimensions"], timeout=timeout)
                can_tune = tune and check.get("tuning", {}).get("tunable_workloads", 0) > 0
                tactics = f"{name}/tactics.json" if can_tune else None
                if can_tune:
                    measured_profile = _run(bundle, manifest, name, "standalone", 1, timeout)
                    write_json(bundle / name / "tuning-profile.json", measured_profile)
                    priority_path = bundle / name / "tuning-priority.json"
                    write_json(priority_path, _tuning_priority(measured_profile))
                rejected = None
                if can_tune:
                    reference = _default_reference(bundle, name, profile["model"])
                    # The file describes the tuner's launch. The tuner sets its own tuning
                    # mode and artifact for its nested validation runs; everything else
                    # it runs under is recorded here.
                    write_json(bundle / name / "tuning-policy.json", tuner_policy)
                    try:
                        command(["uv", "run", "--no-project", ROOT / "scripts/autotune/model.py",
                                 bundle / profile["model"], "--output", bundle / tactics, "--no-build",
                                 "--priority-json", priority_path,
                                 "--atol", str(manifest["atol"]), "--rtol", str(manifest["rtol"]),
                                 *(["--reference", reference] if reference.is_dir() else []),
                                 *dimensions(profile["dimensions"])], cwd=ROOT, timeout=timeout,
                                env=policy.environment(tuner_policy))
                    except RuntimeError as error:
                        # The tuner validates on the bundle's reference inputs;
                        # tuned kernels that miss its tolerance there are
                        # dropped rather than failing the run.
                        if TUNING_REJECTED not in str(error):
                            raise
                        tactics, rejected = None, _divergence(error)
                original = f"{name}/original.onnx"
                original_check = inspect_model(bundle / original, profile=profile["dimensions"], timeout=timeout)
                baseline_model, baseline_check = ((original, original_check) if original_check.get("compatible")
                                                  else (profile["model"], check))
                benchmark_path = bundle / name / "benchmark.json"
                rows = _load_rows(benchmark_path, "timing")
                # The same model with the same kernels on both sides: nothing to measure.
                same_model = (baseline_model == profile["model"]
                              or digest(bundle / baseline_model) == digest(bundle / profile["model"]))
                stopped = "identical" if same_model and tactics is None else None

                def timing(label):
                    return _timing_row(_run(bundle, manifest, name, "standalone", 1, timeout, mode="timing",
                                            model=baseline_model if label == "baseline" else None,
                                            tactics=tactics if label == "candidate" else None))

                stopped = stopped or _interleave(rows, benchmark_path, samples, timing, futile=_futile)
                improved = not stopped and _improvement(rows["baseline"], rows["candidate"])
                # Diagnostics are profiled runs of the two graphs, apart from the timed pairs:
                # they are deterministic, so they are taken once and kept for a resume.
                diagnostics = held = confirmation = None
                if improved:
                    diagnostics = _diagnose(bundle, manifest, name, timeout, baseline_model, baseline_check,
                                            profile["model"], check, tactics)
                    held = _diagnostics_hold(diagnostics["baseline"], diagnostics["candidate"])
                    improved = held is None
                if improved:
                    confirmation_path = bundle / name / "confirmation.json"
                    confirmation = _load_rows(confirmation_path, "confirmation")

                    def uninstrumented(label):
                        report = _run(bundle, manifest, name, "standalone", 1, timeout, mode="confirmation",
                                      model=baseline_model if label == "baseline" else None,
                                      tactics=tactics if label == "candidate" else None)
                        return {"wall_ms": report["standalone_ms"][-1], "measurement_mode": "confirmation"}

                    _interleave(confirmation, confirmation_path, samples, uninstrumented)
                    improved = _confirmed(confirmation["baseline"], confirmation["candidate"])
                profile["model"] = profile["model"] if improved else baseline_model
                profile["tactics"] = tactics if improved else None
                verification = _run(bundle, manifest, name, "standalone", 20, timeout, tactics=profile["tactics"])
                verification["quality_evaluation"] = _verify_quality(bundle, manifest, name, "standalone", timeout)
                if verification.get("gate") == "parity":
                    verification["quantized_tensors"] = _verify_quantized_tensors(bundle, manifest, name, "standalone",
                                                                                  timeout)
                _require_metric(verification)
                write_json(bundle / name / "validation.json", verification)
                # The medians explain the decision; the timing verdict, the diagnostics and the
                # uninstrumented confirmation alone make it.
                outcomes[name] = {"status": "accelerated" if improved else "compatible_no_measured_gain",
                                  "baseline_model": baseline_model, "samples": samples, "tuned": can_tune,
                                  "tactics_rejected": rejected,
                                  "pairs": min(len(rows["baseline"]), len(rows["candidate"])), "stopped": stopped,
                                  "required_gain": REQUIRED_GAIN, "baseline": _medians(rows["baseline"]),
                                  "candidate": _medians(rows["candidate"]),
                                  "measurement": {"timing": "timing", "gpu_metric": "gpu_span.busy_ms",
                                                  "diagnostics_hold": held,
                                                  "diagnostics": diagnostics,
                                                  "confirmation": None if confirmation is None else {
                                                      "pairs": min(len(confirmation["baseline"]),
                                                                   len(confirmation["candidate"])),
                                                      "baseline": _medians(confirmation["baseline"], ("wall_ms",))["median_wall_ms"],
                                                      "candidate": _medians(confirmation["candidate"], ("wall_ms",))["median_wall_ms"]}}}
            if tool_identity() != experiment_tools:
                raise ValueError("Tools changed during optimization; rerun in a new directory")
            if policy.pipeline_identity() != experiment_policy:
                raise ValueError("Execution policy changed during optimization; rerun in a new directory")
            manifest["tools"] = experiment_tools
            if tune and policy.resolve() != tuner_policy:
                raise ValueError("Tuner execution policy changed during optimization; rerun in a new directory")
            manifest["execution_policy"] = experiment_policy
            manifest["tuner_execution_policy"] = tuner_policy
            manifest["status"] = "verified" if all(p["compatible"] for p in manifest["profiles"].values()) else "incompatible"
            manifest["acceleration_demonstrated"] = any(o["status"] == "accelerated" for o in outcomes.values())
            write_json(bundle / "report.json", outcomes)
            seal(bundle, manifest)
        bundle.rename(destination)
    return verify_bundle(destination)
