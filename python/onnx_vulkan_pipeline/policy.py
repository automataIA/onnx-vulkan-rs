"""Versioned execution-policy record: the one source of the runtime settings a
subprocess is given and of the identity that binds evidence to them.

Only an allowlist of execution-affecting variables is read, never the whole
environment. Each value is canonicalized with the semantics the Rust runtime
applies to it, so an absent default and its explicit equivalent identify
equally and an ignored value (`ONNX_VULKAN_ATTENTION_FP16=true`) does not
change the identity. The record is what a path executes *after* its explicit
overrides: `resolve` applies them, and `environment` turns the same record into
the child's variables, so command construction and identity cannot drift.
"""
from __future__ import annotations

import os
import re

VERSION = 1

# Read by the runtime with presence semantics: set to anything, even `0`, means on.
# `VULKAN_EP_STATS` is a measurement mode of its own and is handled apart.
PROBES = ("VULKAN_GPU_SPAN", "NODE_PROBE", "ALLOC_PROBE", "POOL_PROBE")

# Vulkan loader and driver selectors that decide which devices and layers the
# runtime can see. Values are kept verbatim: an empty string is not assumed to
# equal an unset variable. The device itself is identified separately
# (`conv_blocked --device-json`) and bound next to this record.
DEVICE_SELECTION = (
    "VK_ICD_FILENAMES", "VK_DRIVER_FILES", "VK_ADD_DRIVER_FILES",
    "VK_LOADER_DRIVERS_SELECT", "VK_LOADER_DRIVERS_DISABLE", "VK_LOADER_DEVICE_SELECT",
    "VK_LOADER_DISABLE_SELECT", "VK_LOADER_LAYERS_ENABLE", "VK_LOADER_LAYERS_DISABLE",
    "VK_LOADER_LAYERS_ALLOW", "VK_INSTANCE_LAYERS", "VK_LAYER_PATH", "VK_ADD_LAYER_PATH",
    "MESA_VK_DEVICE_SELECT", "MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE", "DRI_PRIME",
    "__NV_PRIME_RENDER_OFFLOAD", "__VK_LAYER_NV_optimus",
)

# How the pipeline measures. `diagnostic` is the profiler: per-dispatch timestamps
# force one command buffer, so it explains partitions and transfers but is never a
# deployment timing. `timing` keeps the chunked submission of production and adds
# only the lightweight GPU span probe (two timestamps per command buffer).
# `confirmation` runs with nothing enabled, so its wall clock is the one a user sees.
# A mode names exactly which of the covered variables are on: ambient settings of
# them never reach a child.
MEASUREMENT_VERSION = 1
MODES = {
    "diagnostic": {"stats": True, "probes": []},
    "timing": {"stats": False, "probes": ["VULKAN_GPU_SPAN"]},
    "confirmation": {"stats": False, "probes": []},
}

# A tuned artifact is reusable only where these sections match. Measurement
# changes timing, not results, and tuning is the artifact itself; both are
# still reported when they differ.
COMPATIBILITY = ("numerics", "allocator", "ep", "device_selection")

# Rust `u64::from_str`: an optional `+` and ASCII digits, nothing else.
_U64 = re.compile(r"\+?[0-9]+")


def _pool_budget_mb(value):
    """`VULKAN_POOL_BUDGET_MB` as the runtime reads it: a megabyte count, or `auto`
    (a fraction of the largest device-local heap, so it depends on the device)
    when unset or when the runtime warns and ignores the value."""
    if value is not None and _U64.fullmatch(value) and int(value) < 1 << 64:
        return int(value)
    return "auto"


def resolve(*, tactics=None, ep_compile=None, stats=None, mode=None, environ=None):
    """The policy a child process runs under.

    `tactics` is the sealed tuned artifact (relative to its bundle) the run
    consumes; none means tuning is off. `ep_compile` and `stats`, when given,
    are explicit overrides of the path (True forces the setting); None passes
    the caller's environment through. The pipeline always sets the tuning
    variables, so an ambient tuning mode or artifact never reaches a child.

    `mode` (a key of `MODES`) replaces `stats` and the probes together: the child
    runs with exactly that measurement, whatever the caller's environment holds.
    """
    if mode is not None:
        stats = MODES[mode]["stats"]
    environ = os.environ if environ is None else environ
    ops = environ.get("VULKAN_EP_OPS")
    explicit = environ.get("VULKAN_EP_COMPILE")
    if ep_compile is None:
        ep_compile = explicit is not None and explicit not in ("0", "")
    return {
        "version": VERSION,
        "numerics": {"attention_fp16": environ.get("ONNX_VULKAN_ATTENTION_FP16") == "1"},
        "allocator": {"pool_budget_mb": _pool_budget_mb(environ.get("VULKAN_POOL_BUDGET_MB"))},
        "tuning": {"mode": "cache-only" if tactics else "off", "artifact": tactics or None},
        # The compiling path is also on whenever tuning is, as in the runtime.
        "ep": {"compile": bool(ep_compile or tactics),
               "op_filter": None if ops is None else sorted({op.strip() for op in ops.split(",")})},
        "measurement": {"stats": "VULKAN_EP_STATS" in environ if stats is None else stats,
                        "probes": (list(MODES[mode]["probes"]) if mode is not None
                                   else [name for name in PROBES if name in environ])},
        "device_selection": {name: environ[name] for name in DEVICE_SELECTION if name in environ},
    }


def measurement_plan():
    """The measurement modes the pipeline times with, recorded in resume
    identities: evidence taken under another plan is not mixed with this one."""
    return {"version": MEASUREMENT_VERSION, "modes": {name: dict(mode) for name, mode in MODES.items()}}


def pipeline_identity(environ=None):
    """The policy of the pipeline's validation runs, without the per-run tuning
    artifact (tuned artifacts are identified by their sealed files). This is the
    record resume identities and optimized manifests carry."""
    policy = resolve(ep_compile=True, stats=True, environ=environ)
    del policy["tuning"]
    return policy


def environment(policy, *, bundle=None):
    """The variables that realize `policy`, to pass to `process.command` (a None
    value removes an inherited variable). Every variable the record covers is
    set or removed; unrelated variables are still inherited by the command."""
    tuning, ep = policy["tuning"], policy["ep"]
    budget = policy["allocator"]["pool_budget_mb"]
    env = {
        "ONNX_VULKAN_ATTENTION_FP16": "1" if policy["numerics"]["attention_fp16"] else None,
        "VULKAN_POOL_BUDGET_MB": None if budget == "auto" else str(budget),
        "ONNX_VULKAN_TUNING_MODE": tuning["mode"],
        "ONNX_VULKAN_TUNING_ARTIFACT": str(bundle / tuning["artifact"]) if tuning["artifact"] else "",
        "VULKAN_EP_COMPILE": "1" if ep["compile"] else None,
        "VULKAN_EP_OPS": None if ep["op_filter"] is None else ",".join(ep["op_filter"]),
        "VULKAN_EP_STATS": "1" if policy["measurement"]["stats"] else None,
    }
    env.update({name: "1" if name in policy["measurement"]["probes"] else None for name in PROBES})
    env.update({name: policy["device_selection"].get(name) for name in DEVICE_SELECTION})
    return env


def differences(recorded, current, sections=None):
    """Human-readable `section.key: recorded → current` changes between two
    records; `sections` limits the comparison."""
    changes = []
    for section in sections or sorted({*recorded, *current} - {"version"}):
        old, new = recorded.get(section), current.get(section)
        if old == new:
            continue
        if isinstance(old, dict) and isinstance(new, dict):
            changes += [f"{section}.{key}: {old.get(key)!r} → {new.get(key)!r}"
                        for key in sorted({*old, *new}) if old.get(key) != new.get(key)]
        else:
            changes.append(f"{section}: {old!r} → {new!r}")
    if recorded.get("version") != current.get("version"):
        changes.insert(0, f"version: {recorded.get('version')!r} → {current.get('version')!r}")
    return changes
