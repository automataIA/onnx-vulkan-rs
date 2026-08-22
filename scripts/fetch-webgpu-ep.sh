#!/usr/bin/env bash
# Downloads Microsoft's standalone WebGPU plugin EP (linux-x64) for the
# `webgpu` mode of the test suite.
#
# The plugin is distributed as a Python wheel, but the payload is a plain
# `.so`: the wheel is `py3-none`, so nothing here needs a Python runtime.
# The wheel URL is resolved from the PyPI JSON API rather than hardcoded,
# because the CDN path carries a content hash.
#
# On Linux this EP runs on Dawn -> Vulkan, i.e. the same device our own EP
# uses. That is the point of the comparison.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEBGPU_EP_VERSION="0.2.1"
WEBGPU_DIR="$ROOT/third_party/webgpu-ep"
SO_NAME="libonnxruntime_providers_webgpu.so"
PYPI_JSON="https://pypi.org/pypi/onnxruntime-ep-webgpu/$WEBGPU_EP_VERSION/json"

command -v jq >/dev/null || {
    echo "jq is required (the test suite needs it too): apt install jq" >&2
    exit 1
}
command -v unzip >/dev/null || {
    echo "unzip is required: apt install unzip" >&2
    exit 1
}

if [ -f "$WEBGPU_DIR/$SO_NAME" ]; then
    echo "ok: $WEBGPU_DIR/$SO_NAME (already present)"
    exit 0
fi

mkdir -p "$WEBGPU_DIR"

echo "resolving: onnxruntime-ep-webgpu $WEBGPU_EP_VERSION (manylinux x86_64)"
wheel_url="$(curl -fL --retry 3 "$PYPI_JSON" |
    jq -r '.urls[] | select(.filename | test("manylinux.*x86_64\\.whl$")) | .url' |
    head -1)"

[ -n "$wheel_url" ] || {
    echo "no manylinux x86_64 wheel for version $WEBGPU_EP_VERSION" >&2
    exit 1
}

wheel="$WEBGPU_DIR/$(basename "$wheel_url")"
echo "downloading: $wheel_url"
curl -fL --retry 3 -o "$wheel.part" "$wheel_url"
mv "$wheel.part" "$wheel"

# the wheel lays the payload out under onnxruntime_ep_webgpu/; flatten it,
# the suite wants one predictable path
unzip -o -j "$wheel" "*/$SO_NAME" -d "$WEBGPU_DIR"

[ -f "$WEBGPU_DIR/$SO_NAME" ] || {
    echo "wheel did not contain $SO_NAME" >&2
    exit 1
}

echo "done."
ls -lh "$WEBGPU_DIR"
echo
echo "shared-library dependencies (Dawn is expected to be linked in):"
ldd "$WEBGPU_DIR/$SO_NAME" | grep -i "not found" && {
    echo "WARNING: unresolved dependencies above" >&2
} || echo "  all resolved"
