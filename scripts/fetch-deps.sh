#!/usr/bin/env bash
# Downloads ONNX Runtime 1.29.0 (linux-x64) and the Parakeet TDT 0.6B v3 int8 model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_VERSION="1.29.0"
ORT_DIR="$ROOT/third_party/onnxruntime"
MODELS_DIR="$ROOT/models/parakeet-tdt-0.6b-v3-onnx"
# pinned commit: `main` can move, and the checksums below belong to this one
HF_REV="8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce"
HF_BASE="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/$HF_REV"

mkdir -p "$ORT_DIR" "$MODELS_DIR"

fetch() { # url, dest, sha256
    if [ -f "$2" ]; then
        echo "ok: $2 (already present)"
    else
        echo "downloading: $1"
        curl -fL --retry 3 -o "$2.part" "$1"
        # verified before the rename: a file that fails never reaches its final name
        echo "$3  $2.part" | sha256sum -c --quiet || {
            echo "sha256 mismatch: $1" >&2
            rm -f "$2.part"
            exit 1
        }
        mv "$2.part" "$2"
    fi
}

# --- ONNX Runtime release binaries + header ---
ORT_GH="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION"

# digest published by GitHub for the release asset
ORT_SHA256="c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba"
fetch "$ORT_GH/onnxruntime-linux-x64-$ORT_VERSION.tgz" "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION.tgz" "$ORT_SHA256"
# The guard is on the extracted tree's own VERSION_NUMBER, not on the directory
# existing: bumping ORT_VERSION alone used to download the new tarball and then
# silently keep the old headers and .so, which is exactly the drift a version
# bump exists to catch.
installed=""
[ -f "$ORT_DIR/linux-x64/VERSION_NUMBER" ] && installed="$(cat "$ORT_DIR/linux-x64/VERSION_NUMBER")"
if [ "$installed" != "$ORT_VERSION" ]; then
    [ -n "$installed" ] && echo "replacing extracted ORT $installed with $ORT_VERSION"
    rm -rf "$ORT_DIR/linux-x64" "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION"
    tar -xzf "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION.tgz" -C "$ORT_DIR"
    mv "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION" "$ORT_DIR/linux-x64"
fi

# --- Parakeet TDT 0.6B v3 model (int8 ONNX) ---
# SKIP_MODEL=1 to download only ORT (e.g. in CI)
if [ "${SKIP_MODEL:-0}" != "1" ]; then
    while read -r sha f; do
        fetch "$HF_BASE/$f" "$MODELS_DIR/$f" "$sha"
    done <<'SUMS'
666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466 config.json
d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d vocab.txt
a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f nemo128.onnx
eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70 decoder_joint-model.int8.onnx
6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09 encoder-model.int8.onnx
SUMS
fi

echo "done."
ls -lh "$ORT_DIR" "$MODELS_DIR"
