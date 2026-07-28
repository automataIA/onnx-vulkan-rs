#!/usr/bin/env bash
# Downloads ONNX Runtime 1.27.1 (linux-x64 + win-x64) and the Parakeet TDT 0.6B v3 int8 model.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ORT_VERSION="1.27.1"
ORT_DIR="$ROOT/third_party/onnxruntime"
MODELS_DIR="$ROOT/models/parakeet-tdt-0.6b-v3-onnx"
HF_BASE="https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"

mkdir -p "$ORT_DIR" "$MODELS_DIR"

fetch() { # url, dest
    if [ -f "$2" ]; then
        echo "ok: $2 (already present)"
    else
        echo "downloading: $1"
        curl -fL --retry 3 -o "$2.part" "$1"
        mv "$2.part" "$2"
    fi
}

# --- ONNX Runtime release binaries + header ---
ORT_GH="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION"

fetch "$ORT_GH/onnxruntime-linux-x64-$ORT_VERSION.tgz" "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION.tgz"
if [ ! -d "$ORT_DIR/linux-x64" ]; then
    tar -xzf "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION.tgz" -C "$ORT_DIR"
    mv "$ORT_DIR/onnxruntime-linux-x64-$ORT_VERSION" "$ORT_DIR/linux-x64"
fi

fetch "$ORT_GH/onnxruntime-win-x64-$ORT_VERSION.zip" "$ORT_DIR/onnxruntime-win-x64-$ORT_VERSION.zip"
if [ ! -d "$ORT_DIR/win-x64" ]; then
    unzip -q "$ORT_DIR/onnxruntime-win-x64-$ORT_VERSION.zip" -d "$ORT_DIR"
    mv "$ORT_DIR/onnxruntime-win-x64-$ORT_VERSION" "$ORT_DIR/win-x64"
fi

# --- Parakeet TDT 0.6B v3 model (int8 ONNX) ---
# SKIP_MODEL=1 to download only ORT (e.g. in CI)
if [ "${SKIP_MODEL:-0}" != "1" ]; then
    for f in config.json vocab.txt nemo128.onnx decoder_joint-model.int8.onnx encoder-model.int8.onnx; do
        fetch "$HF_BASE/$f" "$MODELS_DIR/$f"
    done
fi

echo "done."
ls -lh "$ORT_DIR" "$MODELS_DIR"
