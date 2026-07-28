#!/bin/sh
# Downloads test models (besides Parakeet) into models/zoo/.
#
# Weights of large models live in external `.onnx_data` files: the graph alone is
# enough for op-coverage analysis, weights are only needed to run them. Hence the
# three modes:
#
#   ./scripts/fetch-models.sh graphs   # only the .onnx (~20 MB), coverage analysis
#   ./scripts/fetch-models.sh small    # + small vision model weights (~250 MB)
#   ./scripts/fetch-models.sh all      # + SAM 3 / Gemma / Qwen2.5-VL (~4.7 GB)
set -eu

MODE="${1:-graphs}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZOO="$ROOT/models/zoo"
HF="https://huggingface.co"

fetch() { # url dest
    if [ -f "$2" ]; then
        echo "  = $(basename "$2") already present"
        return
    fi
    mkdir -p "$(dirname "$2")"
    echo "  ↓ $(basename "$2")"
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
    mv "$2.part" "$2"
}

hf() { # repo path dest_dir
    fetch "$HF/$1/resolve/main/$2" "$ZOO/$3/$(basename "$2")"
}

echo "== graphs (op coverage analysis)"
# MobileNetV2: fp32, int8 QOperator and QDQ — the three formats compared
hf onnxmodelzoo/mobilenetv2-12 mobilenetv2-12.onnx mobilenetv2
# YOLOv8n fp32 (Ultralytics does not publish ONNX: community mirror)
hf cabelo/yolov8 yolov8n.onnx yolov8
# RF-DETR: single graph, weights included (no .onnx_data)
hf onnx-community/rfdetr_base-ONNX onnx/model_int8.onnx rfdetr
# SAM 3 tracker, Gemma 3 1B int4, Qwen2.5-VL 3B int4: graphs only
hf onnx-community/sam3-tracker-ONNX onnx/vision_encoder_int8.onnx sam3
hf onnx-community/sam3-tracker-ONNX onnx/prompt_encoder_mask_decoder_int8.onnx sam3
hf onnx-community/gemma-3-1b-it-ONNX onnx/model_q4.onnx gemma3-1b
hf onnx-community/Qwen2.5-VL-3B-Instruct-ONNX onnx/vision_encoder_q4.onnx qwen25vl
hf onnx-community/Qwen2.5-VL-3B-Instruct-ONNX onnx/decoder_model_merged_q4.onnx qwen25vl

[ "$MODE" = "graphs" ] && { echo "== done (graphs only)"; du -sh "$ZOO"; exit 0; }

echo "== small vision model weights"
hf onnx-community/rfdetr_base-ONNX onnx/model.onnx rfdetr

[ "$MODE" = "small" ] && { echo "== done (graphs + small vision)"; du -sh "$ZOO"; exit 0; }

echo "== external weights of large models (~4.5 GB)"
hf onnx-community/sam3-tracker-ONNX onnx/vision_encoder_int8.onnx_data sam3
hf onnx-community/sam3-tracker-ONNX onnx/prompt_encoder_mask_decoder_int8.onnx_data sam3
hf onnx-community/gemma-3-1b-it-ONNX onnx/model_q4.onnx_data gemma3-1b
hf onnx-community/Qwen2.5-VL-3B-Instruct-ONNX onnx/vision_encoder_q4.onnx_data qwen25vl
hf onnx-community/Qwen2.5-VL-3B-Instruct-ONNX onnx/decoder_model_merged_q4.onnx_data qwen25vl

echo "== done"
du -sh "$ZOO"/*
