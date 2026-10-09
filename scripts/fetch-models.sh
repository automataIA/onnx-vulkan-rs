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
#
# Every file is fetched from a pinned commit and checked against its sha256
# (the LFS digest Hugging Face reports for that commit): `main` can move.
set -eu

MODE="${1:-graphs}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZOO="$ROOT/models/zoo"
HF="https://huggingface.co"

MOBILENET=onnxmodelzoo/mobilenetv2-12@23a2a25eaaa6fcd61865414c8a1acf13dda66b7a
YOLO=cabelo/yolov8@91e55d55d5b0cbf13fb661dea4f27e270c841428
RFDETR=onnx-community/rfdetr_base-ONNX@7b18fa82ba7b1d6bc9446518cf2cc1d991097a3f
SAM3=onnx-community/sam3-tracker-ONNX@429305c8a5b3de597243d919a07e4e6bdcd00ef7
GEMMA=onnx-community/gemma-3-1b-it-ONNX@a58439f40017d3b99c7d378ff525e54e0ba08ebf
QWEN=onnx-community/Qwen2.5-VL-3B-Instruct-ONNX@c1511d390484a0fb77ac8fbee7277e46b859f1e3

fetch() { # url dest sha256
    if [ -f "$2" ]; then
        echo "  = $(basename "$2") already present"
        return
    fi
    mkdir -p "$(dirname "$2")"
    echo "  ↓ $(basename "$2")"
    curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
    # verified before the rename: a file that fails never reaches its final name
    echo "$3  $2.part" | sha256sum -c --quiet || {
        echo "sha256 mismatch: $1" >&2
        rm -f "$2.part"
        exit 1
    }
    mv "$2.part" "$2"
}

hf() { # repo@rev path dest_dir sha256
    fetch "$HF/${1%@*}/resolve/${1#*@}/$2" "$ZOO/$3/$(basename "$2")" "$4"
}

echo "== graphs (op coverage analysis)"
# MobileNetV2: fp32, int8 QOperator and QDQ — the three formats compared
hf $MOBILENET mobilenetv2-12.onnx mobilenetv2 c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5
# YOLOv8n fp32 (Ultralytics does not publish ONNX: community mirror)
hf $YOLO yolov8n.onnx yolov8 c03e8dc50385b49f150a8de1b3fa6f3f8a0ba0ce2a458a353179c084f8012f35
# RF-DETR: single graph, weights included (no .onnx_data)
hf $RFDETR onnx/model_int8.onnx rfdetr 5651782096a43cbbab1320d9844c19e5851295d83ebc2a8a0df7ccc0efe1d6d9
# SAM 3 tracker, Gemma 3 1B int4, Qwen2.5-VL 3B int4: graphs only
hf $SAM3 onnx/vision_encoder_int8.onnx sam3 1795b2157a6a3660b874247fc7d07e7f5cd347386efa49e35fb2a9fac313e5dd
hf $SAM3 onnx/prompt_encoder_mask_decoder_int8.onnx sam3 a27cdec9cd61521b29d3ec15da3a058e658301f25f3159734b7fdcc3466dbb87
hf $GEMMA onnx/model_q4.onnx gemma3-1b 69686023e5892376e38fcbcdd0c77af432c55b3bcd03aee6d561bd1f04507da0
hf $QWEN onnx/vision_encoder_q4.onnx qwen25vl f109d59590acf3715bbf5b4980d3bf08e7b0408ad80155772a687b33ad3eec59
hf $QWEN onnx/decoder_model_merged_q4.onnx qwen25vl 26cfd9ca8263fc9bd030c9695b30fee943eb9dbc1b7de87666e71f08a90a3f92

[ "$MODE" = "graphs" ] && { echo "== done (graphs only)"; du -sh "$ZOO"; exit 0; }

echo "== small vision model weights"
hf $RFDETR onnx/model.onnx rfdetr 48b9502c04be895713efc0579c8904e69a97b3e03413a2f5aab1bdecda7e36ff

[ "$MODE" = "small" ] && { echo "== done (graphs + small vision)"; du -sh "$ZOO"; exit 0; }

echo "== external weights of large models (~4.5 GB)"
hf $SAM3 onnx/vision_encoder_int8.onnx_data sam3 061c08129985b36f17e229e429b536028c8ec0a56c625522a1c9795139f1ce54
hf $SAM3 onnx/prompt_encoder_mask_decoder_int8.onnx_data sam3 f6bc264ca57dea733f475327658158e36abbdde4cc53a494a0e1783d2e96ac9d
hf $GEMMA onnx/model_q4.onnx_data gemma3-1b c2370070be257a98d50e17d81be13e18304c39e7e6d9d1416f8f883681d2a17b
hf $QWEN onnx/vision_encoder_q4.onnx_data qwen25vl 85aa0c8d7bc5a0de9d544054f6149ae33d2469376d26dcf3597d11bbd1838b98
hf $QWEN onnx/decoder_model_merged_q4.onnx_data qwen25vl 2394fe5ee74f760ce944d994ad5d40450863b5020c50276ba8155c91a3fc9486

echo "== done"
du -sh "$ZOO"/*
