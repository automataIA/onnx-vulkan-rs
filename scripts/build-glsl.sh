#!/usr/bin/env bash
# Regenerate the checked-in SPIR-V for the GLSL shaders.
#
# Only two kinds of kernel live in GLSL: cooperative-matrix ones (naga's WGSL
# frontend has no 8-bit scalars and only square cooperative matrix shapes) and
# ones that must not let the driver contract a multiply-add into an FMA (naga
# cannot emit NoContraction; GLSL `precise` does). The SPIR-V is
# committed so that the build needs no GLSL toolchain; run this by hand after
# editing a .comp file.
#
# GLSLANG=/path/to/glslangValidator scripts/build-glsl.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shaders="$root/crates/onnx-vulkan-core/src/shaders"
glslang="${GLSLANG:-}"

if [[ -z "$glslang" ]]; then
    if command -v glslangValidator >/dev/null 2>&1; then
        glslang=glslangValidator
    fi
fi
if [[ -z "$glslang" ]]; then
    echo "glslangValidator not found; set GLSLANG=<path>" >&2
    exit 1
fi

# (K_TILE, SUBGROUP_SIZE, accumulator) triples, one per cooperative matrix
# configuration seen in the wild, plus an optional TM x TN block of 16x16 output
# tiles per workgroup (1x1 when omitted). Each configuration is built twice:
# plain (i32 out) and with the dequantizing epilogue (f32 out, suffix _epi).
# Check what a device really supports with
#   cargo run -p vk-compute --example coopmat_probe
build_coop() {
    local ktile=$1 subgroup=$2 acc=$3 tm=${4:-1} tn=${5:-1}
    local name="matmul_integer_coop_k${ktile}_sg${subgroup}_${acc}"
    [[ "$tm$tn" != "11" ]] && name+="_b${tm}x${tn}"
    local defs=("-DK_TILE=$ktile" "-DSUBGROUP_SIZE=$subgroup" "-DTM=$tm" "-DTN=$tn")
    [[ "$acc" == "i32" ]] && defs+=("-DACC_SIGNED=1")
    local suffix extra
    for suffix in "" _epi; do
        extra=()
        [[ -n "$suffix" ]] && extra=("-DEPILOGUE=1")
        local out="$shaders/spv/$name$suffix.spv"
        "$glslang" --target-env vulkan1.2 -S comp "${defs[@]}" "${extra[@]}" \
            -o "$out" "$shaders/glsl/matmul_integer_coop.comp"
        echo "  -> $(basename "$out")"
    done
}

# The shared-memory build: WM x WN subgroups per workgroup, each TM x TN tiles,
# staging BK bytes of K per chunk. Named after the workgroup block.
build_coop_smem() {
    local ktile=$1 subgroup=$2 acc=$3 wm=$4 wn=$5 tm=$6 tn=$7 bk=$8
    local name="matmul_integer_coop_smem_k${ktile}_sg${subgroup}_${acc}_$((wm * tm * 16))x$((wn * tn * 16))"
    local defs=("-DK_TILE=$ktile" "-DSUBGROUP_SIZE=$subgroup" "-DWM=$wm" "-DWN=$wn"
        "-DTM=$tm" "-DTN=$tn" "-DBK=$bk")
    [[ "$acc" == "i32" ]] && defs+=("-DACC_SIGNED=1")
    local suffix extra
    for suffix in "" _epi; do
        extra=()
        [[ -n "$suffix" ]] && extra=("-DEPILOGUE=1")
        local out="$shaders/spv/$name$suffix.spv"
        "$glslang" --target-env vulkan1.2 -S comp "${defs[@]}" "${extra[@]}" \
            -o "$out" "$shaders/glsl/matmul_integer_coop_smem.comp"
        echo "  -> $(basename "$out")"
    done
}

echo "glslang: $glslang"
build_coop 32 32 u32       # NVIDIA: 16x16x32 u8*u8 -> u32, subgroup 32
build_coop 16 64 i32       # AMD:    16x16x16 u8*u8 -> i32, subgroup 64
# NVIDIA, 128x128 per workgroup of 8 subgroups, 64 bytes of K staged per chunk
build_coop_smem 32 32 u32 4 2 2 4 64
# NVIDIA, 64x64 per workgroup of 4 subgroups: products too small to fill the
# device with 128x128 blocks (see matmul_integer::coop_variant_for)
build_coop_smem 32 32 u32 2 2 2 2 64

# FlashAttention, f16 x f16 -> f32 16x16x16 cooperative matrices, heads of 32 and 64,
# each without and with the additive bias
for head in 32 64; do
    for suffix in "" _bias; do
        extra=()
        [[ -n "$suffix" ]] && extra=("-DBIAS=1")
        out="$shaders/spv/flash_attention_coop_f16_d$head$suffix.spv"
        "$glslang" --target-env vulkan1.2 -S comp -DHEAD=$head "${extra[@]}" -o "$out" \
            "$shaders/glsl/flash_attention_coop.comp"
        echo "  -> $(basename "$out")"
    done
done

out="$shaders/spv/interleaved_rope.spv"
"$glslang" --target-env vulkan1.2 -S comp -o "$out" "$shaders/glsl/interleaved_rope.comp"
echo "  -> $(basename "$out")"

out="$shaders/spv/matmul_integer_any_k_epi.spv"
"$glslang" --target-env vulkan1.2 -S comp -o "$out" "$shaders/glsl/matmul_integer_any_k.comp"
echo "  -> $(basename "$out")"

out="$shaders/spv/matmul_integer_small_epi.spv"
"$glslang" --target-env vulkan1.2 -S comp -o "$out" "$shaders/glsl/matmul_integer_small.comp"
echo "  -> $(basename "$out")"

out="$shaders/spv/mul_add_relu.spv"
"$glslang" --target-env vulkan1.2 -S comp -o "$out" "$shaders/glsl/mul_add_relu.comp"
echo "  -> $(basename "$out")"
