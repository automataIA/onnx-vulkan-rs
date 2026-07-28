#!/usr/bin/env bash
# Regenerate the checked-in SPIR-V for the GLSL shaders.
#
# Only cooperative-matrix kernels live in GLSL (naga's WGSL frontend has no
# 8-bit scalars and only square cooperative matrix shapes). The SPIR-V is
# committed so that neither the build nor cross-compilation needs a GLSL
# toolchain; run this by hand after editing a .comp file.
#
# GLSLANG=/path/to/glslangValidator scripts/build-glsl.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shaders="$root/crates/onnx-vulkan-core/src/shaders"
glslang="${GLSLANG:-}"

if [[ -z "$glslang" ]]; then
    for candidate in glslangValidator /mnt/c/VulkanSDK/*/Bin/glslangValidator.exe; do
        if command -v "$candidate" >/dev/null 2>&1; then
            glslang="$candidate"
            break
        fi
    done
fi
if [[ -z "$glslang" ]]; then
    echo "glslangValidator not found; set GLSLANG=<path>" >&2
    exit 1
fi

# (K_TILE, SUBGROUP_SIZE, accumulator) triples, one per cooperative matrix
# configuration seen in the wild. Check what a device really supports with
#   cargo run -p vk-compute --example coopmat_probe
build_coop() {
    local ktile=$1 subgroup=$2 acc=$3
    local out="$shaders/spv/matmul_integer_coop_k${ktile}_sg${subgroup}_${acc}.spv"
    local defs=("-DK_TILE=$ktile" "-DSUBGROUP_SIZE=$subgroup")
    [[ "$acc" == "i32" ]] && defs+=("-DACC_SIGNED=1")
    "$glslang" --target-env vulkan1.2 -S comp "${defs[@]}" \
        -o "$out" "$shaders/glsl/matmul_integer_coop.comp"
    echo "  -> $(basename "$out")"
}

echo "glslang: $glslang"
build_coop 32 32 u32   # NVIDIA: 16x16x32 u8*u8 -> u32, subgroup 32
build_coop 16 64 i32   # AMD:    16x16x16 u8*u8 -> i32, subgroup 64
