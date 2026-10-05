#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  printf 'Usage: %s CUDA_ROOT NEW_COMPAT_DIRECTORY\n' "$0" >&2
  exit 2
fi
cuda=$(realpath "$1")
destination=$(realpath -m "$2")
patch_file=$(realpath "$(dirname "$0")/patches/cuda-13.1-glibc-rsqrt.patch")
if [[ -e "$destination" || "$destination" == "$cuda"/* ]]; then
  printf 'Compatibility directory must be new and outside CUDA_ROOT\n' >&2
  exit 2
fi
mkdir -p "$destination/include"
cp -a "$cuda/targets/x86_64-linux/include/crt" "$destination/include/"
cp -a "$cuda/targets/x86_64-linux/include/cuda_runtime.h" "$destination/include/"
patch --fuzz=0 -p1 -d "$destination/include/crt" -i "$patch_file"
printf 'Use PV_DEVICE_CUDA_FLAGS=-I%s/include\n' "$destination"
