#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_BUILD="${CPU_BUILD:-$ROOT_DIR/build_cpu_probe}"
GPU_BUILD="${GPU_BUILD:-$ROOT_DIR/build_gpu_probe}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_policy}"

mkdir -p "$OUT_DIR"
cmake --build "$CPU_BUILD" --target astr -j2
cmake --build "$GPU_BUILD" --target astr nscbc_characteristic_policy_probe -j2
mpirun -np 1 "$CPU_BUILD/bin/astr" test bcgc > "$OUT_DIR/cpu.log"
"$GPU_BUILD/bin/nscbc_characteristic_policy_probe" > "$OUT_DIR/gpu.log"
python3 "$ROOT_DIR/tests/gpu_validation/compare_nscbc_characteristic_policy.py" \
  --cpu "$OUT_DIR/cpu.log" --gpu "$OUT_DIR/gpu.log" --atol 1e-12
