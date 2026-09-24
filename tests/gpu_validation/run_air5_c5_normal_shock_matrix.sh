#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new evidence directory}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
DELTAT="${DELTAT:-1.d-8}"

if [[ -e "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

run_gate() {
  local name="$1" grid="$2" np="$3" topology="$4"
  GRID="$grid" NP="$np" TOPOLOGY="$topology" DIFFTERM=t \
    DELTAT="$DELTAT" BUILD_DIR="$BUILD_DIR" OUT_DIR="$OUT_DIR/$name" \
    "$ROOT_DIR/tests/gpu_validation/run_air5_c5_normal_shock_compare.sh"
}

run_gate np1 16,6,6 1 1,1,1
run_gate np2_x 16,6,6 2 2,1,1
run_gate np2_y 16,12,6 2 1,2,1
run_gate np2_z 16,6,12 2 1,1,2

printf 'AIR5_C5_NORMAL_SHOCK_MATRIX_PASS\n'
