#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_sbli_physical_nscbc_matrix}"
MAXSTEP="${MAXSTEP:-2}"

run_case() {
  local name="$1"
  local np="$2"
  local topology="$3"
  local km="$4"
  OUT_DIR="$OUT_DIR/$name" NP="$np" TOPOLOGY="$topology" KM="$km" \
    MAXSTEP="$MAXSTEP" FEQCHKPT="$MAXSTEP" \
    "$ROOT_DIR/tests/gpu_validation/run_s2_sbli_physical_nscbc_compare.sh"
}

run_case np1 1 1,1,1 8
run_case np2_x 2 2,1,1 8
run_case np2_y 2 1,2,1 8
run_case np2_z 2 1,1,2 16

printf 'S2 physical SBLI NSCBC MPI matrix passed: %s\n' "$OUT_DIR"
