#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_symmetry_six_face_matrix}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

run_case() {
  local axis="$1"
  local np="$2"
  local topology="$3"
  local label="${axis}_np${np}"

  AXIS="$axis" NP="$np" TOPOLOGY="$topology" MAXSTEP="${MAXSTEP:-1}" \
  OUT_DIR="$OUT_DIR/$label" FIELD_ATOL="${FIELD_ATOL:-1e-10}" \
  FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  ANALYTIC_NORMAL_ATOL="${ANALYTIC_NORMAL_ATOL:-1e-6}" \
  PROJECTION_ATOL="${PROJECTION_ATOL:-1e-12}" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_symmetry_compare.sh"
  printf 'pass axis=%s np=%s topology=%s out=%s\n' \
    "$axis" "$np" "$topology" "$OUT_DIR/$label" >> "$SUMMARY"
}

run_case x 1 1,1,1
run_case y 1 1,1,1
run_case z 1 1,1,1
run_case x 2 2,1,1
run_case y 2 1,2,1
run_case z 2 1,1,2
run_case x 4 2,2,1
run_case y 4 2,2,1
run_case z 4 2,1,2
run_case x 8 2,2,2
run_case y 8 2,2,2
run_case z 8 2,2,2

cat "$SUMMARY"
