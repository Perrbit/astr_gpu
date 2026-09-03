#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_wall41_matrix}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

run_case() {
  local label="$1"
  local np="$2"
  local topology="$3"
  local maxstep="$4"
  local lfilter="$5"
  local diffterm="$6"

  OUT_DIR="$OUT_DIR/$label" \
  NP="$np" TOPOLOGY="$topology" MAXSTEP="$maxstep" FEQCHKPT="$maxstep" \
  LFILTER="$lfilter" DIFFTERM="$diffterm" \
  FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  STATS_ATOL="${STATS_ATOL:-1e-9}" STATS_RTOL="${STATS_RTOL:-1e-10}" \
  WALL_ATOL="${WALL_ATOL:-1e-12}" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall41_compare.sh"

  printf 'pass label=%s np=%s topology=%s maxstep=%s lfilter=%s diffterm=%s out=%s\n' \
    "$label" "$np" "$topology" "$maxstep" "$lfilter" "$diffterm" \
    "$OUT_DIR/$label" >> "$SUMMARY"
}

run_case np1_conv 1 1,1,1 1 f f
run_case np1_filter_diff 1 1,1,1 5 t t
run_case np2_x_filter_diff 2 2,1,1 5 t t
run_case np4_xy_filter_diff 4 2,2,1 1 t t

cat "$SUMMARY"
