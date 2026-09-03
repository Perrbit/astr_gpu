#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_c8_filter_matrix}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

run_case() {
  local label="$1"
  local np="$2"
  local topology="$3"
  local maxstep="$4"
  local km="$5"
  local diffterm="$6"

  OUT_DIR="$OUT_DIR/$label" NP="$np" TOPOLOGY="$topology" MAXSTEP="$maxstep" \
  IM="${IM:-64}" JM="${JM:-64}" KM="$km" DIFFTERM="$diffterm" \
  GRID_WARP_X="${GRID_WARP_X:-0.4}" GRID_WARP_Y="${GRID_WARP_Y:-0.2}" \
  FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  STATS_ATOL="${STATS_ATOL:-1e-10}" STATS_RTOL="${STATS_RTOL:-1e-10}" \
  WALL_ATOL="${WALL_ATOL:-1e-12}" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c8_filter_compare.sh"

  printf 'pass label=%s np=%s topology=%s maxstep=%s grid=%sx%sx%s diffterm=%s out=%s\n' \
    "$label" "$np" "$topology" "$maxstep" "${IM:-64}" "${JM:-64}" "$km" \
    "$diffterm" "$OUT_DIR/$label" >> "$SUMMARY"
}

run_case np1_filter_isolation 1 1,1,1 1 8 f
run_case np1_long 1 1,1,1 20 8 t
run_case np2_x 2 2,1,1 5 8 t
run_case np2_y 2 1,2,1 5 8 t
run_case np2_z 2 1,1,2 5 16 t
run_case np4_xy 4 2,2,1 1 8 t

cat "$SUMMARY"
