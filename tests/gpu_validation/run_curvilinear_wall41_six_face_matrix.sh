#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_wall41_six_face_matrix}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

run_case() {
  local axis="$1"
  local label="$2"
  local np="$3"
  local topology="$4"
  local maxstep="$5"
  local lfilter="$6"
  local diffterm="$7"
  local case_out="$OUT_DIR/${axis}_${label}"

  WALL_AXIS="$axis" OUT_DIR="$case_out" \
  NP="$np" TOPOLOGY="$topology" MAXSTEP="$maxstep" FEQCHKPT="$maxstep" \
  LFILTER="$lfilter" DIFFTERM="$diffterm" \
  FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  STATS_ATOL="${STATS_ATOL:-1e-9}" STATS_RTOL="${STATS_RTOL:-1e-10}" \
  WALL_ATOL="${WALL_ATOL:-1e-12}" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall41_compare.sh"

  printf 'pass axis=%s label=%s np=%s topology=%s maxstep=%s lfilter=%s diffterm=%s out=%s\n' \
    "$axis" "$label" "$np" "$topology" "$maxstep" "$lfilter" "$diffterm" \
    "$case_out" >> "$SUMMARY"
}

for axis in x y z; do
  case "$axis" in
    x)
      slab="2,1,1"
      plane="2,2,1"
      ;;
    y)
      slab="1,2,1"
      plane="2,2,1"
      ;;
    z)
      slab="1,1,2"
      plane="2,1,2"
      ;;
  esac

  run_case "$axis" np1_conv 1 1,1,1 1 f f
  run_case "$axis" np1_filter_diff 1 1,1,1 5 t t
  run_case "$axis" np2_physical_filter_diff 2 "$slab" 5 t t
  run_case "$axis" np4_plane_filter_diff 4 "$plane" 1 t t
  run_case "$axis" np8_xyz_filter_diff 8 2,2,2 1 t t
done

cat "$SUMMARY"
