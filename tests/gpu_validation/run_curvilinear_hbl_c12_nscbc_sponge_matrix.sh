#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_c12_nscbc_sponge_matrix}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

run_case() {
  local label="$1"
  local np="$2"
  local topology="$3"
  local maxstep="$4"
  local km="$5"
  local sponge_im="$6"
  local compare_sensor="$7"

  OUT_DIR="$OUT_DIR/$label" NP="$np" TOPOLOGY="$topology" MAXSTEP="$maxstep" \
  IM="${IM:-64}" JM="${JM:-64}" KM="$km" SPONGE_IM="$sponge_im" \
  COMPARE_SENSOR="$compare_sensor" \
  GRID_WARP_X="${GRID_WARP_X:-0.4}" GRID_WARP_Y="${GRID_WARP_Y:-0.2}" \
  FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  STATS_ATOL="${STATS_ATOL:-1e-10}" STATS_RTOL="${STATS_RTOL:-1e-10}" \
  SENSOR_ATOL="${SENSOR_ATOL:-1e-10}" SENSOR_RTOL="${SENSOR_RTOL:-1e-10}" \
  WALL_ATOL="${WALL_ATOL:-1e-12}" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c12_nscbc_sponge_compare.sh"

  printf 'pass label=%s np=%s topology=%s maxstep=%s grid=%sx%sx%s sponge_im=%s sensor=%s out=%s\n' \
    "$label" "$np" "$topology" "$maxstep" "${IM:-64}" "${JM:-64}" "$km" \
    "$sponge_im" "$compare_sensor" "$OUT_DIR/$label" >> "$SUMMARY"
}

run_case np1_nscbc_isolation 1 1,1,1 1 8 0 t
run_case np1_sponge_isolation 1 1,1,1 1 8 16 t
run_case np1_sponge_long 1 1,1,1 20 8 16 f
run_case np2_x 2 2,1,1 3 8 16 f
run_case np2_y 2 1,2,1 3 8 16 f
run_case np2_z 2 1,1,2 3 16 16 f
run_case np4_xy 4 2,2,1 3 8 16 f
run_case np4_xz 4 2,1,2 3 16 16 f
run_case np4_yz 4 1,2,2 3 16 16 f
run_case np8_xyz 8 2,2,2 3 16 16 f

if [[ "${RUN_STRESS:-f}" == "t" ]]; then
  run_case np1_stress 1 1,1,1 100 8 16 f
  run_case np8_stress 8 2,2,2 100 16 16 f
fi

cat "$SUMMARY"
