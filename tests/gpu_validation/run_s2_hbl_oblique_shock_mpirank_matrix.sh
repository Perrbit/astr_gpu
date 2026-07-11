#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_oblique_shock_mpirank_matrix}"
RUN_NP1="${RUN_NP1:-t}"
RUN_NP2="${RUN_NP2:-t}"
RUN_NP4="${RUN_NP4:-t}"
RUN_NP8="${RUN_NP8:-t}"
RUN_LONG="${RUN_LONG:-t}"
RUN_STRESS="${RUN_STRESS:-f}"
MATRIX_NP1="${MATRIX_NP1:-1:1,1,1}"
MATRIX_NP2="${MATRIX_NP2:-2:2,1,1 2:1,2,1 2:1,1,2}"
MATRIX_NP4="${MATRIX_NP4:-4:2,2,1 4:2,1,2 4:1,2,2}"
MATRIX_NP8="${MATRIX_NP8:-8:2,2,2}"
MATRIX_LONG="${MATRIX_LONG:-1:1,1,1 8:2,2,2}"
MATRIX_STRESS="${MATRIX_STRESS:-1:1,1,1 8:2,2,2}"
MAXSTEP="${MAXSTEP:-2}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
MAXSTEP_LONG="${MAXSTEP_LONG:-20}"
MAXSTEP_STRESS="${MAXSTEP_STRESS:-100}"
IM="${IM:-192}"
JM="${JM:-192}"
KM_NO_Z_SPLIT="${KM_NO_Z_SPLIT:-8}"
KM_Z_SPLIT="${KM_Z_SPLIT:-16}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

km_for_topology() {
  local topology="$1"
  local ix jy kz
  IFS=',' read -r ix jy kz <<< "$topology"
  if [[ "${kz:-1}" -gt 1 ]]; then
    printf '%s' "$KM_Z_SPLIT"
  else
    printf '%s' "$KM_NO_Z_SPLIT"
  fi
}

run_case() {
  local entry="$1"
  local np="${entry%%:*}"
  local topology="${entry#*:}"
  local tag km case_out
  tag="$(topology_tag "$topology")"
  km="$(km_for_topology "$topology")"
  case_out="$OUT_DIR/np${np}_${tag}"

  NP="$np" TOPOLOGY="$topology" IM="$IM" JM="$JM" KM="$km" \
    MAXSTEP="$MAXSTEP" FEQCHKPT="$FEQCHKPT" OUT_DIR="$case_out" \
    "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh"

  printf 'pass np=%s topology=%s im=%s jm=%s km=%s maxstep=%s out=%s\n' \
    "$np" "$topology" "$IM" "$JM" "$km" "$MAXSTEP" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

if [[ "$RUN_NP1" == "t" ]]; then
  for entry in $MATRIX_NP1; do
    run_case "$entry"
  done
fi

if [[ "$RUN_NP2" == "t" ]]; then
  for entry in $MATRIX_NP2; do
    run_case "$entry"
  done
fi

if [[ "$RUN_NP4" == "t" ]]; then
  for entry in $MATRIX_NP4; do
    run_case "$entry"
  done
fi

if [[ "$RUN_NP8" == "t" ]]; then
  for entry in $MATRIX_NP8; do
    run_case "$entry"
  done
fi

if [[ "$RUN_LONG" == "t" ]]; then
  old_maxstep="$MAXSTEP"
  old_feqchkpt="$FEQCHKPT"
  MAXSTEP="$MAXSTEP_LONG"
  FEQCHKPT="$MAXSTEP_LONG"
  for entry in $MATRIX_LONG; do
    run_case "$entry"
  done
  MAXSTEP="$old_maxstep"
  FEQCHKPT="$old_feqchkpt"
fi

if [[ "$RUN_STRESS" == "t" ]]; then
  old_maxstep="$MAXSTEP"
  old_feqchkpt="$FEQCHKPT"
  MAXSTEP="$MAXSTEP_STRESS"
  FEQCHKPT="$MAXSTEP_STRESS"
  for entry in $MATRIX_STRESS; do
    run_case "$entry"
  done
  MAXSTEP="$old_maxstep"
  FEQCHKPT="$old_feqchkpt"
fi
