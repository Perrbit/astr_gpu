#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/channel_phased_mpirank_matrix}"
MATRIX="${MATRIX:-2:2,1,1 2:1,2,1 2:1,1,2}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-1}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-32,32,32}"
DELTAT="${DELTAT:-5.d-4}"
STATS_ATOL="${STATS_ATOL:-1e-8}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
RUN_FIELD="${RUN_FIELD:-t}"
CHANNEL_FORCE_MODE="${CHANNEL_FORCE_MODE:-feedback}"
CHANNEL_FORCE_FIXED="${CHANNEL_FORCE_FIXED:-}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

run_case() {
  local entry="$1"
  local np topology tag case_out

  IFS=':' read -r np topology <<< "$entry"
  tag="$(topology_tag "$topology")"
  case_out="$OUT_DIR/channel_np${np}_${tag}"

  NP="$np" TOPOLOGY="$topology" MAXSTEP="$MAXSTEP" FEQCHKPT="$FEQCHKPT" \
    LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" GRID="$GRID" DELTAT="$DELTAT" \
    CHANNEL_FORCE_MODE="$CHANNEL_FORCE_MODE" CHANNEL_FORCE_FIXED="$CHANNEL_FORCE_FIXED" \
    COMPARE_STATS=t COMPARE_FIELD="$RUN_FIELD" \
    STATS_ATOL="$STATS_ATOL" STATS_RTOL="$STATS_RTOL" FIELD_ATOL="$FIELD_ATOL" FIELD_RTOL="$FIELD_RTOL" \
    OUT_DIR="$case_out" "$ROOT_DIR/tests/gpu_validation/run_channel_phased_compare.sh"

  printf 'pass np=%s topology=%s lfilter=%s diffterm=%s force_mode=%s field=%s out=%s\n' \
    "$np" "$topology" "$LFILTER" "$DIFFTERM" "$CHANNEL_FORCE_MODE" "$RUN_FIELD" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

for entry in $MATRIX; do
  run_case "$entry"
done
