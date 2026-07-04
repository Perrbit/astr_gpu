#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/zeroextrap_phaseb_mpirank_matrix}"
MATRIX="${MATRIX:-x:2:1,2,1 x:2:1,1,2 x:2:2,1,1 y:2:2,1,1 y:2:1,1,2 y:2:1,2,1 z:2:2,1,1 z:2:1,2,1 z:2:1,1,2 x:4:1,2,2 x:4:2,2,1 x:4:2,1,2 y:4:2,1,2 y:4:2,2,1 y:4:1,2,2 z:4:2,2,1 z:4:2,1,2 z:4:1,2,2}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-1}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-64,64,64}"
DELTAT="${DELTAT:-5.d-4}"
STATS_ATOL="${STATS_ATOL:-1e-9}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
RUN_FIELD="${RUN_FIELD:-t}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

run_case() {
  local entry="$1"
  local axis np topology tag case_out

  IFS=':' read -r axis np topology <<< "$entry"
  tag="$(topology_tag "$topology")"
  case_out="$OUT_DIR/${axis}zero_np${np}_${tag}"

  ZERO_AXIS="$axis" NP="$np" TOPOLOGY="$topology" MAXSTEP="$MAXSTEP" FEQCHKPT="$FEQCHKPT" \
    LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" GRID="$GRID" DELTAT="$DELTAT" \
    COMPARE_STATS=t COMPARE_FIELD="$RUN_FIELD" \
    STATS_ATOL="$STATS_ATOL" STATS_RTOL="$STATS_RTOL" FIELD_ATOL="$FIELD_ATOL" FIELD_RTOL="$FIELD_RTOL" \
    OUT_DIR="$case_out" "$ROOT_DIR/tests/gpu_validation/run_xextrap_phaseb_compare.sh"

  printf 'pass axis=%s np=%s topology=%s lfilter=%s diffterm=%s field=%s out=%s\n' \
    "$axis" "$np" "$topology" "$LFILTER" "$DIFFTERM" "$RUN_FIELD" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

for entry in $MATRIX; do
  run_case "$entry"
done
