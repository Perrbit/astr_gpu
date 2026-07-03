#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/2dvort_mpirank_matrix}"
MATRIX="${MATRIX:-2:2,1,1 2:1,2,1 2:1,1,2}"
MAXSTEP_STATS="${MAXSTEP_STATS:-1}"
MAXSTEP_FIELD="${MAXSTEP_FIELD:-1}"
FEQCHKPT_STATS="${FEQCHKPT_STATS:-99}"
FEQCHKPT_FIELD="${FEQCHKPT_FIELD:-1}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-128,64,16}"
DELTAT="${DELTAT:-5.d-4}"
NOFILTER_STATS_ATOL="${NOFILTER_STATS_ATOL:-1e-10}"
NOFILTER_FIELD_ATOL="${NOFILTER_FIELD_ATOL:-1e-10}"
FILTER_STATS_ATOL="${FILTER_STATS_ATOL:-1e-9}"
FILTER_FIELD_ATOL="${FILTER_FIELD_ATOL:-5e-9}"
RTOL="${RTOL:-1e-10}"
RUN_NOFILTER="${RUN_NOFILTER:-t}"
RUN_FILTER="${RUN_FILTER:-t}"
RUN_FIELD="${RUN_FIELD:-t}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

run_case() {
  local entry="$1"
  local filter_flag="$2"
  local maxstep="$3"
  local feqchkpt="$4"
  local compare_stats="$5"
  local compare_field="$6"
  local stats_atol="$7"
  local field_atol="$8"
  local prefix="$9"
  local np="${entry%%:*}"
  local topology="${entry#*:}"
  local tag
  tag="$(topology_tag "$topology")"
  local case_out="$OUT_DIR/${prefix}_np${np}_${tag}"

  NP="$np" TOPOLOGY="$topology" MAXSTEP="$maxstep" FEQCHKPT="$feqchkpt" \
    LFILTER="$filter_flag" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" GRID="$GRID" DELTAT="$DELTAT" \
    COMPARE_STATS="$compare_stats" COMPARE_FIELD="$compare_field" \
    STATS_ATOL="$stats_atol" STATS_RTOL="$RTOL" FIELD_ATOL="$field_atol" FIELD_RTOL="$RTOL" \
    OUT_DIR="$case_out" "$ROOT_DIR/tests/gpu_validation/run_2dvort_phasea_compare.sh"
  printf 'pass %s np=%s topology=%s lfilter=%s out=%s\n' \
    "$prefix" "$np" "$topology" "$filter_flag" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

for entry in $MATRIX; do
  if [[ "$RUN_NOFILTER" == "t" ]]; then
    run_case "$entry" f "$MAXSTEP_FIELD" "$FEQCHKPT_FIELD" t "$RUN_FIELD" \
      "$NOFILTER_STATS_ATOL" "$NOFILTER_FIELD_ATOL" nofilter
  fi
  if [[ "$RUN_FILTER" == "t" ]]; then
    run_case "$entry" t "$MAXSTEP_FIELD" "$FEQCHKPT_FIELD" t "$RUN_FIELD" \
      "$FILTER_STATS_ATOL" "$FILTER_FIELD_ATOL" filter
  fi
done
