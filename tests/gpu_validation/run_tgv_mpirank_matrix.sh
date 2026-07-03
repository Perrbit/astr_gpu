#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_mpirank_matrix}"
STATS_MATRIX="${STATS_MATRIX:-2:2,1,1 2:1,2,1 2:1,1,2 4:2,2,1 4:2,1,2 4:1,2,2 8:2,2,2}"
FIELD_MATRIX="${FIELD_MATRIX:-2:2,1,1 2:1,2,1 2:1,1,2 4:2,2,1 4:2,1,2 4:1,2,2 8:2,2,2}"
SMOKE_MATRIX="${SMOKE_MATRIX:-16:4,2,2 16:2,4,2 16:2,2,4 32:4,4,2}"
RUN_SMOKE="${RUN_SMOKE:-f}"
MAXSTEP_STATS="${MAXSTEP_STATS:-2}"
MAXSTEP_FIELD="${MAXSTEP_FIELD:-1}"
MAXSTEP_SMOKE="${MAXSTEP_SMOKE:-1}"
FEQCHKPT_STATS="${FEQCHKPT_STATS:-99}"
FEQCHKPT_FIELD="${FEQCHKPT_FIELD:-1}"
FEQCHKPT_SMOKE="${FEQCHKPT_SMOKE:-99}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

run_stats_case() {
  local entry="$1"
  local maxstep="$2"
  local feqchkpt="$3"
  local prefix="$4"
  local np="${entry%%:*}"
  local topology="${entry#*:}"
  local tag
  tag="$(topology_tag "$topology")"
  local case_out="$OUT_DIR/${prefix}_np${np}_${tag}_stats"

  NP="$np" TOPOLOGY="$topology" MAXSTEP="$maxstep" FEQCHKPT="$feqchkpt" \
    LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" OUT_DIR="$case_out" \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh"
  printf 'pass stats %s np=%s topology=%s out=%s\n' "$prefix" "$np" "$topology" "$case_out" >> "$SUMMARY"
}

run_field_case() {
  local entry="$1"
  local np="${entry%%:*}"
  local topology="${entry#*:}"
  local tag
  tag="$(topology_tag "$topology")"
  local case_out="$OUT_DIR/field_np${np}_${tag}"

  NP="$np" TOPOLOGY="$topology" MAXSTEP="$MAXSTEP_FIELD" FEQCHKPT="$FEQCHKPT_FIELD" \
    LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" OUT_DIR="$case_out" \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_mpirank2_field_compare.sh"
  printf 'pass field np=%s topology=%s out=%s\n' "$np" "$topology" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

for entry in $STATS_MATRIX; do
  run_stats_case "$entry" "$MAXSTEP_STATS" "$FEQCHKPT_STATS" core
done

for entry in $FIELD_MATRIX; do
  run_field_case "$entry"
done

if [[ "$RUN_SMOKE" == "t" ]]; then
  for entry in $SMOKE_MATRIX; do
    run_stats_case "$entry" "$MAXSTEP_SMOKE" "$FEQCHKPT_SMOKE" smoke
  done
fi
