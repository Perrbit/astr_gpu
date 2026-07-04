#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/rti_phasej_matrix}"
RUN_SINGLE="${RUN_SINGLE:-t}"
RUN_NP2="${RUN_NP2:-t}"
RUN_NP4="${RUN_NP4:-t}"
RUN_NP8="${RUN_NP8:-t}"
RUN_LONG="${RUN_LONG:-t}"
DRY_RUN="${DRY_RUN:-f}"
GRID="${GRID:-32,64,32}"
DELTAT="${DELTAT:-1.95d-4}"
SCHEME="${SCHEME:-643e}"
MATRIX_SINGLE="${MATRIX_SINGLE:-f:f f:t t:f t:t}"
MATRIX_NP2="${MATRIX_NP2:-2:1,2,1 2:2,1,1 2:1,1,2}"
MATRIX_NP4="${MATRIX_NP4:-4:2,2,1 4:2,1,2 4:1,2,2}"
MATRIX_NP8="${MATRIX_NP8:-8:2,2,2}"
MATRIX_LONG="${MATRIX_LONG:-1:1,1,1 4:2,2,1}"
MAXSTEP_SHORT="${MAXSTEP_SHORT:-5}"
MAXSTEP_LONG="${MAXSTEP_LONG:-20}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

run_case() {
  local label="$1"
  local np="$2"
  local topology="$3"
  local maxstep="$4"
  local lfilter="$5"
  local diffterm="$6"
  local case_out="$OUT_DIR/$label"

  if [[ "$DRY_RUN" == "t" ]]; then
    printf 'dry-run label=%s np=%s topology=%s maxstep=%s lfilter=%s diffterm=%s out=%s\n' \
      "$label" "$np" "$topology" "$maxstep" "$lfilter" "$diffterm" "$case_out" >> "$SUMMARY"
    return
  fi

  OUT_DIR="$case_out" MAXSTEP="$maxstep" FEQCHKPT="$maxstep" \
    LFILTER="$lfilter" DIFFTERM="$diffterm" SCHEME="$SCHEME" GRID="$GRID" DELTAT="$DELTAT" \
    NP="$np" TOPOLOGY="$topology" COMPARE_STATS=t COMPARE_FIELD=t \
    STATS_ATOL="$STATS_ATOL" STATS_RTOL="$STATS_RTOL" FIELD_ATOL="$FIELD_ATOL" FIELD_RTOL="$FIELD_RTOL" \
    "$ROOT_DIR/tests/gpu_validation/run_rti_phasej_compare.sh"

  printf 'pass label=%s np=%s topology=%s maxstep=%s lfilter=%s diffterm=%s out=%s\n' \
    "$label" "$np" "$topology" "$maxstep" "$lfilter" "$diffterm" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

if [[ "$RUN_SINGLE" == "t" ]]; then
  for entry in $MATRIX_SINGLE; do
    IFS=':' read -r lfilter diffterm <<< "$entry"
    if [[ -z "${lfilter:-}" || -z "${diffterm:-}" ]]; then
      printf 'invalid RTI single matrix entry=%s; expected lfilter:diffterm\n' "$entry" >&2
      exit 2
    fi
    run_case "sr_lfilter_${lfilter}_diffterm_${diffterm}" 1 1,1,1 "$MAXSTEP_SHORT" "$lfilter" "$diffterm"
  done
fi

if [[ "$RUN_NP2" == "t" ]]; then
  for entry in $MATRIX_NP2; do
    IFS=':' read -r np topology <<< "$entry"
    run_case "np${np}_$(topology_tag "$topology")" "$np" "$topology" "$MAXSTEP_SHORT" t t
  done
fi

if [[ "$RUN_NP4" == "t" ]]; then
  for entry in $MATRIX_NP4; do
    IFS=':' read -r np topology <<< "$entry"
    run_case "np${np}_$(topology_tag "$topology")" "$np" "$topology" "$MAXSTEP_SHORT" t t
  done
fi

if [[ "$RUN_NP8" == "t" ]]; then
  for entry in $MATRIX_NP8; do
    IFS=':' read -r np topology <<< "$entry"
    run_case "np${np}_$(topology_tag "$topology")" "$np" "$topology" "$MAXSTEP_SHORT" t t
  done
fi

if [[ "$RUN_LONG" == "t" ]]; then
  for entry in $MATRIX_LONG; do
    IFS=':' read -r np topology <<< "$entry"
    run_case "long_np${np}_$(topology_tag "$topology")" "$np" "$topology" "$MAXSTEP_LONG" t t
  done
fi
