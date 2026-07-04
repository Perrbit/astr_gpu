#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/source_phasek_matrix}"
RUN_NOSOURCE="${RUN_NOSOURCE:-t}"
RUN_CHANNEL="${RUN_CHANNEL:-t}"
RUN_RTI="${RUN_RTI:-t}"
DRY_RUN="${DRY_RUN:-f}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

run_tgv_nosource() {
  local case_out="$OUT_DIR/tgv_nosource"

  if [[ "$DRY_RUN" == "t" ]]; then
    printf 'dry-run no-source case=tgv out=%s\n' "$case_out" >> "$SUMMARY"
    return
  fi

  OUT_DIR="$case_out" MAXSTEP=1 FEQCHKPT=1 LFILTER=f DIFFTERM=f \
    ATOL=1e-10 RTOL=1e-10 \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_stats_compare.sh"
  printf 'pass no-source case=tgv out=%s\n' "$case_out" >> "$SUMMARY"
}

run_channel_source() {
  local case_out="$OUT_DIR/channel_source"

  if [[ "$DRY_RUN" == "t" ]]; then
    printf 'dry-run source case=channel out=%s\n' "$case_out" >> "$SUMMARY"
    return
  fi

  OUT_DIR="$case_out" MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 \
    LFILTER=t DIFFTERM=t COMPARE_STATS=t COMPARE_FIELD=t \
    STATS_ATOL=1e-8 STATS_RTOL=1e-10 FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 \
    "$ROOT_DIR/tests/gpu_validation/run_channel_phased_compare.sh"
  printf 'pass source case=channel out=%s\n' "$case_out" >> "$SUMMARY"
}

run_rti_source() {
  local case_out="$OUT_DIR/rti_source"

  if [[ "$DRY_RUN" == "t" ]]; then
    printf 'dry-run source case=rti out=%s\n' "$case_out" >> "$SUMMARY"
    return
  fi

  OUT_DIR="$case_out" MAXSTEP=5 FEQCHKPT=5 GRID=32,64,32 \
    LFILTER=t DIFFTERM=t NP=2 TOPOLOGY=1,2,1 \
    COMPARE_STATS=t COMPARE_FIELD=t \
    STATS_ATOL=1e-10 STATS_RTOL=1e-10 FIELD_ATOL=1e-8 FIELD_RTOL=1e-10 \
    "$ROOT_DIR/tests/gpu_validation/run_rti_phasej_compare.sh"
  printf 'pass source case=rti np=2 topology=1,2,1 out=%s\n' "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

if [[ "$RUN_NOSOURCE" == "t" ]]; then
  run_tgv_nosource
fi

if [[ "$RUN_CHANNEL" == "t" ]]; then
  run_channel_source
fi

if [[ "$RUN_RTI" == "t" ]]; then
  run_rti_source
fi
