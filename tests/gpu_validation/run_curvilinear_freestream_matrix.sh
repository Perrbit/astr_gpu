#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_freestream_matrix}"
MATRIX="${MATRIX:-1:1,1,1 2:2,1,1 2:1,2,1 2:1,1,2 4:2,2,1 4:2,1,2 4:1,2,2 8:2,2,2}"
GRID="${GRID:-32,32,32}"
AMPLITUDE="${AMPLITUDE:-0.15}"
MAXSTEP="${MAXSTEP:-5}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

for entry in $MATRIX; do
  np="${entry%%:*}"
  topology="${entry#*:}"
  tag="$(topology_tag "$topology")"
  case_out="$OUT_DIR/np${np}_${tag}"

  NP="$np" TOPOLOGY="$topology" GRID="$GRID" AMPLITUDE="$AMPLITUDE" \
    MAXSTEP="$MAXSTEP" OUT_DIR="$case_out" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_freestream_compare.sh"

  printf 'pass np=%s topology=%s grid=%s amplitude=%s maxstep=%s out=%s\n' \
    "$np" "$topology" "$GRID" "$AMPLITUDE" "$MAXSTEP" "$case_out" >> "$SUMMARY"
done
