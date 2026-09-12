#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE_DRIVER="$ROOT_DIR/tests/gpu_validation/run_mp3_characteristic_flux_xphysical_compare.sh"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp3_characteristic_flux_xphysical_matrix_$STAMP}"
TOLERANCE_FILE="${TOLERANCE_FILE:-}"
[[ -n "$TOLERANCE_FILE" ]] || { echo "TOLERANCE_FILE is required" >&2; exit 2; }
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"
[[ "$TOLERANCE_FILE" == /* ]] || TOLERANCE_FILE="$PWD/$TOLERANCE_FILE"
[[ -f "$TOLERANCE_FILE" ]] || { echo "missing tolerance file: $TOLERANCE_FILE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite MP3 x-physical matrix: $OUT_DIR" >&2; exit 2; }

source "$TOLERANCE_FILE"
mkdir -p "$OUT_DIR"

NP=1 TOPOLOGY=1,1,1 GRID=400,8,8 MAXSTEP=3 \
  OUT_DIR="$OUT_DIR/np1" \
  CANDIDATE_FIELD_ATOL="$CANDIDATE_FIELD_ATOL" \
  CANDIDATE_FIELD_RTOL="$CANDIDATE_FIELD_RTOL" \
  CANDIDATE_STATS_ATOL="$CANDIDATE_STATS_ATOL" \
  CANDIDATE_STATS_RTOL="$CANDIDATE_STATS_RTOL" "$COMPARE_DRIVER"

NP=2 TOPOLOGY=2,1,1 GRID=400,8,8 MAXSTEP=3 \
  OUT_DIR="$OUT_DIR/np2_2x1x1" \
  CANDIDATE_FIELD_ATOL="$CANDIDATE_FIELD_ATOL" \
  CANDIDATE_FIELD_RTOL="$CANDIDATE_FIELD_RTOL" \
  CANDIDATE_STATS_ATOL="$CANDIDATE_STATS_ATOL" \
  CANDIDATE_STATS_RTOL="$CANDIDATE_STATS_RTOL" "$COMPARE_DRIVER"

echo "MP3 characteristic_flux x-physical matrix passed: $OUT_DIR"
