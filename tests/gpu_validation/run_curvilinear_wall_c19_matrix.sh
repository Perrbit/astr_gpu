#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_wall_c19_matrix}"
MATRIX="${MATRIX:-41:t:1:1,1,1 42:t:1:1,1,1 411:t:1:1,1,1 411:f:1:1,1,1 421:t:1:1,1,1 411:f:2:1,2,1 421:t:2:1,2,1 41:t:2:1,2,1 411:f:4:2,2,1 421:t:4:2,2,1 41:t:4:2,2,1 411:f:8:2,2,2 421:t:8:2,2,2 41:t:8:2,2,2}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

mkdir -p "$OUT_DIR"
: > "$SUMMARY"
for entry in $MATRIX; do
  IFS=: read -r kind blowing np topology <<< "$entry"
  tag="${topology//,/x}"
  case_out="$OUT_DIR/bc${kind}_blow${blowing}_np${np}_${tag}"
  BC_KIND="$kind" WALL_BLOWING="$blowing" NP="$np" TOPOLOGY="$topology" \
    OUT_DIR="$case_out" MAXSTEP="${MAXSTEP:-1}" FEQCHKPT="${FEQCHKPT:-1}" \
    LFILTER="${LFILTER:-f}" DIFFTERM="${DIFFTERM:-f}" GRID="${GRID:-32,32,32}" \
    FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall_c19_compare.sh"
  printf 'pass kind=%s blowing=%s np=%s topology=%s out=%s\n' \
    "$kind" "$blowing" "$np" "$topology" "$case_out" >> "$SUMMARY"
done

signed_out="$OUT_DIR/bc41_suction_np1_1x1x1"
BC_KIND=41 WALL_BLOWING=t WALL_AMPLITUDE=-0.01 NP=1 TOPOLOGY=1,1,1 \
  OUT_DIR="$signed_out" MAXSTEP="${MAXSTEP:-1}" FEQCHKPT="${FEQCHKPT:-1}" \
  LFILTER="${LFILTER:-f}" DIFFTERM="${DIFFTERM:-f}" GRID="${GRID:-32,32,32}" \
  FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall_c19_compare.sh"
printf 'pass kind=41 blowing=suction amplitude=-0.01 np=1 topology=1,1,1 out=%s\n' \
  "$signed_out" >> "$SUMMARY"

for kind_blowing in 41:t 411:f 421:t; do
  IFS=: read -r kind blowing <<< "$kind_blowing"
  baseline="$OUT_DIR/bc${kind}_blow${blowing}_np1_1x1x1"
  for np_topology in 2:1,2,1 4:2,2,1 8:2,2,2; do
    IFS=: read -r np topology <<< "$np_topology"
    candidate="$OUT_DIR/bc${kind}_blow${blowing}_np${np}_${topology//,/x}"
    python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
      --cpu "$baseline/cpu/outdat/rk_complete_snapshot.h5" \
      --gpu "$candidate/cpu/outdat/rk_complete_snapshot.h5" \
      --report "$candidate/cpu_topology_compare.txt" \
      --atol "${TOPOLOGY_ATOL:-1e-10}" --rtol "${TOPOLOGY_RTOL:-1e-10}"
    python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
      --cpu "$baseline/gpu" --gpu "$candidate/gpu" \
      --report "$candidate/gpu_topology_compare.txt" \
      --atol "${TOPOLOGY_ATOL:-1e-10}" --rtol "${TOPOLOGY_RTOL:-1e-10}"
  done
done
