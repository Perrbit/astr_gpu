#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_embedded_reactor_compare}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-6,6,6}"
MAXSTEP="${MAXSTEP:-0}"
DELTAT="${DELTAT:-2.d-10}"
MPI_NP="${MPI_NP:-${NP:-1}}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-9}"
ELEMENT_RTOL="${ELEMENT_RTOL:-5e-12}"

mkdir -p "$OUT_DIR" "$TMP_DIR"

for mode in cpu gpu; do
  use_gpu=f
  if [[ "$mode" == gpu ]]; then use_gpu=t; fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
    --destination "$OUT_DIR/$mode" \
    --grid "$GRID" \
    --maxstep "$MAXSTEP" \
    --deltat "$DELTAT" \
    --diffterm t \
    --use-gpu "$use_gpu" \
    --initial-condition reactor
  mkdir -p "$OUT_DIR/$mode/validation"
  (
    cd "$OUT_DIR/$mode"
    TMPDIR="$TMP_DIR" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
      ASTR_AIR5_C4_CONSERVATION=f \
      mpirun -np "$MPI_NP" "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
  )
done

python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
  --cpu-prefix "$OUT_DIR/cpu/validation/air5" \
  --gpu-prefix "$OUT_DIR/gpu/validation/air5" \
  --report "$OUT_DIR/cpu_gpu_phase_compare.txt" \
  --labels pre_chemistry,post_chemistry,post_transport \
  --atol "$ATOL" \
  --rtol "$RTOL" \
  --active-only

for mode in cpu gpu; do
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_embedded_reactor.py" \
    --prefix "$OUT_DIR/$mode/validation/air5" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --report "$OUT_DIR/${mode}_reactor_contract.txt" \
    --atol "$ATOL" \
    --element-rtol "$ELEMENT_RTOL"
done
