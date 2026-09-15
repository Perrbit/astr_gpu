#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_frozen_transport_compare}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-48,6,6}"
MAXSTEP="${MAXSTEP:-0}"
DELTAT="${DELTAT:-5.d-7}"
MPI_NP="${MPI_NP:-${NP:-1}}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"
RHS_ATOL="${RHS_ATOL:-1e-8}"
RHS_RTOL="${RHS_RTOL:-2e-10}"
STRUCTURAL_TOL="${STRUCTURAL_TOL:-2e-11}"
TRANSLATION_TOL="${TRANSLATION_TOL:-2e-9}"
MAX_CFL="${MAX_CFL:-1.0}"
DELTAT_PY="${DELTAT//d/e}"
DELTAT_PY="${DELTAT_PY//D/e}"

check_cfl() {
  local log_file="$1"
  local observed
  observed="$(awk '/current CFL:/ {value=$3} END {if (value == "") exit 1; print value}' "$log_file")"
  python3 -c 'import math,sys; cfl=float(sys.argv[1]); limit=float(sys.argv[2]); print(f"CFL gate: {cfl:.8g} < {limit:.8g}"); sys.exit(0 if math.isfinite(cfl) and cfl < limit else 1)' "$observed" "$MAX_CFL"
}

mkdir -p "$OUT_DIR" "$TMP_DIR"

for frozen_case in advection-wave diffusion-layer ev-pulse; do
  diffterm=t
  if [[ "$frozen_case" == "advection-wave" ]]; then diffterm=f; fi
  case_dir="$OUT_DIR/$frozen_case"

  for mode in cpu gpu; do
    use_gpu=f
    if [[ "$mode" == "gpu" ]]; then use_gpu=t; fi
    python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
      --destination "$case_dir/$mode" \
      --grid "$GRID" \
      --maxstep "$MAXSTEP" \
      --deltat "$DELTAT" \
      --diffterm "$diffterm" \
      --use-gpu "$use_gpu" \
      --initial-condition "$frozen_case"
    mkdir -p "$case_dir/$mode/validation"
    (
      cd "$case_dir/$mode"
      TMPDIR="$TMP_DIR" \
        ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
        ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
        ASTR_AIR5_C4_CONSERVATION=f \
      mpirun -np "$MPI_NP" "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
    )
    check_cfl "$case_dir/$mode/$mode.log"
  done

  python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
    --cpu-prefix "$case_dir/cpu/validation/air5" \
    --gpu-prefix "$case_dir/gpu/validation/air5" \
    --report "$case_dir/cpu_gpu_compare.txt" \
    --labels pre_rhs,post_update \
    --atol "$ATOL" \
    --rtol "$RTOL" \
    --active-only

  for mode in cpu gpu; do
    python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_frozen_transport.py" \
      --prefix "$case_dir/$mode/validation/air5" \
      --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
      --case "$frozen_case" \
      --grid "$GRID" \
      --topology "$TOPOLOGY" \
      --deltat "$DELTAT_PY" \
      --ref-len 1e-2 \
      --rhs-atol "$RHS_ATOL" \
      --rhs-rtol "$RHS_RTOL" \
      --structural-tol "$STRUCTURAL_TOL" \
      --translation-tol "$TRANSLATION_TOL" \
      --report "$case_dir/${mode}_independent_contract.txt"
  done
done
