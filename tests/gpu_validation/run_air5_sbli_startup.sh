#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXE="${EXE:-$ROOT/tests/gpu_validation/out/c4_cuda_build_4/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT/tests/gpu_validation/out/air5_sbli_startup}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-$NP,1,1}"
GRID="${GRID:-31,31,7}"
MAXSTEP="${MAXSTEP:-2}"
DELTAT="${DELTAT:-1.d-11}"
[[ "$NP" == 1 || "$NP" == 2 ]] || { echo 'Initial gate supports NP=1/2 only' >&2; exit 2; }
IFS=, read -r TX TY TZ <<< "$TOPOLOGY"
[[ "$TOPOLOGY" =~ ^[1-9][0-9]*,[1-9][0-9]*,[1-9][0-9]*$ ]] || exit 2
(( TX * TY * TZ == NP )) || { echo 'Topology does not match NP' >&2; exit 2; }
[[ "$MAXSTEP" =~ ^[0-9]+$ ]] || { echo 'Invalid MAXSTEP' >&2; exit 2; }
[[ -x "$EXE" ]] || { echo "Missing executable: $EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "Refusing to overwrite $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
for mode in cpu gpu; do
  usegpu=f
  if [[ "$mode" == gpu ]]; then usegpu=t; fi
  python3 "$ROOT/tests/gpu_validation/prepare_air5_sbli_case.py" \
    --destination "$OUT_DIR/$mode" --grid "$GRID" --maxstep "$MAXSTEP" \
    --deltat "$DELTAT" --use-gpu "$usegpu"
  (
    cd "$OUT_DIR/$mode"
    env ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
      ASTR_VALIDATION_RHS_STEP=0 ASTR_VALIDATION_RHS_STEP_SECONDARY="$MAXSTEP" \
      ASTR_AIR5_C4_CONSERVATION=f ASTR_AIR5_SOURCE_MODE=coupled \
      ASTR_GPU_PRECISION_MODE=fp64 ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_FILTER_WORKSPACE=full \
      OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_osc=pt2pt OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      timeout --kill-after=10s "${TIMEOUT_SECONDS:-300}s" \
      mpirun --oversubscribe -np "$NP" "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
  )
  grep -F 'The job is done!' "$OUT_DIR/$mode/$mode.log" >/dev/null
  grep -F 'ASTR_AIR5_SOURCE_MODE=coupled' "$OUT_DIR/$mode/$mode.log" >/dev/null
  sensor_args=()
  if [[ "$mode" == gpu ]]; then sensor_args=(--compare-sensors-with "$OUT_DIR/cpu"); fi
  python3 "$ROOT/tests/gpu_validation/check_air5_sbli_startup.py" \
    --case "$OUT_DIR/$mode" --topology "$TOPOLOGY" "${sensor_args[@]}"
done
python3 "$ROOT/tests/gpu_validation/compare_q_validation_snapshots.py" \
  --cpu-prefix "$OUT_DIR/cpu/validation/air5" --gpu-prefix "$OUT_DIR/gpu/validation/air5" \
  --report "$OUT_DIR/cpu_gpu_same_phase.txt" \
  --labels post_chemistry,pre_rhs,post_update,post_transport --active-only \
  --atol 1e-9 --rtol 1e-10
