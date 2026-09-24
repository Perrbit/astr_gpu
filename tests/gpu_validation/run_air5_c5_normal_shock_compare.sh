#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTLET_REFERENCE_LENGTH="${OUTLET_REFERENCE_LENGTH:-}"
outlet_args=()
if [[ -n "$OUTLET_REFERENCE_LENGTH" ]]; then
  outlet_args=(--outlet-reference-length "$OUTLET_REFERENCE_LENGTH")
fi
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_normal_shock}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-32,6,6}"
MAXSTEP="${MAXSTEP:-0}"
DELTAT="${DELTAT:-1.d-8}"
DIFFTERM="${DIFFTERM:-f}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-2e-9}"
RTOL="${RTOL:-2e-10}"
MAX_CFL="${MAX_CFL:-1.0}"
labels=pre_chemistry,post_chemistry,pre_rhs,post_update,post_transport
# GPU startup prepares the pressure outlet before the first pre_chemistry dump.
if [[ -n "$OUTLET_REFERENCE_LENGTH" ]]; then
  labels=post_chemistry,pre_rhs,post_update,post_transport
fi

check_cfl() {
  local log_file="$1"
  local observed
  observed="$(awk '/current CFL:/ {value=$3} END {if (value == "") exit 1; print value}' "$log_file")"
  python3 -c 'import math,sys; value=float(sys.argv[1]); limit=float(sys.argv[2]); sys.exit(0 if math.isfinite(value) and value < limit else 1)' "$observed" "$MAX_CFL"
}

IFS=',' read -r TOPOLOGY_X TOPOLOGY_Y TOPOLOGY_Z <<< "$TOPOLOGY"
if (( TOPOLOGY_X * TOPOLOGY_Y * TOPOLOGY_Z != NP )); then
  echo "normal-shock gate requires TOPOLOGY product equal to NP" >&2
  exit 2
fi
if [[ ! -x "$EXE" ]]; then
  echo "ASTR executable is missing or not executable: $EXE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$TMP_DIR"
for mode in cpu gpu; do
  use_gpu=f
  if [[ "$mode" == gpu ]]; then use_gpu=t; fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
    --destination "$OUT_DIR/$mode" \
    --grid "$GRID" \
    --maxstep "$MAXSTEP" \
    --deltat "$DELTAT" \
    --diffterm "$DIFFTERM" \
    --lfilter f \
    --use-gpu "$use_gpu" \
    --initial-condition normal-shock
  python3 "$ROOT_DIR/tests/gpu_validation/generate_air5_normal_shock_states.py" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --output "$OUT_DIR/$mode/datin/air5_normal_shock_states.dat" "${outlet_args[@]}"
  mkdir -p "$OUT_DIR/$mode/validation"
  (
    cd "$OUT_DIR/$mode"
    TMPDIR="$TMP_DIR" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
      ASTR_VALIDATION_RHS_STEP=0 \
      ASTR_VALIDATION_RHS_STEP_SECONDARY="$MAXSTEP" \
      ASTR_AIR5_C4_CONSERVATION=f \
      ASTR_AIR5_SOURCE_MODE=coupled \
      mpirun -np "$NP" "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
  )
  grep -F "ASTR_AIR5_SOURCE_MODE=coupled" "$OUT_DIR/$mode/$mode.log" >/dev/null
  check_cfl "$OUT_DIR/$mode/$mode.log"
done

python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
  --cpu-prefix "$OUT_DIR/cpu/validation/air5" \
  --gpu-prefix "$OUT_DIR/gpu/validation/air5" \
  --report "$OUT_DIR/cpu_gpu_phase_compare.txt" \
  --labels "$labels" \
  --atol "$ATOL" \
  --rtol "$RTOL" \
  --active-only

for mode in cpu gpu; do
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_normal_shock.py" \
    --prefix "$OUT_DIR/$mode/validation/air5" \
    --boundary-states "$OUT_DIR/$mode/datin/air5_normal_shock_states.dat" \
    --topology "$TOPOLOGY" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --report "$OUT_DIR/${mode}_normal_shock.txt"
done
