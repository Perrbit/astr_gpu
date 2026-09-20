#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_hbl_compare}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-31,31,7}"
MAXSTEP="${MAXSTEP:-0}"
VALIDATION_STEP="${VALIDATION_STEP:-$MAXSTEP}"
VALIDATION_STEP_SECONDARY="${VALIDATION_STEP_SECONDARY:-}"
DELTAT="${DELTAT:-1.d-10}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
LIST_FREQUENCY="${LIST_FREQUENCY:-100}"
MPI_NP="${MPI_NP:-${NP:-1}}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-1e-9}"
RTOL="${RTOL:-1e-10}"
EXTRUSION_SCALED_TOL="${EXTRUSION_SCALED_TOL:-2.0e-10}"
SAME_PHASE_SCALED_TOL="${SAME_PHASE_SCALED_TOL:-}"
MAX_CFL="${MAX_CFL:-1.0}"
REF_LEN="${REF_LEN:-4.41262150017878821e-5}"
HM="${HM:-5}"

check_cfl() {
  local log_file="$1"
  local observed
  observed="$(awk '/current CFL:/ {value=$3} END {if (value == "") exit 1; print value}' "$log_file")"
  python3 -c 'import math,sys; cfl=float(sys.argv[1]); limit=float(sys.argv[2]); print(f"CFL gate: {cfl:.8g} < {limit:.8g}"); sys.exit(0 if math.isfinite(cfl) and cfl < limit else 1)' "$observed" "$MAX_CFL"
}

if [[ ! -x "$EXE" ]]; then
  echo "ASTR executable is missing or not executable: $EXE" >&2
  exit 2
fi
IFS=',' read -r GI GJ GK <<< "$GRID"
IFS=',' read -r TI TJ TK <<< "$TOPOLOGY"
for value in "$GI" "$GJ" "$GK" "$TI" "$TJ" "$TK" "$MPI_NP" "$HM"; do
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "GRID, TOPOLOGY, MPI_NP, and HM must contain positive integers" >&2
    exit 2
  fi
done
for value in "$MAXSTEP" "$VALIDATION_STEP" "$TIMEOUT_SECONDS" "$LIST_FREQUENCY"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "MAXSTEP, VALIDATION_STEP, and TIMEOUT_SECONDS must be non-negative integers" >&2
    exit 2
  fi
done
if (( VALIDATION_STEP > MAXSTEP )); then
  echo "VALIDATION_STEP must not exceed MAXSTEP" >&2
  exit 2
fi
if [[ -n "$VALIDATION_STEP_SECONDARY" ]]; then
  if [[ ! "$VALIDATION_STEP_SECONDARY" =~ ^[0-9]+$ ]]; then
    echo "VALIDATION_STEP_SECONDARY must be a non-negative integer" >&2
    exit 2
  fi
  if (( VALIDATION_STEP_SECONDARY > MAXSTEP )); then
    echo "VALIDATION_STEP_SECONDARY must not exceed MAXSTEP" >&2
    exit 2
  fi
fi
if (( TIMEOUT_SECONDS == 0 )); then
  echo "TIMEOUT_SECONDS must be positive" >&2
  exit 2
fi
if (( LIST_FREQUENCY == 0 )); then
  echo "LIST_FREQUENCY must be positive" >&2
  exit 2
fi
if (( TI * TJ * TK != MPI_NP )); then
  echo "TOPOLOGY product must equal MPI_NP" >&2
  exit 2
fi
minimum_local_extent=$((GI / TI))
minimum_local_extent=$((GJ / TJ < minimum_local_extent ? GJ / TJ : minimum_local_extent))
minimum_local_extent=$((GK / TK < minimum_local_extent ? GK / TK : minimum_local_extent))
if (( minimum_local_extent < HM )); then
  echo "minimum local grid extent $minimum_local_extent is smaller than hm=$HM" >&2
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
    --list-frequency "$LIST_FREQUENCY" \
    --diffterm t \
    --use-gpu "$use_gpu" \
    --initial-condition high-enthalpy-boundary-layer
  mkdir -p "$OUT_DIR/$mode/validation"
  (
    cd "$OUT_DIR/$mode"
    TMPDIR="$TMP_DIR" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
      ASTR_VALIDATION_RHS_STEP="$VALIDATION_STEP" \
      ASTR_VALIDATION_RHS_STEP_SECONDARY="$VALIDATION_STEP_SECONDARY" \
      ASTR_AIR5_C4_CONSERVATION=f \
      ASTR_AIR5_SOURCE_MODE="coupled" \
      OMPI_MCA_coll='^hcoll,ucc' \
      OMPI_MCA_pml=ob1 \
      OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_osc=pt2pt \
      OMPI_MCA_opal_cuda_support=0 \
      UCX_MEMTYPE_CACHE=n \
      timeout --kill-after=10s "${TIMEOUT_SECONDS}s" mpirun --oversubscribe -np "$MPI_NP" \
        "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
  )
  grep -F "ASTR_AIR5_SOURCE_MODE=coupled" "$OUT_DIR/$mode/$mode.log" >/dev/null
  check_cfl "$OUT_DIR/$mode/$mode.log"
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_hbl.py" \
    --prefix "$OUT_DIR/$mode/validation/air5" \
    --profile "$OUT_DIR/$mode/datin/air5_hbl_profile.dat" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --ref-len "$REF_LEN" \
    --step "$VALIDATION_STEP" \
    --extrusion-scaled-tol "$EXTRUSION_SCALED_TOL" \
    --report "$OUT_DIR/${mode}_hbl_contract.txt"
  if [[ -n "$VALIDATION_STEP_SECONDARY" ]]; then
    python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_hbl.py" \
      --prefix "$OUT_DIR/$mode/validation/air5" \
      --profile "$OUT_DIR/$mode/datin/air5_hbl_profile.dat" \
      --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
      --ref-len "$REF_LEN" \
      --step "$VALIDATION_STEP_SECONDARY" \
      --extrusion-scaled-tol "$EXTRUSION_SCALED_TOL" \
      --report "$OUT_DIR/${mode}_hbl_contract_step${VALIDATION_STEP_SECONDARY}.txt"
  fi
done

compare_args=(
  --cpu-prefix "$OUT_DIR/cpu/validation/air5"
  --gpu-prefix "$OUT_DIR/gpu/validation/air5"
  --report "$OUT_DIR/cpu_gpu_same_phase_compare.txt"
  --labels post_chemistry,pre_rhs
  --atol "$ATOL"
  --rtol "$RTOL"
  --active-only
)
if [[ -n "$SAME_PHASE_SCALED_TOL" ]]; then
  compare_args+=(--scaled-tol "$SAME_PHASE_SCALED_TOL")
fi
python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
  "${compare_args[@]}"
