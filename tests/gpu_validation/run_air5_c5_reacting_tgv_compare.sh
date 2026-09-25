#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_reacting_tgv_compare}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-12,12,12}"
MAXSTEP="${MAXSTEP:-0}"
LFILTER="${LFILTER:-f}"
DELTAT="${DELTAT:-2.d-10}"
MPI_NP="${MPI_NP:-${NP:-1}}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-1e-8}"
RTOL="${RTOL:-2e-11}"
MAX_CFL="${MAX_CFL:-1.0}"

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
mkdir -p "$OUT_DIR" "$TMP_DIR"

for mode in cpu gpu; do
  use_gpu=f
  conservation=f
  if [[ "$mode" == gpu ]]; then
    use_gpu=t
    conservation=t
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
    --destination "$OUT_DIR/$mode" \
    --grid "$GRID" \
    --maxstep "$MAXSTEP" \
    --deltat "$DELTAT" \
    --diffterm t \
    --lfilter "$LFILTER" \
    --use-gpu "$use_gpu" \
    --initial-condition high-temperature-tgv
  mkdir -p "$OUT_DIR/$mode/validation"
  (
    cd "$OUT_DIR/$mode"
    TMPDIR="$TMP_DIR" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
      ASTR_AIR5_C4_CONSERVATION="$conservation" \
      ASTR_AIR5_SOURCE_MODE="coupled" \
      mpirun -np "$MPI_NP" "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
  )
  grep -F "ASTR_AIR5_SOURCE_MODE=coupled" "$OUT_DIR/$mode/$mode.log" >/dev/null
  check_cfl "$OUT_DIR/$mode/$mode.log"
done

python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
  --cpu-prefix "$OUT_DIR/cpu/validation/air5" \
  --gpu-prefix "$OUT_DIR/gpu/validation/air5" \
  --report "$OUT_DIR/cpu_gpu_phase_compare.txt" \
  --labels pre_chemistry,post_chemistry,pre_rhs,post_update,post_transport \
  --atol "$ATOL" \
  --rtol "$RTOL" \
  --active-only

for mode in cpu gpu; do
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_reacting_tgv.py" \
    --prefix "$OUT_DIR/$mode/validation/air5" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --report "$OUT_DIR/${mode}_reacting_tgv_contract.txt"
done

python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c4_conservation.py" \
  --input "$OUT_DIR/gpu/air5_c4_conservation.dat" \
  --report "$OUT_DIR/gpu_global_conservation.txt" \
  --allow-species-change
