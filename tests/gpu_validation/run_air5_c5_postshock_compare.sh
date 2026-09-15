#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_postshock_compare}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-48,6,6}"
MAXSTEP="${MAXSTEP:-0}"
DELTAT="${DELTAT:-1.d-8}"
DIFFTERM="${DIFFTERM:-f}"
MPI_NP="${MPI_NP:-${NP:-1}}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-1e-9}"
RTOL="${RTOL:-1e-10}"
BOUNDARY_CUT="${BOUNDARY_CUT:-6}"
MAX_CFL="${MAX_CFL:-1.0}"

check_cfl() {
  local log_file="$1"
  local observed
  observed="$(awk '/current CFL:/ {value=$3} END {if (value == "") exit 1; print value}' "$log_file")"
  python3 -c 'import math,sys; cfl=float(sys.argv[1]); limit=float(sys.argv[2]); print(f"CFL gate: {cfl:.8g} < {limit:.8g}"); sys.exit(0 if math.isfinite(cfl) and cfl < limit else 1)' "$observed" "$MAX_CFL"
}

IFS=',' read -r IA JA KA <<< "$GRID"
if [[ -z "${IA:-}" || -z "${JA:-}" || -z "${KA:-}" ]]; then
  echo "GRID must contain three comma-separated integers" >&2
  exit 2
fi
POINTS=$((IA + 1))

if [[ ! -x "$EXE" ]]; then
  echo "ASTR executable is missing or not executable: $EXE" >&2
  exit 2
fi
mkdir -p "$OUT_DIR" "$TMP_DIR"

for source_mode in chemical vt coupled; do
  case_dir="$OUT_DIR/$source_mode"
  for mode in cpu gpu; do
    use_gpu=f
    if [[ "$mode" == gpu ]]; then use_gpu=t; fi
    python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
      --destination "$case_dir/$mode" \
      --grid "$GRID" \
      --maxstep "$MAXSTEP" \
      --deltat "$DELTAT" \
      --diffterm "$DIFFTERM" \
      --use-gpu "$use_gpu" \
      --initial-condition postshock
    python3 "$ROOT_DIR/tests/gpu_validation/generate_air5_postshock_profile.py" \
      --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
      --output "$case_dir/$mode/datin/air5_postshock_profile.dat" \
      --points "$POINTS" \
      --length 2e-2 \
      --source-mode "$source_mode" \
      --rtol 1e-10
    mkdir -p "$case_dir/$mode/validation"
    (
      cd "$case_dir/$mode"
      TMPDIR="$TMP_DIR" \
        ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
        ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
        ASTR_AIR5_C4_CONSERVATION=f \
        ASTR_AIR5_SOURCE_MODE="$source_mode" \
        mpirun -np "$MPI_NP" "$EXE" run datin/input.air5_c4 > "$mode.log" 2>&1
    )
    grep -F "ASTR_AIR5_SOURCE_MODE=$source_mode" "$case_dir/$mode/$mode.log" >/dev/null
    check_cfl "$case_dir/$mode/$mode.log"
  done

  python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
    --cpu-prefix "$case_dir/cpu/validation/air5" \
    --gpu-prefix "$case_dir/gpu/validation/air5" \
    --report "$case_dir/cpu_gpu_phase_compare.txt" \
    --labels pre_chemistry,post_chemistry,pre_rhs,post_update,post_transport \
    --atol "$ATOL" \
    --rtol "$RTOL" \
    --active-only

  for mode in cpu gpu; do
    python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_postshock.py" \
      --prefix "$case_dir/$mode/validation/air5" \
      --profile "$case_dir/$mode/datin/air5_postshock_profile.dat" \
      --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
      --topology "$TOPOLOGY" \
      --boundary-cut "$BOUNDARY_CUT" \
      --report "$case_dir/${mode}_independent_contract.txt"
  done
done
