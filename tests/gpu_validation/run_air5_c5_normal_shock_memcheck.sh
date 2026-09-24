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
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_normal_shock_memcheck}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-12,6,6}"
DELTAT="${DELTAT:-1.d-8}"
GPU_ID="${GPU_ID:-0}"
CASE_DIR="$OUT_DIR/case"
LOG_FILE="$OUT_DIR/memcheck.log"

if [[ ! -x "$EXE" ]]; then
  echo "ASTR executable is missing or not executable: $EXE" >&2
  exit 2
fi
command -v compute-sanitizer >/dev/null 2>&1 || {
  echo "compute-sanitizer is required" >&2
  exit 127
}
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite normal-shock memcheck evidence: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR" "$TMP_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
  --destination "$CASE_DIR" \
  --grid "$GRID" \
  --maxstep 0 \
  --deltat "$DELTAT" \
  --diffterm t \
  --lfilter f \
  --use-gpu t \
  --initial-condition normal-shock
python3 "$ROOT_DIR/tests/gpu_validation/generate_air5_normal_shock_states.py" \
  --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
  --output "$CASE_DIR/datin/air5_normal_shock_states.dat" "${outlet_args[@]}"
mkdir -p "$CASE_DIR/validation"

(
  cd "$CASE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    TMPDIR="$TMP_DIR" \
    ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
    ASTR_AIR5_C4_CONSERVATION=f \
    ASTR_AIR5_SOURCE_MODE=coupled \
    OMPI_MCA_coll='^hcoll,ucc' \
    OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self \
    OMPI_MCA_osc=pt2pt \
    OMPI_MCA_opal_cuda_support=0 \
    UCX_MEMTYPE_CACHE=n \
    timeout --kill-after=10s 1800s mpirun -np 1 \
      compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
      "$EXE" run datin/input.air5_c4 > "$LOG_FILE" 2>&1
)

grep -Fq 'ASTR_AIR5_SOURCE_MODE=coupled' "$LOG_FILE"
grep -Fq 'The job is done!' "$LOG_FILE"
grep -Fq 'ERROR SUMMARY: 0 errors' "$LOG_FILE"
grep -Fq 'LEAK SUMMARY: 0 bytes leaked' "$LOG_FILE"

python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_normal_shock.py" \
  --prefix "$CASE_DIR/validation/air5" \
  --boundary-states "$CASE_DIR/datin/air5_normal_shock_states.dat" \
  --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
  --report "$OUT_DIR/normal_shock_contract.txt"

printf 'AIR5_C5_NORMAL_SHOCK_MEMCHECK_PASS\n'
