#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c5_hbl_memcheck}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-31,31,7}"
DELTAT="${DELTAT:-1.d-10}"
MAX_CFL="${MAX_CFL:-1.0}"
REF_LEN="${REF_LEN:-4.41262150017878821e-5}"
GPU_ID="${GPU_ID:-0}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi
if [[ "$TMP_DIR" != /* ]]; then
  TMP_DIR="$ROOT_DIR/$TMP_DIR"
fi
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
  echo "refusing to overwrite HBL memcheck evidence directory: $OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR" "$TMP_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
  --destination "$CASE_DIR" \
  --grid "$GRID" \
  --maxstep 0 \
  --deltat "$DELTAT" \
  --diffterm t \
  --use-gpu t \
  --initial-condition high-enthalpy-boundary-layer
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
    timeout --kill-after=10s 600s mpirun -np 1 \
      compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 \
      "$EXE" run datin/input.air5_c4 > "$LOG_FILE" 2>&1
)

grep -Fq 'ASTR_AIR5_SOURCE_MODE=coupled' "$LOG_FILE"
grep -Fq 'The job is done!' "$LOG_FILE"
grep -Fq 'ERROR SUMMARY: 0 errors' "$LOG_FILE"
grep -Fq 'LEAK SUMMARY: 0 bytes leaked' "$LOG_FILE"

observed_cfl="$(awk '/current CFL:/ {value=$3} END {if (value == "") exit 1; print value}' "$LOG_FILE")"
python3 -c 'import math,sys; cfl=float(sys.argv[1]); limit=float(sys.argv[2]); print(f"CFL gate: {cfl:.8g} < {limit:.8g}"); sys.exit(0 if math.isfinite(cfl) and cfl < limit else 1)' \
  "$observed_cfl" "$MAX_CFL"

python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_hbl.py" \
  --prefix "$CASE_DIR/validation/air5" \
  --profile "$CASE_DIR/datin/air5_hbl_profile.dat" \
  --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
  --ref-len "$REF_LEN" \
  --report "$OUT_DIR/hbl_contract.txt"

printf 'AIR5_C5_HBL_MEMCHECK_PASS\n'
