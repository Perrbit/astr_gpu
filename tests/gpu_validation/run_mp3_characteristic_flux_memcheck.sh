#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp3_characteristic_flux_memcheck_$STAMP}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"
[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
command -v compute-sanitizer >/dev/null || { echo "compute-sanitizer is unavailable" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite MP3 memcheck evidence: $OUT_DIR" >&2; exit 2; }

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Shuosher" --dst-case "$OUT_DIR/case" \
  --input-name input.shuosher --flowtype shuosher --homogeneous t,t,t \
  --bctype 1,1,1,1,1,1 --use-gpu t --maxstep 1 --feqchkpt 9999 \
  --lfilter f --diffterm f --scheme 643e --conschm 543e --difschm 643e \
  --recon-schem 3 --lchardecomp t --grid 40,8,8 --deltat 1.d-4

(
  cd "$OUT_DIR/case"
  ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
  ASTR_SHOCK_SENSOR_DUMP="$OUT_DIR/shock_sensor.dat" \
  ASTR_GPU_PRECISION_MODE=mixed_workspace \
  ASTR_GPU_MIXED_CANDIDATE=characteristic_flux \
  ASTR_GPU_SYNC_MODE=explicit OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
  OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
    mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
      "$GPU_EXE" run datin/input.shuosher > "$OUT_DIR/memcheck.log" 2>&1
)

rg -q 'The job is done!' "$OUT_DIR/memcheck.log"
rg -q 'ERROR SUMMARY: 0 errors' "$OUT_DIR/memcheck.log"
echo "MP3 characteristic_flux memcheck passed: $OUT_DIR"
