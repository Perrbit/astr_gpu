#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_derivative_memcheck_$STAMP}"
CANDIDATE="${MP2_CANDIDATE:-derivative}"
[[ "$CANDIDATE" == "derivative" || "$CANDIDATE" == "viscous_flux" ]] || {
  echo "MP2_CANDIDATE must be derivative or viscous_flux" >&2; exit 2;
}
GRID="${GRID:-16,16,16}"
MAXSTEP="${MAXSTEP:-1}"
GPU_ID="${GPU_ID:-0}"
CASE_DIR="$OUT_DIR/case"

[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
command -v compute-sanitizer >/dev/null 2>&1 || { echo "compute-sanitizer not found" >&2; exit 127; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite memcheck evidence directory: $OUT_DIR" >&2; exit 2; }

mkdir -p "$OUT_DIR"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
  --dst-case "$CASE_DIR" --input-name input.tgv --flowtype tgv \
  --homogeneous t,t,t --bctype 1,1,1,1,1,1 --use-gpu t \
  --grid "$GRID" --maxstep "$MAXSTEP" --feqchkpt 9999 \
  --lfilter f --diffterm t --conschm 643e --difschm 643e \
  --recon-schem 0 --lchardecomp f --deltat 1.d-4

(
  cd "$CASE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_GPU_PRECISION_MODE=mixed_workspace \
    ASTR_GPU_MIXED_CANDIDATE="$CANDIDATE" ASTR_GPU_SYNC_MODE=explicit \
    OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
    mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
      "$GPU_EXE" run datin/input.tgv > "$OUT_DIR/memcheck.log" 2>&1
)

rg -q "^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=${CANDIDATE}$" "$OUT_DIR/memcheck.log"
rg -q 'The job is done!' "$OUT_DIR/memcheck.log"
rg -q 'ERROR SUMMARY: 0 errors' "$OUT_DIR/memcheck.log"
tail -20 "$OUT_DIR/memcheck.log"
