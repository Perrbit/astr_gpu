#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/Taylor_Green_Vortex}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_derivative_tgv_$STAMP}"
CANDIDATE="${MP2_CANDIDATE:-derivative}"
CANDIDATE_TARGET="gpu_${CANDIDATE}"
[[ "$CANDIDATE" == "derivative" || "$CANDIDATE" == "viscous_flux" ]] || {
  echo "MP2_CANDIDATE must be derivative or viscous_flux" >&2; exit 2;
}
GRID="${GRID:-32,32,32}"
MAXSTEP="${MAXSTEP:-10}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
DELTAT="${DELTAT:-1.d-4}"
FIELD_ATOL="${FIELD_ATOL:-0}"
FIELD_RTOL="${FIELD_RTOL:-0}"
STATS_ATOL="${STATS_ATOL:-1e-8}"
STATS_RTOL="${STATS_RTOL:-0}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"

[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "OUT_DIR already exists: $OUT_DIR" >&2; exit 2; }

prepare_case() {
  local target="$1"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" --dst-case "$OUT_DIR/$target" \
    --input-name input.tgv --flowtype tgv --homogeneous t,t,t \
    --bctype 1,1,1,1,1,1 --use-gpu t --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" --lfilter f --diffterm t \
    --conschm 643e --difschm 643e --recon-schem 0 --lchardecomp f \
    --grid "$GRID" --deltat "$DELTAT"
}

run_case() {
  local target="$1" mode="$2" candidate="$3"
  (
    cd "$OUT_DIR/$target"
    env OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
      OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_PRECISION_MODE="$mode" ASTR_GPU_MIXED_CANDIDATE="$candidate" \
      mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv > run.log 2>&1
  )
}

mkdir -p "$OUT_DIR"
prepare_case gpu_fp64
prepare_case "$CANDIDATE_TARGET"
run_case gpu_fp64 fp64 flux
run_case "$CANDIDATE_TARGET" mixed_workspace "$CANDIDATE"

rg -q '^ASTR_GPU_PRECISION_MODE=fp64$' "$OUT_DIR/gpu_fp64/run.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=none$' "$OUT_DIR/gpu_fp64/run.log"
rg -q '^ASTR_GPU_PRECISION_MODE=mixed_workspace$' "$OUT_DIR/$CANDIDATE_TARGET/run.log"
rg -q "^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=${CANDIDATE}$" "$OUT_DIR/$CANDIDATE_TARGET/run.log"

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/$CANDIDATE_TARGET" \
  --report "$OUT_DIR/fp64_vs_${CANDIDATE}_flowfield.txt" \
  --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/$CANDIDATE_TARGET" \
  --report "$OUT_DIR/fp64_vs_${CANDIDATE}_flowstate.txt" \
  --atol "$STATS_ATOL" --rtol "$STATS_RTOL"

{
  rg '^ASTR_GPU_(PRECISION_MODE|MIXED_CANDIDATE|ACTIVE_MIXED_WORKSPACE|MIXED_WORKSPACE_BYTES|FP64_WORKSPACE_BYTES)=' \
    "$OUT_DIR/gpu_fp64/run.log"
  rg '^ASTR_GPU_(PRECISION_MODE|MIXED_CANDIDATE|ACTIVE_MIXED_WORKSPACE|MIXED_WORKSPACE_BYTES|FP64_WORKSPACE_BYTES)=' \
    "$OUT_DIR/$CANDIDATE_TARGET/run.log"
} > "$OUT_DIR/workspace_summary.txt"

echo "MP2 $CANDIDATE TGV comparison passed: $OUT_DIR"
