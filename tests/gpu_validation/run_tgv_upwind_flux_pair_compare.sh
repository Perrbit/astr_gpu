#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_upwind_flux_pair_compare}"
GRID="${GRID:-32,32,32}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
RECON_SCHEM="${RECON_SCHEM:-1}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
FIELD_ATOL="${FIELD_ATOL:-1e-12}"
MIXED_FIELD_ATOL="${MIXED_FIELD_ATOL:-1e-8}"
STATS_ATOL="${STATS_ATOL:-1e-14}"

if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite comparison evidence directory: $OUT_DIR" >&2
  exit 2
fi
if [[ "$RECON_SCHEM" != "1" && "$RECON_SCHEM" != "3" ]]; then
  echo "RECON_SCHEM must be 1 (WENO7) or 3 (MP7)" >&2
  exit 2
fi

prepare_case() {
  local target="$1"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
    --dst-case "$OUT_DIR/$target" --input-name input.tgv --flowtype tgv \
    --homogeneous t,t,t --bctype 1,1,1,1,1,1 --use-gpu t \
    --grid "$GRID" --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" \
    --lfilter f --diffterm f --scheme 643e --conschm 543e \
    --difschm 643e --recon-schem "$RECON_SCHEM" --lchardecomp f \
    --deltat 1.d-4
}

run_case() {
  local target="$1" precision="$2" pair_mode="$3"
  (
    cd "$OUT_DIR/$target"
    env CUDA_VISIBLE_DEVICES=0 OMPI_MCA_coll_hcoll_enable=0 \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_GPU_PRECISION_MODE="$precision" \
      ASTR_GPU_FLUX_PAIR_MODE="$pair_mode" \
      ASTR_GPU_SYNC_MODE=explicit \
      mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv > run.log 2>&1
  )
  rg -q "^ASTR_GPU_PRECISION_MODE=$precision$" "$OUT_DIR/$target/run.log"
  rg -q "^ASTR_GPU_FLUX_PAIR_MODE=$pair_mode$" "$OUT_DIR/$target/run.log"
}

compare_pair() {
  local label="$1" reference="$2" candidate="$3" field_atol="$4"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
    --cpu "$OUT_DIR/$reference" --gpu "$OUT_DIR/$candidate" \
    --report "$OUT_DIR/${label}_flowstate.txt" --atol "$STATS_ATOL" --rtol 0
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$OUT_DIR/$reference" --gpu "$OUT_DIR/$candidate" \
    --report "$OUT_DIR/${label}_flowfield.txt" --atol "$field_atol" --rtol 0
}

mkdir -p "$OUT_DIR"
for target in fp64_split fp64_fused mixed_split mixed_fused; do
  prepare_case "$target"
done
run_case fp64_split fp64 split
run_case fp64_fused fp64 fused
run_case mixed_split mixed_workspace split
run_case mixed_fused mixed_workspace fused

compare_pair fp64_split_vs_fused fp64_split fp64_fused "$FIELD_ATOL"
compare_pair mixed_split_vs_fused mixed_split mixed_fused "$MIXED_FIELD_ATOL"

echo "TGV upwind flux-pair comparison passed: $OUT_DIR"
