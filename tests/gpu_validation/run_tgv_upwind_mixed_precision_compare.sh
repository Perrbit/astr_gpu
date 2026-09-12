#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/Taylor_Green_Vortex}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_upwind_mixed_precision_compare}"
GRID="${GRID:-32,32,32}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
DELTAT="${DELTAT:-1.d-4}"
RECON_SCHEM="${RECON_SCHEM:-1}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
FP64_ATOL="${FP64_ATOL:-1e-10}"
FP64_RTOL="${FP64_RTOL:-1e-10}"
MIXED_FIELD_ATOL="${MIXED_FIELD_ATOL:-2e-6}"
MIXED_FIELD_RTOL="${MIXED_FIELD_RTOL:-0}"
MIXED_STATS_ATOL="${MIXED_STATS_ATOL:-5e-11}"
MIXED_STATS_RTOL="${MIXED_STATS_RTOL:-0}"

if [[ ! -x "$CPU_EXE" || ! -x "$GPU_EXE" ]]; then
  echo "CPU_EXE and GPU_EXE must be executable" >&2
  exit 2
fi
if [[ "$RECON_SCHEM" != "1" && "$RECON_SCHEM" != "3" ]]; then
  echo "RECON_SCHEM must be 1 (WENO7) or 3 (MP7)" >&2
  exit 2
fi

prepare_case() {
  local target="$1" use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$OUT_DIR/$target" \
    --input-name input.tgv \
    --flowtype tgv \
    --homogeneous t,t,t \
    --bctype 1,1,1,1,1,1 \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" \
    --lfilter f \
    --diffterm f \
    --scheme 643e \
    --conschm 543e \
    --difschm 643e \
    --recon-schem "$RECON_SCHEM" \
    --lchardecomp f \
    --grid "$GRID" \
    --deltat "$DELTAT"
}

run_case() {
  local target="$1"
  local exe="$GPU_EXE"
  shift
  if [[ "$target" == "cpu_fp64" ]]; then
    exe="$CPU_EXE"
  fi
  (
    cd "$OUT_DIR/$target"
    env OMPI_MCA_coll_hcoll_enable=0 \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" "$@" \
      mpirun -np "$NP" "$exe" \
      run datin/input.tgv > run.log 2>&1
  )
}

compare_pair() {
  local label="$1" reference="$2" candidate="$3"
  local stats_atol="$4" stats_rtol="$5" field_atol="$6" field_rtol="$7"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
    --cpu "$OUT_DIR/$reference" \
    --gpu "$OUT_DIR/$candidate" \
    --report "$OUT_DIR/${label}_flowstate.txt" \
    --atol "$stats_atol" \
    --rtol "$stats_rtol"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$OUT_DIR/$reference" \
    --gpu "$OUT_DIR/$candidate" \
    --report "$OUT_DIR/${label}_flowfield.txt" \
    --atol "$field_atol" \
    --rtol "$field_rtol"
}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
prepare_case cpu_fp64 f
prepare_case gpu_fp64 t
prepare_case gpu_mixed t

run_case cpu_fp64
run_case gpu_fp64 ASTR_GPU_PRECISION_MODE="fp64" ASTR_GPU_SYNC_MODE=explicit
run_case gpu_mixed ASTR_GPU_PRECISION_MODE="mixed_workspace" ASTR_GPU_SYNC_MODE=explicit

rg -q '^ASTR_GPU_PRECISION_MODE=fp64$' "$OUT_DIR/gpu_fp64/run.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=none$' "$OUT_DIR/gpu_fp64/run.log"
rg -q '^ASTR_GPU_PRECISION_MODE=mixed_workspace$' "$OUT_DIR/gpu_mixed/run.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=flux_work$' "$OUT_DIR/gpu_mixed/run.log"

compare_pair cpu_vs_gpu_fp64 cpu_fp64 gpu_fp64 \
  "$FP64_ATOL" "$FP64_RTOL" "$FP64_ATOL" "$FP64_RTOL"
compare_pair gpu_fp64_vs_mixed gpu_fp64 gpu_mixed \
  "$MIXED_STATS_ATOL" "$MIXED_STATS_RTOL" \
  "$MIXED_FIELD_ATOL" "$MIXED_FIELD_RTOL"
compare_pair cpu_vs_gpu_mixed cpu_fp64 gpu_mixed \
  "$MIXED_STATS_ATOL" "$MIXED_STATS_RTOL" \
  "$MIXED_FIELD_ATOL" "$MIXED_FIELD_RTOL"

{
  rg '^ASTR_GPU_(PRECISION_MODE|ACTIVE_MIXED_WORKSPACE|FLUX_WORKSPACE_BYTES)=' \
    "$OUT_DIR/gpu_fp64/run.log"
  rg '^ASTR_GPU_(PRECISION_MODE|ACTIVE_MIXED_WORKSPACE|FLUX_WORKSPACE_BYTES)=' \
    "$OUT_DIR/gpu_mixed/run.log"
} > "$OUT_DIR/workspace_summary.txt"

echo "TGV upwind mixed-precision comparison passed: $OUT_DIR"
