#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_256_performance_profile}"
PROFILE_TOOL="${PROFILE_TOOL:-nsys}"
GRID="${GRID:-256,256,256}"
MAXSTEP="${MAXSTEP:-2}"
FEQCHKPT="${FEQCHKPT:-9999}"
GPU_ID="${GPU_ID:-0}"
NSYS_TRACE="${NSYS_TRACE:-cuda,mpi}"
NCU_SET="${NCU_SET:-full}"
NCU_KERNEL="${NCU_KERNEL:-solver_gpu_diffusion_flux_global_kernel_}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"

if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
if [[ "$FEQCHKPT" -le "$MAXSTEP" ]]; then
  echo "FEQCHKPT must exceed MAXSTEP so the profiled loop does not write fields" >&2
  exit 2
fi

case "$PROFILE_TOOL" in
  nsys)
    command -v nsys >/dev/null 2>&1 || { echo "nsys not found" >&2; exit 127; }
    ;;
  ncu)
    command -v ncu >/dev/null 2>&1 || { echo "ncu not found" >&2; exit 127; }
    ;;
  *)
    echo "PROFILE_TOOL must be nsys or ncu" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
CASE_DIR="$OUT_DIR/${PROFILE_TOOL}_case"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
  --dst-case "$CASE_DIR" --use-gpu t --grid "$GRID" \
  --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" \
  --lfilter t --diffterm t --scheme 643e

if [[ "$PROFILE_TOOL" == nsys ]]; then
  (
    cd "$CASE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
      nsys profile --trace="$NSYS_TRACE" --stats=true --force-overwrite=true \
        -o ../tgv_256_nsys \
        mpirun -np 1 "$GPU_EXE" run datin/input.tgv > ../nsys.log 2>&1
  )
  python3 "$ROOT_DIR/tests/gpu_validation/analyze_nsys_rk_residency.py" \
    --input "$OUT_DIR/tgv_256_nsys.sqlite" \
    --report "$OUT_DIR/rk_residency.txt" \
    --start-kernel filter_x_global_kernel \
    --large-transfer-bytes 65536 \
    --allow-d2h-after-kernel statistic_gpu_kenergy_partial_kernel \
    --allow-d2h-after-kernel statistic_gpu_enstophy_partial_kernel \
    --allow-d2h-after-kernel statistic_gpu_dissipation_partial_kernel
else
  (
    cd "$CASE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
      ncu --target-processes all --set "$NCU_SET" \
        --kernel-name "regex:${NCU_KERNEL}" \
        --launch-skip "$NCU_LAUNCH_SKIP" --launch-count 1 \
        --force-overwrite --export ../diffusion_flux_full \
        mpirun -np 1 "$GPU_EXE" run datin/input.tgv > ../ncu.log 2>&1
  )
fi
