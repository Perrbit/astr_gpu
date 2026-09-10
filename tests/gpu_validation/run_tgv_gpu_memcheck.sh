#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_gpu_memcheck}"
GRID="${GRID:-32,32,32}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-9999}"
GPU_ID="${GPU_ID:-0}"
SYNC_MODE="${SYNC_MODE:-explicit}"
FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"
CASE_DIR="$OUT_DIR/case"

if [[ "$GPU_EXE" != /* ]]; then
  GPU_EXE="$ROOT_DIR/$GPU_EXE"
fi
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
  CASE_DIR="$OUT_DIR/case"
fi
if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
command -v compute-sanitizer >/dev/null 2>&1 || {
  echo "compute-sanitizer not found" >&2
  exit 127
}
if [[ "$FEQCHKPT" -le "$MAXSTEP" ]]; then
  echo "FEQCHKPT must exceed MAXSTEP so memcheck does not write checkpoints" >&2
  exit 2
fi
if [[ "$FILTER_WORKSPACE" != "full" && "$FILTER_WORKSPACE" != "scalar" ]]; then
  echo "FILTER_WORKSPACE must be full or scalar" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
  --dst-case "$CASE_DIR" --use-gpu t --grid "$GRID" \
  --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" \
  --lfilter t --diffterm t --scheme 643e

(
  cd "$CASE_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_GPU_SYNC_MODE="$SYNC_MODE" \
    ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
    OMPI_MCA_pml=ob1 OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
    mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode 99 \
      "$GPU_EXE" run datin/input.tgv > "$OUT_DIR/memcheck.log" 2>&1
)

grep -q 'The job is done!' "$OUT_DIR/memcheck.log"
grep -q 'ERROR SUMMARY: 0 errors' "$OUT_DIR/memcheck.log"
tail -20 "$OUT_DIR/memcheck.log"
