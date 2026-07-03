#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_field_compare}"
MAXSTEP="${MAXSTEP:-10}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"

mkdir -p "$OUT_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" \
  --dst-case "$OUT_DIR/cpu" \
  --use-gpu f \
  --maxstep "$MAXSTEP" \
  --feqchkpt "$FEQCHKPT" \
  --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" \
  --scheme "$SCHEME"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" \
  --dst-case "$OUT_DIR/gpu" \
  --use-gpu t \
  --maxstep "$MAXSTEP" \
  --feqchkpt "$FEQCHKPT" \
  --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" \
  --scheme "$SCHEME"

(
  cd "$OUT_DIR/cpu"
  mpirun -np 1 "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  mpirun -np 1 "$GPU_EXE" run datin/input.tgv > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" \
  --atol "$ATOL" \
  --rtol "$RTOL"
