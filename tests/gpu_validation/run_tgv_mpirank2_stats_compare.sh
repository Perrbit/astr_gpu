#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_mpirank2_stats_compare}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-99}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"
MPI_NP="${MPI_NP:-${NP:-2}}"
TOPOLOGY="${TOPOLOGY:-2,1,1}"

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
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" mpirun -np "$MPI_NP" "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" mpirun -np "$MPI_NP" "$GPU_EXE" run datin/input.tgv > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" \
  --atol "$ATOL" \
  --rtol "$RTOL"
