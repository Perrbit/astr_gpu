#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/hit_phasea_compare}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-99}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-64,64,64}"
DELTAT="${DELTAT:-5.d-4}"
AMPLITUDE="${AMPLITUDE:-0.05}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-9}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-9}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
COMPARE_STATS="${COMPARE_STATS:-t}"
COMPARE_FIELD="${COMPARE_FIELD:-f}"

mkdir -p "$OUT_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" \
  --dst-case "$OUT_DIR/cpu" \
  --input-name input.tgv \
  --flowtype hit \
  --use-gpu f \
  --maxstep "$MAXSTEP" \
  --feqchkpt "$FEQCHKPT" \
  --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" \
  --scheme "$SCHEME" \
  --grid "$GRID" \
  --deltat "$DELTAT"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" \
  --dst-case "$OUT_DIR/gpu" \
  --input-name input.tgv \
  --flowtype hit \
  --use-gpu t \
  --maxstep "$MAXSTEP" \
  --feqchkpt "$FEQCHKPT" \
  --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" \
  --scheme "$SCHEME" \
  --grid "$GRID" \
  --deltat "$DELTAT"

python3 "$ROOT_DIR/tests/gpu_validation/generate_hit_velocity.py" \
  --output "$OUT_DIR/cpu/datin/velocity.h5" \
  --grid "$GRID" \
  --amplitude "$AMPLITUDE"
cp "$OUT_DIR/cpu/datin/velocity.h5" "$OUT_DIR/gpu/datin/velocity.h5"

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv > gpu.log 2>&1
)

if [[ "$COMPARE_STATS" == "t" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
    --cpu "$OUT_DIR/cpu" \
    --gpu "$OUT_DIR/gpu" \
    --report "$OUT_DIR/flowstate_compare.txt" \
    --atol "$STATS_ATOL" \
    --rtol "$STATS_RTOL"
fi

if [[ "$COMPARE_FIELD" == "t" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$OUT_DIR/cpu" \
    --gpu "$OUT_DIR/gpu" \
    --report "$OUT_DIR/flowfield_compare.txt" \
    --atol "$FIELD_ATOL" \
    --rtol "$FIELD_RTOL"
fi
