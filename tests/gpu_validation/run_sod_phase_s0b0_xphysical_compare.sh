#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/sod}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/sod_phase_s0b0_xphysical}"
MAXSTEP="${MAXSTEP:-3}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
GRID="${GRID:-200,8,8}"
DELTAT="${DELTAT:-5.d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"

IFS=',' read -r GRID_I GRID_J GRID_K <<< "$GRID"
if [[ -z "${GRID_I:-}" || -z "${GRID_J:-}" || -z "${GRID_K:-}" ]] \
  || ! [[ "$GRID_I" =~ ^[0-9]+$ && "$GRID_J" =~ ^[0-9]+$ && "$GRID_K" =~ ^[0-9]+$ ]]; then
  echo "S0-B0 GRID must contain three comma-separated non-negative integers" >&2
  exit 2
fi
if (( GRID_I < 5 || GRID_J < 5 || GRID_K < 5 )); then
  echo "S0-B0 requires every 3-D grid dimension >= hm=5; got GRID=$GRID" >&2
  exit 2
fi

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$OUT_DIR/$target" \
    --input-name input.sod \
    --flowtype sod \
    --homogeneous f,t,t \
    --bctype 50,50,1,1,1,1 \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" \
    --lfilter f \
    --diffterm f \
    --scheme 643e \
    --conschm 543e \
    --difschm 643e \
    --recon-schem 3 \
    --lchardecomp f \
    --grid "$GRID" \
    --deltat "$DELTAT"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.sod > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.sod > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" \
  --atol "$STATS_ATOL" \
  --rtol "$STATS_RTOL"

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" \
  --atol "$FIELD_ATOL" \
  --rtol "$FIELD_RTOL"
