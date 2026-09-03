#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Lid-Driven-Cavity"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/ldcavity_phaseia_compare}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-32,32,32}"
DELTAT="${DELTAT:-}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
COMPARE_STATS="${COMPARE_STATS:-t}"
COMPARE_FIELD="${COMPARE_FIELD:-t}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  local args=(
    "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py"
    --src-case "$CASE_DIR"
    --dst-case "$OUT_DIR/$target"
    --input-name input.ldcav2d
    --use-gpu "$use_gpu"
    --maxstep "$MAXSTEP"
    --feqchkpt "$FEQCHKPT"
    --lfilter "$LFILTER"
    --diffterm "$DIFFTERM"
    --scheme "$SCHEME"
    --grid "$GRID"
  )
  if [[ -n "$DELTAT" ]]; then
    args+=(--deltat "$DELTAT")
  fi
  python3 "${args[@]}"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.ldcav2d > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.ldcav2d > gpu.log 2>&1
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
    --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" \
    --gpu "$OUT_DIR/gpu" \
    --report "$OUT_DIR/flowfield_compare.txt" \
    --atol "$FIELD_ATOL" \
    --rtol "$FIELD_RTOL"
fi
