#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/Rayleigh–Taylor-Instability}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/rti_phasej_compare}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-32,64,32}"
DELTAT="${DELTAT:-1.95d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
COMPARE_STATS="${COMPARE_STATS:-t}"
COMPARE_FIELD="${COMPARE_FIELD:-t}"
EXPECT_GPU_FAIL="${EXPECT_GPU_FAIL:-f}"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$OUT_DIR/$target" \
    --input-name input.rti \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" \
    --lfilter "$LFILTER" \
    --diffterm "$DIFFTERM" \
    --scheme "$SCHEME" \
    --grid "$GRID" \
    --deltat "$DELTAT"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.rti > cpu.log 2>&1
)

set +e
(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.rti > gpu.log 2>&1
)
gpu_status=$?
set -e

if [[ "$EXPECT_GPU_FAIL" == "t" ]]; then
  if [[ "$gpu_status" -eq 0 ]]; then
    printf 'expected GPU run to fail, but it completed\n' >&2
    exit 1
  fi
  printf 'GPU failed as expected with status %s. See %s/gpu/gpu.log\n' "$gpu_status" "$OUT_DIR"
  exit 0
fi

if [[ "$gpu_status" -ne 0 ]]; then
  printf 'GPU run failed with status %s. See %s/gpu/gpu.log\n' "$gpu_status" "$OUT_DIR" >&2
  exit "$gpu_status"
fi

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
