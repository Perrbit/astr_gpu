#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s1_hbl_s1b0}"
IM="${IM:-96}"
JM="${JM:-96}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-2}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
REYNOLDS="${REYNOLDS:-100000}"
MACH="${MACH:-3.0}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-2.5093503198764615}"
PROFILE_DELTA="${PROFILE_DELTA:-0.08}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" \
    --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" \
    --conschm 543e --reynolds "$REYNOLDS" --mach "$MACH" \
    --wall-temperature "$WALL_TEMPERATURE" --profile-delta "$PROFILE_DELTA" \
    --x-min -1.0 --x-max 10.0 --y-stretch 3.0 --z-length 0.25 \
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.flatplate > cpu.log 2>&1
)
(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.flatplate > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" --atol "$STATS_ATOL" --rtol "$STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"
