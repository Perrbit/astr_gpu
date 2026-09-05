#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/profile_inflow11_state}"
IM="${IM:-32}"
JM="${JM:-32}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-1}"
MACH="${MACH:-0.3}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" --mach "$MACH" \
    --conschm 543e --diffterm t --lfilter f --wall-temperature 1.4 \
    --isobaric-profile --ninit 3 --maxstep "$MAXSTEP" --feqchkpt "$MAXSTEP"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$OUT_DIR/$target/datin/flowini3d.h5" --grid "$IM,$JM,$KM" \
    --density 0.7 --u1 0.2 --u2 -0.1 --u3 0.05 --temperature 0.8
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY=1,1,1 ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
    mpirun -np 1 "$CPU_EXE" run datin/input.flatplate > cpu.log 2>&1
)
(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    mpirun -np 1 "$GPU_EXE" run datin/input.flatplate > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" --atol 1e-10 --rtol 1e-10
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" --atol 1e-10 --rtol 1e-10

for target in cpu gpu; do
  input="$OUT_DIR/$target"
  if [[ "$target" == "cpu" ]]; then
    input="$OUT_DIR/cpu/$CPU_SNAPSHOT"
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/check_profile_inflow11_invariants.py" \
    --input "$input" --profile "$OUT_DIR/$target/datin/inlet.prof" \
    --mach "$MACH" --report "$OUT_DIR/${target}_inflow11_invariants.txt"
done
