#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_uniform_np1}"
GRID="${GRID:-32,24,32}"
AMPLITUDE="${AMPLITUDE:-0.15}"
MAXSTEP="${MAXSTEP:-10}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
DELTAT="${DELTAT:-1e-4}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"
MACH="${MACH:-0.3}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"
IFS=, read -r IM JM KM <<< "$GRID"

prepare_case() {
  local mode="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$mode" --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" --conschm 643e \
    --diffterm f --lfilter f --upper-bctype 52 --ninit 3 \
    --uniform-profile --wall-temperature 1.0 --mach "$MACH" \
    --maxstep "$MAXSTEP" --feqchkpt "$MAXSTEP" --deltat "$DELTAT"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$OUT_DIR/$mode/datin/grid.flatplate.h5" \
    --report "$OUT_DIR/$mode/grid_generation.txt" --grid "$GRID" \
    --mapping y-wavy --amplitude "$AMPLITUDE"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$OUT_DIR/$mode/datin/flowini3d.h5" --grid "$GRID" \
    --u1 0.0 --u2 0.0 --u3 0.0 --density 1.0 --temperature 1.0
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t
python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_nscbc_geometry.py" \
  --grid "$OUT_DIR/cpu/datin/grid.flatplate.h5" \
  --report "$OUT_DIR/geometry.txt" --amplitude "$AMPLITUDE"

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
  ASTR_NSCBC_FARFIELD_MODE=nonreflecting \
  ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.flatplate > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
  ASTR_NSCBC_FARFIELD_MODE=nonreflecting \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.flatplate > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" --atol "$ATOL" --rtol "$RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/check_uniform_flowfield.py" \
  --input "$OUT_DIR/cpu/$CPU_SNAPSHOT" --report "$OUT_DIR/cpu_drift.txt" \
  --u1 0.0 --u2 0.0 --u3 0.0 --mach "$MACH" --upper-planes 2 \
  --atol "$ATOL" --rtol "$RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/check_uniform_flowfield.py" \
  --input "$OUT_DIR/gpu" --report "$OUT_DIR/gpu_drift.txt" \
  --u1 0.0 --u2 0.0 --u3 0.0 --mach "$MACH" --upper-planes 2 \
  --atol "$ATOL" --rtol "$RTOL"
