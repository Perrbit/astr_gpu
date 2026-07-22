#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_freestream_compare}"
GRID="${GRID:-32,32,32}"
AMPLITUDE="${AMPLITUDE:-0.15}"
MAXSTEP="${MAXSTEP:-10}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-}"
DELTAT="${DELTAT:-1.d-4}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

mkdir -p "$OUT_DIR"

for mode in cpu gpu; do
  if [[ "$mode" == cpu ]]; then
    use_gpu=f
  else
    use_gpu=t
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$OUT_DIR/$mode" \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$MAXSTEP" \
    --lfilter f \
    --diffterm f \
    --lreadgrid t \
    --gridfile datin/grid.curvilinear.h5 \
    --grid "$GRID" \
    --scheme 643e \
    --ninit 3 \
    --deltat "$DELTAT"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$OUT_DIR/$mode/datin/grid.curvilinear.h5" \
    --report "$OUT_DIR/$mode/grid_quality.txt" \
    --grid "$GRID" \
    --amplitude "$AMPLITUDE"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$OUT_DIR/$mode/datin/flowini3d.h5" \
    --grid "$GRID"
done

(
  cd "$OUT_DIR/cpu"
  if [[ -n "$TOPOLOGY" ]]; then
    export ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY"
  fi
  ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  if [[ -n "$TOPOLOGY" ]]; then
    export ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY"
  fi
  mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" \
  --atol "$ATOL" \
  --rtol "$RTOL"

python3 "$ROOT_DIR/tests/gpu_validation/check_uniform_flowfield.py" \
  --input "$OUT_DIR/cpu/$CPU_SNAPSHOT" \
  --report "$OUT_DIR/cpu_freestream_drift.txt" \
  --atol "$ATOL" \
  --rtol "$RTOL"

python3 "$ROOT_DIR/tests/gpu_validation/check_uniform_flowfield.py" \
  --input "$OUT_DIR/gpu" \
  --report "$OUT_DIR/gpu_freestream_drift.txt" \
  --atol "$ATOL" \
  --rtol "$RTOL"
