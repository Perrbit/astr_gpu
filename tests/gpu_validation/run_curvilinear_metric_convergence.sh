#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_metric_convergence}"
SIZES="${SIZES:-16 24 32}"
AMPLITUDE="${AMPLITUDE:-0.15}"
IDENTITY_ATOL="${IDENTITY_ATOL:-1e-10}"
reports=()

mkdir -p "$OUT_DIR"

for size in $SIZES; do
  case_out="$OUT_DIR/n${size}"
  grid="${size},${size},${size}"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$case_out" \
    --use-gpu f \
    --maxstep 0 \
    --feqchkpt 1 \
    --lfilter f \
    --diffterm f \
    --lreadgrid t \
    --gridfile datin/grid.curvilinear.h5 \
    --grid "$grid" \
    --scheme 643e
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$case_out/datin/grid.curvilinear.h5" \
    --report "$case_out/grid_quality.txt" \
    --grid "$grid" \
    --amplitude "$AMPLITUDE"
  (
    cd "$case_out"
    ASTR_GEOMETRY_DUMP="$case_out/geometry.h5" \
      mpirun -np 1 "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
  )
  report="$case_out/metric_compare.txt"
  python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_metrics.py" \
    --input "$case_out/geometry.h5" \
    --report "$report" \
    --grid "$grid" \
    --amplitude "$AMPLITUDE" \
    --identity-atol "$IDENTITY_ATOL"
  reports+=("$report")
done

python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_metric_convergence.py" \
  --reports "${reports[@]}" \
  --output "$OUT_DIR/convergence_summary.txt"
