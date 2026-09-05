#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_open_boundary_rejects}"
GRID="${GRID:-32,16,16}"
IFS=, read -r IM JM KM <<< "$GRID"

prepare_open_shock() {
  local name="$1"
  local bctype="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$ROOT_DIR/examples/Shuosher" --dst-case "$OUT_DIR/$name" \
    --input-name input.shuosher --flowtype openshock --homogeneous f,t,t \
    --bctype "$bctype" --use-gpu t --maxstep 1 --feqchkpt 1 \
    --lfilter f --diffterm f --lreadgrid t --gridfile ./datin/grid.curve.h5 \
    --scheme 643e --conschm 543e --difschm 643e --recon-schem 3 \
    --lchardecomp f --grid "$GRID" --deltat 1.d-4
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$OUT_DIR/$name/datin/grid.curve.h5" \
    --report "$OUT_DIR/$name/grid_report.txt" --grid "$GRID" \
    --mapping x-wavy --amplitude 0.15
}

expect_reject() {
  local name="$1"
  local input_name="$2"
  local pattern="$3"
  local status
  set +e
  (
    cd "$OUT_DIR/$name"
    ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
      mpirun -np 1 "$GPU_EXE" run "datin/$input_name" > gpu.log 2>&1
  )
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    printf 'unexpected GPU pass for %s\n' "$name" >&2
    return 1
  fi
  if ! grep -Fq "$pattern" "$OUT_DIR/$name/gpu.log"; then
    printf 'unexpected reject reason for %s; expected: %s\n' "$name" "$pattern" >&2
    return 1
  fi
  printf 'pass reject=%s status=%s pattern=%s\n' "$name" "$status" "$pattern"
}

mkdir -p "$OUT_DIR"

prepare_open_shock curve_21 '11,free;21,10.333333333333333;1;1;1;1'
expect_reject curve_21 input.shuosher \
  'GPU bctype=21 x-max requires an axis-aligned physical face'

prepare_open_shock curve_22 '12;22,10.333333333333333;1;1;1;1'
expect_reject curve_22 input.shuosher \
  'GPU CURVE bctype=22 is not validated on nonorthogonal grids'

python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
  --dst-case "$OUT_DIR/curve_51" --use-gpu t --im "$IM" --jm "$JM" --km "$KM" \
  --conschm 543e --diffterm t --lfilter f --ninit 0 --maxstep 1 --feqchkpt 1
python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
  --output "$OUT_DIR/curve_51/datin/grid.flatplate.h5" \
  --report "$OUT_DIR/curve_51/grid_report.txt" --grid "$GRID" \
  --mapping y-wavy --amplitude 0.15
expect_reject curve_51 input.flatplate \
  'GPU bctype=51 y-max requires an axis-aligned physical face'
