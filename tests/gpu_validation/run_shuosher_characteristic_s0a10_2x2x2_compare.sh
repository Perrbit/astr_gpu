#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/Shuosher}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/shuosher_characteristic_s0a10_2x2x2_compare}"
GRID="${GRID:-400,16,16}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
DELTAT="${DELTAT:-1.d-4}"
SENSOR_ATOL="${SENSOR_ATOL:-1e-12}"
SENSOR_RTOL="${SENSOR_RTOL:-1e-12}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"

prepare_case() {
  local target="$1" use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" --dst-case "$OUT_DIR/$target" \
    --input-name input.shuosher --flowtype shuosher --homogeneous t,t,t \
    --bctype 1,1,1,1,1,1 --use-gpu "$use_gpu" --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" --lfilter f --diffterm f --scheme 643e \
    --conschm 543e --difschm 643e --recon-schem 3 --lchardecomp t \
    --grid "$GRID" --deltat "$DELTAT"
}

run_case() {
  local target="$1" exe="$2"
  (
    cd "$OUT_DIR/$target"
    ASTR_FORCE_MPI_TOPOLOGY=2,2,2 \
    ASTR_SHOCK_SENSOR_DUMP="$OUT_DIR/${target}_shock_sensor.dat" \
      mpirun -np 8 "$exe" run datin/input.shuosher > "${target}.log" 2>&1
  )
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t
run_case cpu "$CPU_EXE"
run_case gpu "$GPU_EXE"

python3 "$ROOT_DIR/tests/gpu_validation/compare_shock_sensor.py" \
  --cpu "$OUT_DIR/cpu_shock_sensor.dat" --gpu "$OUT_DIR/gpu_shock_sensor.dat" \
  --report "$OUT_DIR/shock_sensor_compare.txt" --atol "$SENSOR_ATOL" --rtol "$SENSOR_RTOL" \
  --rankwise
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" --report "$OUT_DIR/flowstate_compare.txt" \
  --atol "$STATS_ATOL" --rtol "$STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" --report "$OUT_DIR/flowfield_compare.txt" \
  --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"
