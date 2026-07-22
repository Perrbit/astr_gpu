#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_tgv_compare}"
GRID="${GRID:-32,32,32}"
AMPLITUDE="${AMPLITUDE:-0.15}"
MAXSTEP="${MAXSTEP:-10}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-}"
if [[ -n "${FEQCHKPT+x}" ]]; then
  FEQCHKPT="$FEQCHKPT"
elif [[ "$MAXSTEP" == "0" ]]; then
  FEQCHKPT=1
else
  FEQCHKPT="$MAXSTEP"
fi
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
DELTAT="${DELTAT:-}"
STATS_ATOL="${STATS_ATOL:-1e-9}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
COMPARE_FIELD="${COMPARE_FIELD:-t}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

mkdir -p "$OUT_DIR"

for mode in cpu gpu; do
  if [[ "$mode" == cpu ]]; then
    use_gpu=f
  else
    use_gpu=t
  fi
  prepare_args=(
    --src-case "$CASE_DIR"
    --dst-case "$OUT_DIR/$mode"
    --use-gpu "$use_gpu"
    --maxstep "$MAXSTEP"
    --feqchkpt "$FEQCHKPT"
    --lfilter "$LFILTER"
    --diffterm "$DIFFTERM"
    --lreadgrid t
    --gridfile datin/grid.curvilinear.h5
    --grid "$GRID"
    --scheme 643e
  )
  if [[ -n "$DELTAT" ]]; then
    prepare_args+=(--deltat "$DELTAT")
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    "${prepare_args[@]}"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$OUT_DIR/$mode/datin/grid.curvilinear.h5" \
    --report "$OUT_DIR/$mode/grid_quality.txt" \
    --grid "$GRID" \
    --amplitude "$AMPLITUDE"
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

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" \
  --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" \
  --atol "$STATS_ATOL" \
  --rtol "$STATS_RTOL"

if [[ "$COMPARE_FIELD" == "t" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" \
    --gpu "$OUT_DIR/gpu" \
    --report "$OUT_DIR/flowfield_compare.txt" \
    --atol "$FIELD_ATOL" \
    --rtol "$FIELD_RTOL"
elif [[ "$COMPARE_FIELD" != "f" ]]; then
  echo "COMPARE_FIELD must be t or f" >&2
  exit 2
fi
