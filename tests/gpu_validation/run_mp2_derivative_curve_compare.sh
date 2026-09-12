#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_derivative_curve_$STAMP}"
CANDIDATE="${MP2_CANDIDATE:-derivative}"
CANDIDATE_TARGET="gpu_${CANDIDATE}"
[[ "$CANDIDATE" == "derivative" || "$CANDIDATE" == "viscous_flux" ]] || {
  echo "MP2_CANDIDATE must be derivative or viscous_flux" >&2; exit 2;
}
GRID="${GRID:-32,32,16}"
MAXSTEP="${MAXSTEP:-2}"
DELTAT="${DELTAT:-1e-5}"
AMPLITUDE="${AMPLITUDE:-0.08}"
FIELD_ATOL="${FIELD_ATOL:-0}"
STATS_ATOL="${STATS_ATOL:-1e-8}"
IFS=, read -r IM JM KM <<< "$GRID"

[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite CURVE evidence directory: $OUT_DIR" >&2; exit 2; }

prepare_curve_case() {
  local family="$1" target="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$family/$target" --use-gpu t \
    --im "$IM" --jm "$JM" --km "$KM" --conschm 643e \
    --diffterm t --lfilter f --upper-bctype 52 --ninit 3 \
    --uniform-profile --wall-temperature 1.0 --mach 0.3 \
    --maxstep "$MAXSTEP" --feqchkpt "$MAXSTEP" --deltat "$DELTAT"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$OUT_DIR/$family/$target/datin/grid.flatplate.h5" \
    --report "$OUT_DIR/$family/$target/grid_generation.txt" --grid "$GRID" \
    --mapping y-wavy --amplitude "$AMPLITUDE"
  if [[ "$family" == "uniform" ]]; then
    python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
      --output "$OUT_DIR/$family/$target/datin/flowini3d.h5" --grid "$GRID" \
      --u1 0.0 --u2 0.0 --u3 0.0 --density 1.0 --temperature 1.0
  else
    python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_acoustic_pulse.py" \
      --grid "$OUT_DIR/$family/$target/datin/grid.flatplate.h5" \
      --output "$OUT_DIR/$family/$target/datin/flowini3d.h5" \
      --rho0 1.0 --p0 7.936507936507936 --gamma 1.4 --mach 0.3 \
      --amplitude 1e-5 --center 4.0,0.5,0.125 --width 0.08 \
      --direction 0.0,1.0,0.0 --clearance-widths 2.5 \
      --profile plane-y-compact --minimum-support-intervals 0
  fi
}

run_curve_case() {
  local family="$1" target="$2" mode="$3" candidate="$4"
  (
    cd "$OUT_DIR/$family/$target"
    env OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
      OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
      ASTR_FORCE_MPI_TOPOLOGY=1,1,1 ASTR_NSCBC_FARFIELD_MODE=nonreflecting \
      ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_PRECISION_MODE="$mode" \
      ASTR_GPU_MIXED_CANDIDATE="$candidate" \
      mpirun -np 1 "$GPU_EXE" run datin/input.flatplate > run.log 2>&1
  )
}

validate_curve_pair() {
  local family="$1"
  rg -q "^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=${CANDIDATE}$" \
    "$OUT_DIR/$family/$CANDIDATE_TARGET/run.log"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$OUT_DIR/$family/gpu_fp64" --gpu "$OUT_DIR/$family/$CANDIDATE_TARGET" \
    --report "$OUT_DIR/$family/fp64_vs_${CANDIDATE}_flowfield.txt" \
    --atol "$FIELD_ATOL" --rtol 0
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
    --cpu "$OUT_DIR/$family/gpu_fp64" --gpu "$OUT_DIR/$family/$CANDIDATE_TARGET" \
    --report "$OUT_DIR/$family/fp64_vs_${CANDIDATE}_flowstate.txt" \
    --atol "$STATS_ATOL" --rtol 0
}

mkdir -p "$OUT_DIR"
for family in uniform acoustic; do
  prepare_curve_case "$family" gpu_fp64
  prepare_curve_case "$family" "$CANDIDATE_TARGET"
  run_curve_case "$family" gpu_fp64 fp64 flux
  run_curve_case "$family" "$CANDIDATE_TARGET" mixed_workspace "$CANDIDATE"
  validate_curve_pair "$family"
done

OUT_DIR="$OUT_DIR/viscous_hbl" IM="$IM" JM="$JM" KM="$KM" \
  MAXSTEP="$MAXSTEP" DELTAT="$DELTAT" GRID_WARP_X="$AMPLITUDE" \
  GRID_WARP_Y="$(python3 -c "print(float('$AMPLITUDE')/2.0)")" \
  GPU_EXE="$GPU_EXE" FIELD_ATOL="$FIELD_ATOL" STATS_ATOL="$STATS_ATOL" \
  DIAGNOSTIC_ATOL="${DIAGNOSTIC_ATOL:-0}" DIAGNOSTIC_RTOL="${DIAGNOSTIC_RTOL:-0}" \
  MP2_CANDIDATE="$CANDIDATE" \
  "$ROOT_DIR/tests/gpu_validation/run_mp2_derivative_hbl_compare.sh"

echo "MP2 $CANDIDATE CURVE-C23 comparison passed: $OUT_DIR"
