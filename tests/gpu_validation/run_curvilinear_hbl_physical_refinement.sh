#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_physical_refinement}"
COARSE_DT="${COARSE_DT:-1e-5}"
FINE_DT="${FINE_DT:-5e-6}"
COARSE_STEPS="${COARSE_STEPS:-20}"
FINE_STEPS="${FINE_STEPS:-40}"
KM="${KM:-8}"
WARP_X="${GRID_WARP_X:-0.4}"
WARP_Y="${GRID_WARP_Y:-0.2}"
HOT_WALL_TEMPERATURE="${HOT_WALL_TEMPERATURE:-5.191440547760865}"
COLD_WALL_TEMPERATURE="${COLD_WALL_TEMPERATURE:-3.0}"
PROFILE_POINTS="${PROFILE_POINTS:-8001}"

python3 - "$COARSE_DT" "$COARSE_STEPS" "$FINE_DT" "$FINE_STEPS" <<'PY'
import math
import sys

coarse_dt, coarse_steps, fine_dt, fine_steps = map(float, sys.argv[1:])
if not math.isclose(coarse_dt * coarse_steps, fine_dt * fine_steps, rel_tol=1e-13, abs_tol=1e-15):
    raise SystemExit("coarse and fine time-step runs must end at the same physical time")
PY

mkdir -p "$OUT_DIR"

run_case() {
  local wall_label="$1"
  local wall_temperature="$2"
  local grid_label="$3"
  local size="$4"
  local dt_label="$5"
  local dt="$6"
  local steps="$7"
  local case_dir="$OUT_DIR/${wall_label}/${grid_label}_${dt_label}"

  OUT_DIR="$case_dir" IM="$size" JM="$size" KM="$KM" NP=1 TOPOLOGY=1,1,1 \
    MAXSTEP="$steps" DELTAT="$dt" GRID_WARP_X="$WARP_X" GRID_WARP_Y="$WARP_Y" \
    WALL_TEMPERATURE="$wall_temperature" PROFILE_POINTS="$PROFILE_POINTS" \
    "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c7_compare.sh"

  python3 "$ROOT_DIR/tests/gpu_validation/analyze_curvilinear_hbl_physics.py" \
    --field "$case_dir/cpu/outdat/rk_complete_snapshot.h5" \
    --grid "$case_dir/cpu/datin/grid.flatplate.h5" \
    --report "$case_dir/hbl_physics.txt" \
    --npz "$case_dir/hbl_physics.npz" \
    --wall-csv "$case_dir/wall_diagnostics.csv" \
    --profile-dir "$case_dir/profiles" \
    --wall-temperature "$wall_temperature"
}

run_wall_family() {
  local wall_label="$1"
  local wall_temperature="$2"
  local required_wall_quantity="$3"
  local wall_dir="$OUT_DIR/$wall_label"

  run_case "$wall_label" "$wall_temperature" coarse 96 coarse_dt "$COARSE_DT" "$COARSE_STEPS"
  run_case "$wall_label" "$wall_temperature" coarse 96 fine_dt "$FINE_DT" "$FINE_STEPS"
  run_case "$wall_label" "$wall_temperature" medium 144 coarse_dt "$COARSE_DT" "$COARSE_STEPS"
  run_case "$wall_label" "$wall_temperature" medium 144 fine_dt "$FINE_DT" "$FINE_STEPS"
  run_case "$wall_label" "$wall_temperature" fine 192 coarse_dt "$COARSE_DT" "$COARSE_STEPS"
  run_case "$wall_label" "$wall_temperature" fine 192 fine_dt "$FINE_DT" "$FINE_STEPS"

  python3 "$ROOT_DIR/tests/gpu_validation/summarize_curvilinear_hbl_refinement.py" \
    --coarse-dt "coarse=$wall_dir/coarse_coarse_dt/hbl_physics.npz" \
    --coarse-dt "medium=$wall_dir/medium_coarse_dt/hbl_physics.npz" \
    --coarse-dt "fine=$wall_dir/fine_coarse_dt/hbl_physics.npz" \
    --fine-dt "coarse=$wall_dir/coarse_fine_dt/hbl_physics.npz" \
    --fine-dt "medium=$wall_dir/medium_fine_dt/hbl_physics.npz" \
    --fine-dt "fine=$wall_dir/fine_fine_dt/hbl_physics.npz" \
    --required-wall-quantity cf_geometric \
    $required_wall_quantity \
    --report "$wall_dir/refinement_summary.txt"
}

run_wall_family hot "$HOT_WALL_TEMPERATURE" ""
run_wall_family cold "$COLD_WALL_TEMPERATURE" "--required-wall-quantity qw_geometric"

cat > "$OUT_DIR/validation_scope.txt" <<EOF
hot_wall_temperature=$HOT_WALL_TEMPERATURE
hot_wall_acceptance=Cf,u,T; qw is informational because the wall is near recovery temperature
cold_wall_temperature=$COLD_WALL_TEMPERATURE
cold_wall_acceptance=Cf,qw,u,T
coarse_dt=$COARSE_DT coarse_steps=$COARSE_STEPS
fine_dt=$FINE_DT fine_steps=$FINE_STEPS
profile_points=$PROFILE_POINTS
EOF
