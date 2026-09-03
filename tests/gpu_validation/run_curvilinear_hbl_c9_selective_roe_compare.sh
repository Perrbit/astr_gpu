#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_c9_selective_roe}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi
IM="${IM:-64}"
JM="${JM:-64}"
KM="${KM:-8}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
MAXSTEP="${MAXSTEP:-1}"
WARP_X="${GRID_WARP_X:-0.4}"
WARP_Y="${GRID_WARP_Y:-0.2}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-5.191440547760865}"
COMPARE_SENSOR="${COMPARE_SENSOR:-t}"
CPU_GEOMETRY_DUMP=""
if [[ "$NP" == "1" ]]; then
  CPU_GEOMETRY_DUMP="$OUT_DIR/cpu_geometry.h5"
fi

OUT_DIR="$OUT_DIR" IM="$IM" JM="$JM" KM="$KM" NP="$NP" TOPOLOGY="$TOPOLOGY" \
MAXSTEP="$MAXSTEP" FEQCHKPT="$MAXSTEP" \
GRID_WARP_X="$WARP_X" GRID_WARP_Y="$WARP_Y" \
PROFILE_OBLIQUE_SHOCK=t PROFILE_PRESSURE_MODE=provided \
PROFILE_DENSITY_MODE=provided PROFILE_SHOCK_Y_MIN="${PROFILE_SHOCK_Y_MIN:-0.18}" \
SHOCK_X0="${SHOCK_X0:--1.0}" SHOCK_Y0="${SHOCK_Y0:-0.18}" \
LCHARDECOMP=t DIFFTERM=f SPONGE_IM=0 UPPER_BCTYPE=51 XMIN_BCTYPE=11 \
SAME_PHASE_FIELD=t COMPARE_SENSOR="$COMPARE_SENSOR" \
CPU_GEOMETRY_DUMP="$CPU_GEOMETRY_DUMP" \
FIELD_ATOL="${FIELD_ATOL:-1e-10}" FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
STATS_ATOL="${STATS_ATOL:-1e-10}" STATS_RTOL="${STATS_RTOL:-1e-10}" \
SENSOR_ATOL="${SENSOR_ATOL:-1e-10}" SENSOR_RTOL="${SENSOR_RTOL:-1e-10}" \
  "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh"

python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_bl_grid.py" \
  --grid "$OUT_DIR/cpu/datin/grid.flatplate.h5" --report "$OUT_DIR/grid_check.txt" \
  --warp-x "$WARP_X" --warp-y "$WARP_Y"

if [[ "$NP" == "1" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_bl_metrics.py" \
    --input "$CPU_GEOMETRY_DUMP" --report "$OUT_DIR/metric_check.txt" \
    --grid "$IM,$JM,$KM" --warp-x "$WARP_X" --warp-y "$WARP_Y"
fi

python3 "$ROOT_DIR/tests/gpu_validation/check_wall41_invariants.py" \
  --input "$OUT_DIR/cpu/outdat/rk_complete_snapshot.h5" \
  --report "$OUT_DIR/cpu_wall41_invariants.txt" --axis 1 --side lower \
  --wall-temperature "$WALL_TEMPERATURE" --mach 5.0 --atol "${WALL_ATOL:-1e-12}"
python3 "$ROOT_DIR/tests/gpu_validation/check_wall41_invariants.py" \
  --input "$OUT_DIR/gpu" --report "$OUT_DIR/gpu_wall41_invariants.txt" \
  --axis 1 --side lower --wall-temperature "$WALL_TEMPERATURE" \
  --mach 5.0 --atol "${WALL_ATOL:-1e-12}"
