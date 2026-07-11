#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_oblique_shock}"
IM="${IM:-192}"
JM="${JM:-192}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-2}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
REYNOLDS="${REYNOLDS:-1.83052e6}"
MACH="${MACH:-5.0}"
REFERENCE_TEMPERATURE="${REFERENCE_TEMPERATURE:-226.65}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-5.191440547760865}"
STATION_X="${STATION_X:-1.0}"
VIRTUAL_LEADING_EDGE="${VIRTUAL_LEADING_EDGE:--2.0}"
PROFILE_DENSITY_MODE="${PROFILE_DENSITY_MODE:-provided}"
PROFILE_OBLIQUE_SHOCK="${PROFILE_OBLIQUE_SHOCK:-f}"
if [[ "$PROFILE_OBLIQUE_SHOCK" == "t" ]]; then
  PROFILE_PRESSURE_MODE="${PROFILE_PRESSURE_MODE:-provided}"
else
  PROFILE_PRESSURE_MODE="${PROFILE_PRESSURE_MODE:-reconstruct}"
fi
UPPER_BCTYPE="${UPPER_BCTYPE:-51}"
SHOCK_ANGLE_DEG="${SHOCK_ANGLE_DEG:-35.0}"
SHOCK_X0="${SHOCK_X0:-2.0}"
SHOCK_Y0="${SHOCK_Y0:-0.18}"
SHOCK_Y_MIN="${SHOCK_Y_MIN:-0.02}"
PROFILE_SHOCK_Y_MIN="${PROFILE_SHOCK_Y_MIN:-$SHOCK_Y0}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" \
    --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" \
    --conschm 543e --reynolds "$REYNOLDS" --mach "$MACH" \
    --reference-temperature "$REFERENCE_TEMPERATURE" \
    --wall-temperature "$WALL_TEMPERATURE" --upper-bctype "$UPPER_BCTYPE" --ninit 3 \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT"
}

write_similarity_shock_field() {
  local target="$1"
  local profile_shock_args=()
  if [[ "$PROFILE_OBLIQUE_SHOCK" == "t" ]]; then
    profile_shock_args+=(--profile-oblique-shock --profile-shock-y-min "$PROFILE_SHOCK_Y_MIN")
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$OUT_DIR/$target/datin/grid.flatplate.h5" \
    --output "$OUT_DIR/$target/datin/inlet.prof" \
    --mach "$MACH" --reynolds "$REYNOLDS" \
    --reference-temperature "$REFERENCE_TEMPERATURE" \
    --wall-temperature "$WALL_TEMPERATURE" --station-x "$STATION_X" \
    --density-mode "$PROFILE_DENSITY_MODE" \
    --pressure-mode "$PROFILE_PRESSURE_MODE" \
    --field-output "$OUT_DIR/$target/datin/flowini3d.h5" \
    --virtual-leading-edge "$VIRTUAL_LEADING_EDGE" \
    --field-oblique-shock \
    --shock-angle-deg "$SHOCK_ANGLE_DEG" \
    --shock-x0 "$SHOCK_X0" \
    --shock-y0 "$SHOCK_Y0" \
    --shock-y-min "$SHOCK_Y_MIN" \
    "${profile_shock_args[@]}"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t
write_similarity_shock_field cpu
write_similarity_shock_field gpu

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.flatplate > cpu.log 2>&1
)
(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.flatplate > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" --atol "$STATS_ATOL" --rtol "$STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"
