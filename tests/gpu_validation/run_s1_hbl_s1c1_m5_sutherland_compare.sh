#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s1_hbl_s1c1_m5_sutherland}"
IM="${IM:-192}"
JM="${JM:-192}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-2}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
DELTAT="${DELTAT:-1e-5}"
REYNOLDS="${REYNOLDS:-1.83052e6}"
MACH="${MACH:-5.0}"
REFERENCE_TEMPERATURE="${REFERENCE_TEMPERATURE:-226.65}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-5.191440547760865}"
STATION_X="${STATION_X:-1.0}"
PROFILE_DENSITY_MODE="${PROFILE_DENSITY_MODE:-provided}"
PROFILE_POINTS="${PROFILE_POINTS:-801}"
INITIALIZATION_MODE="${INITIALIZATION_MODE:-profile}"
UPPER_BCTYPE="${UPPER_BCTYPE:-51}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
GRID_WARP_X="${GRID_WARP_X:-0.0}"
GRID_WARP_Y="${GRID_WARP_Y:-0.0}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-t}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" \
    --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" \
    --conschm 543e --diffterm "$DIFFTERM" --lfilter "$LFILTER" \
    --reynolds "$REYNOLDS" --mach "$MACH" \
    --reference-temperature "$REFERENCE_TEMPERATURE" \
    --wall-temperature "$WALL_TEMPERATURE" --upper-bctype "$UPPER_BCTYPE" --ninit "$NINIT" \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --warp-x "$GRID_WARP_X" --warp-y "$GRID_WARP_Y" \
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" --deltat "$DELTAT"
}

write_similarity_profile() {
  local target="$1"
  local field_args=()
  if [[ "$INITIALIZATION_MODE" == "field" ]]; then
    field_args=(--field-output "$OUT_DIR/$target/datin/flowini3d.h5" --virtual-leading-edge "$VIRTUAL_LEADING_EDGE")
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$OUT_DIR/$target/datin/grid.flatplate.h5" \
    --output "$OUT_DIR/$target/datin/inlet.prof" \
    --mach "$MACH" --reynolds "$REYNOLDS" \
    --reference-temperature "$REFERENCE_TEMPERATURE" \
    --wall-temperature "$WALL_TEMPERATURE" --station-x "$STATION_X" \
    --density-mode "$PROFILE_DENSITY_MODE" --points "$PROFILE_POINTS" "${field_args[@]}"
}

case "$INITIALIZATION_MODE" in
  profile)
    NINIT=0
    ;;
  field)
    NINIT=3
    VIRTUAL_LEADING_EDGE="${VIRTUAL_LEADING_EDGE:--2.0}"
    ;;
  *)
    echo "INITIALIZATION_MODE must be profile or field" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t
write_similarity_profile cpu
write_similarity_profile gpu

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_GEOMETRY_DUMP="${CPU_GEOMETRY_DUMP:-}" \
    ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
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
  --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"
