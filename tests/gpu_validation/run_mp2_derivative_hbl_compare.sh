#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_derivative_hbl_$STAMP}"
CANDIDATE="${MP2_CANDIDATE:-derivative}"
CANDIDATE_TARGET="gpu_${CANDIDATE}"
[[ "$CANDIDATE" == "derivative" || "$CANDIDATE" == "viscous_flux" ]] || {
  echo "MP2_CANDIDATE must be derivative or viscous_flux" >&2; exit 2;
}
IM="${IM:-64}"
JM="${JM:-64}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-2}"
DELTAT="${DELTAT:-1e-5}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
WARP_X="${GRID_WARP_X:-0.0}"
WARP_Y="${GRID_WARP_Y:-0.0}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-5.191440547760865}"
FIELD_ATOL="${FIELD_ATOL:-0}"
FIELD_RTOL="${FIELD_RTOL:-0}"
STATS_ATOL="${STATS_ATOL:-1e-8}"
STATS_RTOL="${STATS_RTOL:-0}"
DIAGNOSTIC_ATOL="${DIAGNOSTIC_ATOL:-0}"
DIAGNOSTIC_RTOL="${DIAGNOSTIC_RTOL:-0}"

[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite HBL evidence directory: $OUT_DIR" >&2; exit 2; }

prepare_case() {
  local target="$1"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" --use-gpu t \
    --im "$IM" --jm "$JM" --km "$KM" --conschm 643e \
    --diffterm t --lfilter f --ninit 3 --upper-bctype 51 \
    --reynolds 1.83052e6 --mach 5.0 --reference-temperature 226.65 \
    --wall-temperature "$WALL_TEMPERATURE" \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --warp-x "$WARP_X" --warp-y "$WARP_Y" \
    --maxstep "$MAXSTEP" --feqchkpt "$MAXSTEP" --deltat "$DELTAT"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$OUT_DIR/$target/datin/grid.flatplate.h5" \
    --output "$OUT_DIR/$target/datin/inlet.prof" \
    --field-output "$OUT_DIR/$target/datin/flowini3d.h5" \
    --mach 5.0 --reynolds 1.83052e6 --reference-temperature 226.65 \
    --wall-temperature "$WALL_TEMPERATURE" --station-x 1.0 \
    --virtual-leading-edge -2.0 --density-mode provided --points 801
}

run_case() {
  local target="$1" mode="$2" candidate="$3"
  (
    cd "$OUT_DIR/$target"
    env OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
      OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_PRECISION_MODE="$mode" ASTR_GPU_MIXED_CANDIDATE="$candidate" \
      mpirun -np "$NP" "$GPU_EXE" run datin/input.flatplate > run.log 2>&1
  )
}

mkdir -p "$OUT_DIR"
prepare_case gpu_fp64
prepare_case "$CANDIDATE_TARGET"
run_case gpu_fp64 fp64 flux
run_case "$CANDIDATE_TARGET" mixed_workspace "$CANDIDATE"

rg -q '^ASTR_GPU_PRECISION_MODE=fp64$' "$OUT_DIR/gpu_fp64/run.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=none$' "$OUT_DIR/gpu_fp64/run.log"
rg -q '^ASTR_GPU_PRECISION_MODE=mixed_workspace$' "$OUT_DIR/$CANDIDATE_TARGET/run.log"
rg -q "^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=${CANDIDATE}$" "$OUT_DIR/$CANDIDATE_TARGET/run.log"
for target in gpu_fp64 "$CANDIDATE_TARGET"; do
  head -1 "$OUT_DIR/$target/flowstate.dat" | rg -q 'fbcx.*wallheatflux'
done

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/$CANDIDATE_TARGET" \
  --report "$OUT_DIR/fp64_vs_${CANDIDATE}_flowstate.txt" \
  --atol "$STATS_ATOL" --rtol "$STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/$CANDIDATE_TARGET" \
  --report "$OUT_DIR/fp64_vs_${CANDIDATE}_flowfield.txt" \
  --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"

for target in gpu_fp64 "$CANDIDATE_TARGET"; do
  python3 "$ROOT_DIR/tests/gpu_validation/analyze_curvilinear_hbl_physics.py" \
    --field "$OUT_DIR/$target" \
    --grid "$OUT_DIR/$target/datin/grid.flatplate.h5" \
    --report "$OUT_DIR/$target/hbl_physics.txt" \
    --npz "$OUT_DIR/$target/hbl_physics.npz" \
    --wall-csv "$OUT_DIR/$target/wall_diagnostics.csv" \
    --profile-dir "$OUT_DIR/$target/profiles" \
    --wall-temperature "$WALL_TEMPERATURE"
done
python3 "$ROOT_DIR/tests/gpu_validation/compare_hbl_diagnostics.py" \
  --reference "$OUT_DIR/gpu_fp64/hbl_physics.npz" \
  --candidate "$OUT_DIR/$CANDIDATE_TARGET/hbl_physics.npz" \
  --report "$OUT_DIR/fp64_vs_${CANDIDATE}_hbl_diagnostics.txt" \
  --atol "$DIAGNOSTIC_ATOL" --rtol "$DIAGNOSTIC_RTOL"

echo "MP2 $CANDIDATE HBL comparison passed: $OUT_DIR"
