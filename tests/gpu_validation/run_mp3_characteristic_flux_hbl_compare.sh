#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp3_characteristic_flux_hbl_$STAMP}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"

IM="${IM:-192}"
JM="${JM:-192}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-2}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
CALIBRATE="${CALIBRATE:-f}"
SENSOR_ATOL="${SENSOR_ATOL:-1e-12}"
SENSOR_RTOL="${SENSOR_RTOL:-1e-12}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
CANDIDATE_SENSOR_ATOL="${CANDIDATE_SENSOR_ATOL:-1e-12}"
CANDIDATE_SENSOR_RTOL="${CANDIDATE_SENSOR_RTOL:-0}"
CANDIDATE_STATS_ATOL="${CANDIDATE_STATS_ATOL:-1e-10}"
CANDIDATE_STATS_RTOL="${CANDIDATE_STATS_RTOL:-0}"
CANDIDATE_FIELD_ATOL="${CANDIDATE_FIELD_ATOL:-1e-10}"
CANDIDATE_FIELD_RTOL="${CANDIDATE_FIELD_RTOL:-0}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

[[ -x "$CPU_EXE" ]] || { echo "CPU_EXE is not executable: $CPU_EXE" >&2; exit 2; }
[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite MP3 HBL evidence: $OUT_DIR" >&2; exit 2; }
[[ "$CALIBRATE" == t || "$CALIBRATE" == f ]] || { echo "CALIBRATE must be t or f" >&2; exit 2; }

if [[ "$CALIBRATE" == t ]]; then
  CANDIDATE_SENSOR_ATOL=1.0
  CANDIDATE_SENSOR_RTOL=0
  CANDIDATE_STATS_ATOL=1.0
  CANDIDATE_STATS_RTOL=0
  CANDIDATE_FIELD_ATOL=1.0
  CANDIDATE_FIELD_RTOL=0
fi

prepare_case() {
  local target="$1" use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" \
    --diffterm t --lchardecomp t --shock-threshold 0.001 \
    --conschm 543e --reynolds 1.83052e6 --mach 5.0 \
    --reference-temperature 226.65 \
    --wall-temperature 5.191440547760865 \
    --upper-bctype 51 --x-min-bctype 11 --ninit 3 \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" --sponge-im 0 \
    --deltat 1.0e-5

  python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$OUT_DIR/$target/datin/grid.flatplate.h5" \
    --output "$OUT_DIR/$target/datin/inlet.prof" \
    --mach 5.0 --reynolds 1.83052e6 \
    --reference-temperature 226.65 \
    --wall-temperature 5.191440547760865 --station-x 1.0 \
    --density-mode provided --pressure-mode provided \
    --field-output "$OUT_DIR/$target/datin/flowini3d.h5" \
    --virtual-leading-edge -2.0 --shock-angle-deg 35.0 \
    --shock-x0 -1.0 --shock-y0 0.18 --shock-y-min 0.02 \
    --profile-oblique-shock --profile-shock-y-min 0.18 \
    --field-oblique-shock
}

run_cpu() {
  (
    cd "$OUT_DIR/cpu"
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_SHOCK_SENSOR_DUMP="$OUT_DIR/cpu_shock_sensor.dat" \
    ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
    OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
      mpirun --oversubscribe -np "$NP" "$CPU_EXE" \
        run datin/input.flatplate > cpu.log 2>&1
  )
}

run_gpu() {
  local target="$1" precision="$2" candidate="$3"
  (
    cd "$OUT_DIR/$target"
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_SHOCK_SENSOR_DUMP="$OUT_DIR/${target}_shock_sensor.dat" \
    ASTR_GPU_PRECISION_MODE="$precision" \
    ASTR_GPU_MIXED_CANDIDATE="$candidate" \
    ASTR_GPU_SYNC_MODE=explicit \
    OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
      mpirun --oversubscribe -np "$NP" "$GPU_EXE" \
        run datin/input.flatplate > "$target.log" 2>&1
  )
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu_fp64 t
prepare_case gpu_characteristic_flux t
run_cpu
run_gpu gpu_fp64 fp64 flux
run_gpu gpu_characteristic_flux mixed_workspace characteristic_flux

for target in cpu gpu_fp64 gpu_characteristic_flux; do
  rg -q 'The job is done!' "$OUT_DIR/$target/$target.log"
  [[ -s "$OUT_DIR/$target/flowstate.dat" ]]
  [[ -f "$OUT_DIR/$target/outdat/flowfield.h5" ]]
done
[[ -f "$OUT_DIR/cpu/$CPU_SNAPSHOT" ]]

rg -q '^ASTR_GPU_PRECISION_MODE=fp64$' "$OUT_DIR/gpu_fp64/gpu_fp64.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=none$' "$OUT_DIR/gpu_fp64/gpu_fp64.log"
rg -q '^ASTR_GPU_PRECISION_MODE=mixed_workspace$' \
  "$OUT_DIR/gpu_characteristic_flux/gpu_characteristic_flux.log"
rg -q '^ASTR_GPU_MIXED_CANDIDATE=characteristic_flux$' \
  "$OUT_DIR/gpu_characteristic_flux/gpu_characteristic_flux.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux$' \
  "$OUT_DIR/gpu_characteristic_flux/gpu_characteristic_flux.log"

rankwise_args=()
(( NP > 1 )) && rankwise_args+=(--rankwise)

python3 "$ROOT_DIR/tests/gpu_validation/compare_shock_sensor.py" \
  --cpu "$OUT_DIR/cpu_shock_sensor.dat" \
  --gpu "$OUT_DIR/gpu_fp64_shock_sensor.dat" \
  --report "$OUT_DIR/cpu_vs_gpu_fp64_shock_sensor.txt" \
  --atol "$SENSOR_ATOL" --rtol "$SENSOR_RTOL" "${rankwise_args[@]}"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu_fp64" \
  --report "$OUT_DIR/cpu_vs_gpu_fp64_flowstate.txt" \
  --atol "$STATS_ATOL" --rtol "$STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" --gpu "$OUT_DIR/gpu_fp64" \
  --report "$OUT_DIR/cpu_vs_gpu_fp64_flowfield.txt" \
  --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"

python3 "$ROOT_DIR/tests/gpu_validation/compare_shock_sensor.py" \
  --cpu "$OUT_DIR/gpu_fp64_shock_sensor.dat" \
  --gpu "$OUT_DIR/gpu_characteristic_flux_shock_sensor.dat" \
  --report "$OUT_DIR/gpu_fp64_vs_candidate_shock_sensor.txt" \
  --atol "$CANDIDATE_SENSOR_ATOL" --rtol "$CANDIDATE_SENSOR_RTOL" \
  "${rankwise_args[@]}"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/gpu_characteristic_flux" \
  --report "$OUT_DIR/gpu_fp64_vs_candidate_flowstate.txt" \
  --atol "$CANDIDATE_STATS_ATOL" --rtol "$CANDIDATE_STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/gpu_characteristic_flux" \
  --report "$OUT_DIR/gpu_fp64_vs_candidate_flowfield.txt" \
  --atol "$CANDIDATE_FIELD_ATOL" --rtol "$CANDIDATE_FIELD_RTOL"

echo "MP3 characteristic_flux Cartesian viscous HBL comparison passed: $OUT_DIR"
