#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/Shuosher}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp3_characteristic_flux_$STAMP}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$PWD/$OUT_DIR"
fi
GRID="${GRID:-400,8,8}"
MAXSTEP="${MAXSTEP:-3}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
DELTAT="${DELTAT:-1.d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
RANKWISE="${RANKWISE:-f}"
SHOCK_X="${SHOCK_X:-}"
CALIBRATE="${CALIBRATE:-f}"
SENSOR_ATOL="${SENSOR_ATOL:-1e-12}"
SENSOR_RTOL="${SENSOR_RTOL:-1e-12}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
CANDIDATE_STATS_ATOL="${CANDIDATE_STATS_ATOL:-1e-6}"
CANDIDATE_STATS_RTOL="${CANDIDATE_STATS_RTOL:-0}"
CANDIDATE_FIELD_ATOL="${CANDIDATE_FIELD_ATOL:-1e-5}"
CANDIDATE_FIELD_RTOL="${CANDIDATE_FIELD_RTOL:-0}"

[[ -x "$CPU_EXE" ]] || { echo "CPU_EXE is not executable: $CPU_EXE" >&2; exit 2; }
[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite MP3 evidence directory: $OUT_DIR" >&2; exit 2; }
[[ "$RANKWISE" == t || "$RANKWISE" == f ]] || { echo "RANKWISE must be t or f" >&2; exit 2; }
[[ "$CALIBRATE" == t || "$CALIBRATE" == f ]] || { echo "CALIBRATE must be t or f" >&2; exit 2; }

if [[ "$CALIBRATE" == t ]]; then
  CANDIDATE_FIELD_ATOL=1.0
  CANDIDATE_FIELD_RTOL=0
  CANDIDATE_STATS_ATOL=1.0
  CANDIDATE_STATS_RTOL=0
fi

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
  local target="$1" exe="$2" precision="$3" candidate="$4"
  local -a shock_env=()
  [[ -n "$SHOCK_X" ]] && shock_env+=("ASTR_SHUOSHER_SHOCK_X=$SHOCK_X")
  (
    cd "$OUT_DIR/$target"
    if [[ -n "$precision" ]]; then
      env "${shock_env[@]}" \
        ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
        ASTR_SHOCK_SENSOR_DUMP="$OUT_DIR/${target}_shock_sensor.dat" \
        ASTR_GPU_PRECISION_MODE="$precision" \
        ASTR_GPU_MIXED_CANDIDATE="$candidate" \
        ASTR_GPU_SYNC_MODE=explicit \
        OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
        OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
        mpirun -np "$NP" "$exe" run datin/input.shuosher > "$target.log" 2>&1
    else
      env "${shock_env[@]}" \
        ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
        ASTR_SHOCK_SENSOR_DUMP="$OUT_DIR/${target}_shock_sensor.dat" \
        OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
        OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt \
        mpirun -np "$NP" "$exe" run datin/input.shuosher > "$target.log" 2>&1
    fi
  )
  rg -q 'The job is done!' "$OUT_DIR/$target/$target.log"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu_fp64 t
prepare_case gpu_characteristic_flux t
run_case cpu "$CPU_EXE" "" ""
run_case gpu_fp64 "$GPU_EXE" fp64 flux
run_case gpu_characteristic_flux "$GPU_EXE" mixed_workspace characteristic_flux

rg -q '^ASTR_GPU_PRECISION_MODE=mixed_workspace$' \
  "$OUT_DIR/gpu_characteristic_flux/gpu_characteristic_flux.log"
rg -q '^ASTR_GPU_MIXED_CANDIDATE=characteristic_flux$' \
  "$OUT_DIR/gpu_characteristic_flux/gpu_characteristic_flux.log"
rg -q '^ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux' \
  "$OUT_DIR/gpu_characteristic_flux/gpu_characteristic_flux.log"

rankwise_args=()
[[ "$RANKWISE" == t ]] && rankwise_args+=(--rankwise)

python3 "$ROOT_DIR/tests/gpu_validation/compare_shock_sensor.py" \
  --cpu "$OUT_DIR/cpu_shock_sensor.dat" --gpu "$OUT_DIR/gpu_fp64_shock_sensor.dat" \
  --report "$OUT_DIR/cpu_vs_gpu_fp64_shock_sensor.txt" \
  --atol "$SENSOR_ATOL" --rtol "$SENSOR_RTOL" "${rankwise_args[@]}"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu_fp64" \
  --report "$OUT_DIR/cpu_vs_gpu_fp64_flowstate.txt" \
  --atol "$STATS_ATOL" --rtol "$STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu_fp64" \
  --report "$OUT_DIR/cpu_vs_gpu_fp64_flowfield.txt" \
  --atol "$FIELD_ATOL" --rtol "$FIELD_RTOL"

python3 "$ROOT_DIR/tests/gpu_validation/compare_shock_sensor.py" \
  --cpu "$OUT_DIR/gpu_fp64_shock_sensor.dat" \
  --gpu "$OUT_DIR/gpu_characteristic_flux_shock_sensor.dat" \
  --report "$OUT_DIR/gpu_fp64_vs_candidate_shock_sensor.txt" \
  --atol 0 --rtol 0 "${rankwise_args[@]}"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/gpu_characteristic_flux" \
  --report "$OUT_DIR/gpu_fp64_vs_candidate_flowstate.txt" \
  --atol "$CANDIDATE_STATS_ATOL" --rtol "$CANDIDATE_STATS_RTOL"
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/gpu_fp64" --gpu "$OUT_DIR/gpu_characteristic_flux" \
  --report "$OUT_DIR/gpu_fp64_vs_candidate_flowfield.txt" \
  --atol "$CANDIDATE_FIELD_ATOL" --rtol "$CANDIDATE_FIELD_RTOL"

echo "MP3 characteristic_flux comparison passed: $OUT_DIR"
