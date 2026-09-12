#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new D2 stage evidence directory}"
FIELD_ATOL="${FIELD_ATOL:-1.0e-10}"
GPU_IDS="${GPU_IDS:-0,1}"
FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"

if [[ -e "$OUT_DIR" ]]; then
  printf 'refusing to overwrite D2 stage evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
if [[ ! -x "$CPU_EXE" || ! -x "$GPU_EXE" ]]; then
  printf 'missing CPU or GPU validation executable\n' >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
printf 'case\tstatus\tmax_abs\treport\n' > "$OUT_DIR/status.tsv"

prepare_case() {
  local case_dir="$1" use_gpu="$2" lfilter="$3" warp_x="$4" warp_y="$5"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$case_dir" --use-gpu "$use_gpu" \
    --im 32 --jm 32 --km 16 --mach 0.3 \
    --conschm 543e --diffterm t --lfilter "$lfilter" --wall-temperature 1.4 \
    --isobaric-profile --ninit 3 --maxstep 1 --feqchkpt 1 \
    --deltat 6.0e-6 --turbinf intp --warp-x "$warp_x" --warp-y "$warp_y"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$case_dir/datin/flowini3d.h5" --grid 32,32,16 \
    --density 0.7 --u1 0.2 --u2 -0.1 --u3 0.05 --temperature 0.8
  python3 "$ROOT_DIR/tests/gpu_validation/generate_dynamic_inflow_slices.py" \
    --output "$case_dir/inflow" --jm 32 --km 16 \
    --count 12 --delta-time 1.0e-5 --temporal-mode nonpolynomial
}

run_solver() {
  local kind="$1" np="$2" topology="$3" case_dir="$4" prefix="$5"
  local exe="$CPU_EXE"
  [[ "$kind" == gpu ]] && exe="$GPU_EXE"
  (
    cd "$case_dir"
    if [[ "$kind" == cpu ]]; then
      OMPI_MCA_sharedfp=individual \
        ASTR_FORCE_MPI_TOPOLOGY="$topology" \
        ASTR_VALIDATION_RHS_PREFIX="$prefix" \
        timeout --kill-after=10s 300s mpirun --oversubscribe -np "$np" \
        "$exe" run datin/input.flatplate > run.log 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        OMPI_MCA_sharedfp=individual \
        ASTR_FORCE_MPI_TOPOLOGY="$topology" \
        ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
        ASTR_VALIDATION_RHS_PREFIX="$prefix" \
        timeout --kill-after=10s 300s mpirun --oversubscribe -np "$np" \
        "$exe" run datin/input.flatplate > run.log 2>&1
    fi
  )
}

run_case() {
  local label="$1" np="$2" topology="$3" lfilter="$4" warp_x="$5" warp_y="$6"
  local case_root="$OUT_DIR/$label"
  local cpu_case="$case_root/cpu"
  local gpu_case="$case_root/gpu"
  local cpu_prefix="$case_root/snapshots/cpu"
  local gpu_prefix="$case_root/snapshots/gpu"
  local report="$case_root/stage_compare.txt"
  mkdir -p "$case_root/snapshots"
  prepare_case "$cpu_case" f "$lfilter" "$warp_x" "$warp_y"
  prepare_case "$gpu_case" t "$lfilter" "$warp_x" "$warp_y"
  run_solver cpu "$np" "$topology" "$cpu_case" "$cpu_prefix"
  run_solver gpu "$np" "$topology" "$gpu_case" "$gpu_prefix"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
    --cpu-prefix "$cpu_prefix" --gpu-prefix "$gpu_prefix" \
    --labels pre_rhs,post_update --atol "$FIELD_ATOL" --rtol 0 \
    --report "$report"
  local max_abs
  max_abs="$(awk '/^max_abs:/ {print $2}' "$report")"
  printf '%s\tPASS\t%s\t%s\n' "$label" "$max_abs" "$report" >> "$OUT_DIR/status.tsv"
}

run_case dynamic_flat_np1 1 1,1,1 f 0.0 0.0
run_case dynamic_filter_curve_np4 4 2,2,1 t 0.03 0.02

printf 'DYNAMIC_INFLOW_D2_STAGE_PASS %s\n' "$OUT_DIR"
