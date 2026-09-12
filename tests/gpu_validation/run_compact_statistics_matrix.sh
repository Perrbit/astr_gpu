#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
GPU_IDS="${GPU_IDS:-0,1}"
CASE_SELECTOR="all"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/tests/gpu_validation/out/compact_statistics_$(date +%Y%m%d_%H%M%S)}"
ATOL="${ATOL:-1.0e-10}"
RTOL="${RTOL:-1.0e-10}"

usage() {
  printf 'Usage: %s [--case static-np1|dynamic-np1|curve-filter-np4|z-slab-np2|xyz-np8|restart-np1|all] [--result-dir PATH]\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --case)
      CASE_SELECTOR="${2:?--case requires a value}"
      shift 2
      ;;
    --result-dir)
      RESULT_DIR="${2:?--result-dir requires a value}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$CASE_SELECTOR" in
  static-np1|dynamic-np1|curve-filter-np4|z-slab-np2|xyz-np8|restart-np1|all) ;;
  *)
    printf 'unsupported compact statistics case: %s\n' "$CASE_SELECTOR" >&2
    exit 2
    ;;
esac
if [[ -e "$RESULT_DIR" ]]; then
  printf 'refusing to overwrite compact statistics evidence directory: %s\n' "$RESULT_DIR" >&2
  exit 2
fi
if [[ ! -x "$CPU_EXE" || ! -x "$GPU_EXE" ]]; then
  printf 'missing CPU or GPU executable: %s %s\n' "$CPU_EXE" "$GPU_EXE" >&2
  exit 2
fi
mkdir -p "$RESULT_DIR"
printf 'case\tstatus\tpath\n' > "$RESULT_DIR/status.tsv"

configure_statistics() {
  local controller="$1" maxstep="$2" feqchkpt="$3"
  sed -i 's/^f,f,f,f$/f,f,t,f/' "$controller"
  sed -i "/^# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg$/{n;s/.*/${maxstep},${feqchkpt},9999,9999,1,1/;}" "$controller"
}

configure_restart() {
  local case_dir="$1" maxstep="$2"
  sed -i '/^# lrestar$/{n;s/^[[:space:]]*f[[:space:]]*$/t/;}' "$case_dir/datin/input.flatplate"
  configure_statistics "$case_dir/datin/controller" "$maxstep" "$maxstep"
}

prepare_case() {
  local case_dir="$1" use_gpu="$2" im="$3" jm="$4" km="$5"
  local maxstep="$6" lfilter="$7" warp_x="$8" warp_y="$9" inflow="${10}"
  local args=(
    --dst-case "$case_dir" --use-gpu "$use_gpu"
    --im "$im" --jm "$jm" --km "$km" --mach 0.3
    --conschm 543e --diffterm t --lfilter "$lfilter"
    --wall-temperature 1.4 --isobaric-profile
    --maxstep "$maxstep" --feqchkpt "$maxstep" --deltat 1.0e-5
    --warp-x "$warp_x" --warp-y "$warp_y"
  )
  if [[ "$inflow" == dynamic ]]; then
    args+=(--ninit 3 --turbinf intp)
  else
    args+=(--ninit 0 --turbinf prof)
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" "${args[@]}"
  configure_statistics "$case_dir/datin/controller" "$maxstep" "$maxstep"
  if [[ "$inflow" == dynamic ]]; then
    python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
      --output "$case_dir/datin/flowini3d.h5" --grid "$im,$jm,$km" \
      --density 0.7 --u1 0.2 --u2 -0.1 --u3 0.05 --temperature 0.8
    python3 "$ROOT_DIR/tests/gpu_validation/generate_dynamic_inflow_slices.py" \
      --output "$case_dir/inflow" --jm "$jm" --km "$km" \
      --count 8 --delta-time 1.0e-5 --temporal-mode nonpolynomial
  fi
}

run_solver() {
  local kind="$1" np="$2" topology="$3" case_dir="$4" log_name="$5"
  local exe="$CPU_EXE"
  [[ "$kind" == gpu ]] && exe="$GPU_EXE"
  (
    cd "$case_dir"
    if [[ "$kind" == cpu ]]; then
      OMPI_MCA_sharedfp=individual \
        ASTR_FORCE_MPI_TOPOLOGY="$topology" \
        ASTR_VALIDATION_COMPACT_PREFIX=compact_reference \
        timeout --kill-after=10s 300s mpirun --oversubscribe -np "$np" \
        "$exe" run datin/input.flatplate > "$log_name" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        OMPI_MCA_sharedfp=individual \
        ASTR_FORCE_MPI_TOPOLOGY="$topology" \
        timeout --kill-after=10s 300s mpirun --oversubscribe -np "$np" \
        "$exe" run datin/input.flatplate > "$log_name" 2>&1
    fi
  )
  grep -q 'The job is done!' "$case_dir/$log_name"
}

validate_pair() {
  local label="$1" cpu_dir="$2" gpu_dir="$3"
  python3 "$ROOT_DIR/tests/gpu_validation/compact_statistics_host_reference.py" \
    --reference-prefix "$cpu_dir/compact_reference" \
    --sidecar-dir "$gpu_dir/outdat" \
    --report "$RESULT_DIR/${label}_compare.txt" \
    --atol "$ATOL" --rtol "$RTOL"
  python3 "$ROOT_DIR/scripts/gpu_statistics/assemble_compact_statistics.py" \
    --input-dir "$gpu_dir/outdat" \
    --output-h5 "$RESULT_DIR/${label}_compact.h5" \
    --metadata "$RESULT_DIR/${label}_compact.txt"
}

run_pair() {
  local label="$1" np="$2" topology="$3" im="$4" jm="$5" km="$6"
  local maxstep="$7" lfilter="$8" warp_x="$9" warp_y="${10}" inflow="${11}"
  local case_root="$RESULT_DIR/$label"
  prepare_case "$case_root/cpu" f "$im" "$jm" "$km" "$maxstep" "$lfilter" "$warp_x" "$warp_y" "$inflow"
  prepare_case "$case_root/gpu" t "$im" "$jm" "$km" "$maxstep" "$lfilter" "$warp_x" "$warp_y" "$inflow"
  run_solver cpu "$np" "$topology" "$case_root/cpu" cpu.log
  run_solver gpu "$np" "$topology" "$case_root/gpu" gpu.log
  validate_pair "$label" "$case_root/cpu" "$case_root/gpu"
  printf '%s\tPASS\t%s\n' "$label" "$case_root" >> "$RESULT_DIR/status.tsv"
}

run_restart_case() {
  local label=restart-np1 case_root="$RESULT_DIR/restart-np1"
  prepare_case "$case_root/cpu_continuous" f 16 16 8 4 f 0.0 0.0 static
  prepare_case "$case_root/gpu_continuous" t 16 16 8 4 f 0.0 0.0 static
  prepare_case "$case_root/gpu_restart" t 16 16 8 2 f 0.0 0.0 static
  run_solver cpu 1 1,1,1 "$case_root/cpu_continuous" continuous.log
  run_solver gpu 1 1,1,1 "$case_root/gpu_continuous" continuous.log
  run_solver gpu 1 1,1,1 "$case_root/gpu_restart" fresh.log
  configure_restart "$case_root/gpu_restart" 4
  run_solver gpu 1 1,1,1 "$case_root/gpu_restart" restart.log
  grep -q 'checkpoint file read' "$case_root/gpu_restart/restart.log"
  validate_pair "${label}_continuous" "$case_root/cpu_continuous" "$case_root/gpu_continuous"
  validate_pair "${label}_split" "$case_root/cpu_continuous" "$case_root/gpu_restart"
  printf '%s\tPASS\t%s\n' "$label" "$case_root" >> "$RESULT_DIR/status.tsv"
}

run_selected() {
  local selector="$1"
  case "$selector" in
    static-np1) run_pair static-np1 1 1,1,1 16 16 8 2 f 0.0 0.0 static ;;
    dynamic-np1) run_pair dynamic-np1 1 1,1,1 16 16 8 2 f 0.0 0.0 dynamic ;;
    curve-filter-np4) run_pair curve-filter-np4 4 2,2,1 32 32 16 2 t 0.03 0.02 dynamic ;;
    z-slab-np2) run_pair z-slab-np2 2 1,1,2 16 16 16 2 f 0.0 0.0 static ;;
    xyz-np8) run_pair xyz-np8 8 2,2,2 32 32 16 1 f 0.0 0.0 static ;;
    restart-np1) run_restart_case ;;
  esac
}

if [[ "$CASE_SELECTOR" == all ]]; then
  for selected in static-np1 dynamic-np1 z-slab-np2 curve-filter-np4 xyz-np8 restart-np1; do
    run_selected "$selected"
  done
else
  run_selected "$CASE_SELECTOR"
fi

printf 'COMPACT_STATISTICS_MATRIX_PASS %s\n' "$RESULT_DIR"
