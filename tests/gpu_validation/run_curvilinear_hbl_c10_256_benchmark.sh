#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_c10_256_benchmark}"
GRID="${GRID:-256,256,256}"
MAXSTEP="${MAXSTEP:-5}"
REPEATS="${REPEATS:-3}"
MATRIX="${MATRIX:-np1:1:1,1,1 np2_x:2:2,1,1}"
DELTAT="${DELTAT:-1e-6}"
FEQCHKPT="${FEQCHKPT:-9999}"
PROFILE_POINTS="${PROFILE_POINTS:-8001}"
REUSE_CASES="${REUSE_CASES:-f}"
TIMINGS="$OUT_DIR/timings.tsv"
SUMMARY="$OUT_DIR/benchmark_summary.md"

IFS=',' read -r IM JM KM <<< "$GRID"
if [[ -z "${KM:-}" || "$IM" -lt 8 || "$JM" -lt 8 || "$KM" -lt 8 ]]; then
  echo "GRID must contain three comma-separated dimensions of at least 8" >&2
  exit 2
fi
if [[ "$REPEATS" -lt 3 ]]; then
  echo "REPEATS must be at least 3" >&2
  exit 2
fi

prepare_case() {
  local case_dir="$1"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$case_dir" --use-gpu t \
    --im "$IM" --jm "$JM" --km "$KM" \
    --diffterm t --lfilter f --conschm 543e --lchardecomp t \
    --shock-threshold 0.001 --reynolds 1.83052e6 --mach 5.0 \
    --reference-temperature 226.65 --wall-temperature 5.191440547760865 \
    --upper-bctype 51 --x-min-bctype 11 --ninit 3 \
    --x-min -1.0 --x-max 10.0 --y-stretch 5.0 --z-length 0.25 \
    --warp-x 0.4 --warp-y 0.2 --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" --deltat "$DELTAT"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_compressible_blasius_profile.py" \
    --grid "$case_dir/datin/grid.flatplate.h5" \
    --output "$case_dir/datin/inlet.prof" --mach 5.0 --reynolds 1.83052e6 \
    --reference-temperature 226.65 --wall-temperature 5.191440547760865 \
    --station-x 1.0 --density-mode provided --pressure-mode provided \
    --field-output "$case_dir/datin/flowini3d.h5" --virtual-leading-edge -2.0 \
    --field-oblique-shock --shock-angle-deg 35.0 --shock-x0 -1.0 \
    --shock-y0 0.18 --shock-y-min 0.18 --profile-oblique-shock \
    --profile-shock-y-min 0.18 --points "$PROFILE_POINTS"
}

run_once() {
  local label="$1" np="$2" topology="$3" repeat="$4" record="$5"
  local case_dir log monitor stop
  local start end seconds max_memory max_util monitor_pid
  case_dir="$OUT_DIR/$label"
  log="$case_dir/run_${repeat}.log"
  monitor="$case_dir/gpu_monitor_${repeat}.csv"
  stop="$case_dir/.monitor_stop"
  mkdir -p "$case_dir/tmp"
  rm -f "$stop"
  (
    while [[ ! -e "$stop" ]]; do
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr '\n' ';'
      printf '\n'
      sleep 0.25
    done
  ) > "$monitor" &
  monitor_pid=$!
  start="$(date +%s.%N)"
  (
    cd "$case_dir"
    TMPDIR="$case_dir/tmp" ASTR_FORCE_MPI_TOPOLOGY="$topology" \
      mpirun -np "$np" "$GPU_EXE" run datin/input.flatplate > "$log" 2>&1
  )
  end="$(date +%s.%N)"
  touch "$stop"
  wait "$monitor_pid"
  grep -q 'The job is done!' "$log"
  if grep -Eq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    echo "non-finite or crash marker found in $log" >&2
    exit 1
  fi
  if [[ "$record" == "t" ]]; then
    seconds="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$end")"
    max_memory="$(tr ';' '\n' < "$monitor" | awk -F',' 'NF >= 3 {gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}')"
    max_util="$(tr ';' '\n' < "$monitor" | awk -F',' 'NF >= 3 {gsub(/ /,"",$3); if ($3+0>m) m=$3+0} END {print m+0}')"
    printf '%s\t%s\t%s\t%s\t%.6f\t%s\t%s\n' \
      "$label" "$repeat" "$np" "$topology" "$seconds" "$max_memory" "$max_util" \
      >> "$TIMINGS"
  fi
}

mkdir -p "$OUT_DIR"
printf 'label\trepeat\tnp\ttopology\tseconds\tmax_memory_mib\tmax_utilization_percent\n' > "$TIMINGS"

for entry in $MATRIX; do
  IFS=':' read -r label np topology <<< "$entry"
  if [[ "$REUSE_CASES" != "t" || ! -f "$OUT_DIR/$label/datin/flowini3d.h5" ]]; then
    prepare_case "$OUT_DIR/$label"
  fi
  run_once "$label" "$np" "$topology" warmup f
  for repeat in $(seq 1 "$REPEATS"); do
    run_once "$label" "$np" "$topology" "$repeat" t
  done
done

python3 "$ROOT_DIR/tests/gpu_validation/summarize_repeated_gpu_benchmark.py" \
  --timings "$TIMINGS" --summary "$SUMMARY" --grid "$GRID" --maxstep "$MAXSTEP"
