#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Channel"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/channel_128_benchmark}"
BASELINE_ENTRY="${BASELINE_ENTRY:-1:1,1,1}"
MATRIX="${MATRIX:-1:1,1,1 2:2,1,1 2:1,2,1 2:1,1,2 4:2,2,1 4:2,1,2 4:1,2,2 8:2,2,2}"
MAXSTEP="${MAXSTEP:-100}"
FEQCHKPT="${FEQCHKPT:-9999}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-128,128,128}"
DELTAT="${DELTAT:-7.5d-4}"
STATS_ATOL="${STATS_ATOL:-1e-8}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
CHANNEL_FORCE_MODE="${CHANNEL_FORCE_MODE:-fixed}"
CHANNEL_FORCE_FIXED="${CHANNEL_FORCE_FIXED:-1.d-4}"
RUN_CPU_FOR_ALL="${RUN_CPU_FOR_ALL:-t}"
TIMINGS="$OUT_DIR/benchmark_times.tsv"
SUMMARY="$OUT_DIR/benchmark_summary.md"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

case_tag() {
  local entry="$1" np topology
  IFS=':' read -r np topology <<< "$entry"
  printf 'channel_np%s_%s' "$np" "$(topology_tag "$topology")"
}

seconds_between() {
  python3 -c 'import sys; print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")' "$1" "$2"
}

prepare_case() {
  local use_gpu="$1" dst_case="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$dst_case" \
    --input-name input.chl \
    --homogeneous "t,f,t" \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" \
    --lfilter "$LFILTER" \
    --diffterm "$DIFFTERM" \
    --scheme "$SCHEME" \
    --grid "$GRID" \
    --deltat "$DELTAT"
}

run_timed() {
  local case_name="$1" role="$2" np="$3" topology="$4" exe="$5" run_dir="$6" log_name="$7"
  local start end elapsed

  start="$(date +%s.%N)"
  (
    cd "$run_dir"
    ASTR_CHANNEL_FORCE_MODE="$CHANNEL_FORCE_MODE" \
    ASTR_CHANNEL_FORCE_FIXED="$CHANNEL_FORCE_FIXED" \
    ASTR_FORCE_MPI_TOPOLOGY="$topology" \
      mpirun -np "$np" "$exe" run datin/input.chl > "$log_name" 2>&1
  )
  end="$(date +%s.%N)"
  elapsed="$(seconds_between "$start" "$end")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$case_name" "$role" "$np" "$topology" "$elapsed" "$run_dir" >> "$TIMINGS"
}

run_entry() {
  local entry="$1" np topology tag case_out case_name run_cpu
  IFS=':' read -r np topology <<< "$entry"
  tag="$(topology_tag "$topology")"
  case_name="channel_np${np}_${tag}"
  case_out="$OUT_DIR/$case_name"
  run_cpu="f"

  mkdir -p "$case_out"
  prepare_case f "$case_out/cpu"
  prepare_case t "$case_out/gpu"

  if [[ "$RUN_CPU_FOR_ALL" == "t" || "$entry" == "$BASELINE_ENTRY" ]]; then
    run_cpu="t"
    run_timed "$case_name" cpu "$np" "$topology" "$CPU_EXE" "$case_out/cpu" cpu.log
  fi

  run_timed "$case_name" gpu "$np" "$topology" "$GPU_EXE" "$case_out/gpu" gpu.log

  if [[ "$run_cpu" == "t" ]]; then
    python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
      --cpu "$case_out/cpu" \
      --gpu "$case_out/gpu" \
      --report "$case_out/flowstate_compare.txt" \
      --atol "$STATS_ATOL" \
      --rtol "$STATS_RTOL"
  fi
}

mkdir -p "$OUT_DIR"
: > "$TIMINGS"
printf 'case\trole\tnp\ttopology\tseconds\tout_dir\n' >> "$TIMINGS"

if [[ " $MATRIX " != *" $BASELINE_ENTRY "* ]]; then
  MATRIX="$BASELINE_ENTRY $MATRIX"
fi

for entry in $MATRIX; do
  run_entry "$entry"
done

python3 "$ROOT_DIR/tests/gpu_validation/channel_benchmark_summary.py" \
  --timings "$TIMINGS" \
  --baseline-case "$(case_tag "$BASELINE_ENTRY")" \
  --summary "$SUMMARY" \
  --grid "$GRID" \
  --deltat "$DELTAT" \
  --maxstep "$MAXSTEP"

cat "$SUMMARY"
