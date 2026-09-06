#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_256_performance}"
LABEL="${LABEL:-baseline}"
GRID="${GRID:-256,256,256}"
MAXSTEP="${MAXSTEP:-10}"
REPEATS="${REPEATS:-5}"
DISCARD_STEPS="${DISCARD_STEPS:-1}"
GPU_ID="${GPU_ID:-0}"
SYNC_MODE="${SYNC_MODE:-explicit}"
FEQCHKPT="${FEQCHKPT:-9999}"
TIMINGS="$OUT_DIR/${LABEL}_timings.tsv"
SUMMARY="$OUT_DIR/${LABEL}_summary.md"
CASE_DIR="$OUT_DIR/${LABEL}_case"

if [[ "$GPU_EXE" != /* ]]; then
  GPU_EXE="$ROOT_DIR/$GPU_EXE"
fi
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
  TIMINGS="$OUT_DIR/${LABEL}_timings.tsv"
  SUMMARY="$OUT_DIR/${LABEL}_summary.md"
  CASE_DIR="$OUT_DIR/${LABEL}_case"
fi

if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
if [[ "$REPEATS" -lt 5 ]]; then
  echo "REPEATS must be at least 5 for the performance acceptance gate" >&2
  exit 2
fi
if [[ "$DISCARD_STEPS" -lt 0 || "$DISCARD_STEPS" -ge $((MAXSTEP + 1)) ]]; then
  echo "DISCARD_STEPS must be between 0 and MAXSTEP" >&2
  exit 2
fi
if [[ "$FEQCHKPT" -le "$MAXSTEP" ]]; then
  echo "FEQCHKPT must exceed MAXSTEP so measured runs do not write fields" >&2
  exit 2
fi
if [[ "$SYNC_MODE" != "explicit" && "$SYNC_MODE" != "selective" ]]; then
  echo "SYNC_MODE must be explicit or selective" >&2
  exit 2
fi

prepare_case() {
  local case_dir="$1"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
    --dst-case "$case_dir" --use-gpu t --grid "$GRID" \
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" \
    --lfilter t --diffterm t --scheme 643e
}

run_once() {
  local repeat="$1" record="$2"
  local run_dir log monitor stop monitor_pid start end wall
  local max_memory max_util timing_count retained_count timing_values run_status
  run_dir="$OUT_DIR/${LABEL}_run_${repeat}"
  log="$run_dir/run.log"
  monitor="$run_dir/gpu_monitor.csv"
  stop="$run_dir/.monitor_stop"
  mkdir -p "$run_dir"
  rm -f "$stop"
  (
    while [[ ! -e "$stop" ]]; do
      nvidia-smi --id="$GPU_ID" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 0.1
    done
  ) > "$monitor" &
  monitor_pid=$!
  start="$(date +%s.%N)"
  set +e
  (
    cd "$CASE_DIR"
    CUDA_VISIBLE_DEVICES="$GPU_ID" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
      ASTR_GPU_RK_TIMING=1 ASTR_GPU_SYNC_MODE="$SYNC_MODE" \
      mpirun -np 1 "$GPU_EXE" \
      run datin/input.tgv > "$log" 2>&1
  )
  run_status=$?
  set -e
  end="$(date +%s.%N)"
  touch "$stop"
  wait "$monitor_pid"
  if [[ "$run_status" -ne 0 ]]; then
    echo "ASTR failed with status $run_status; see $log" >&2
    return "$run_status"
  fi

  grep -q 'The job is done!' "$log"
  if grep -Eq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    echo "non-finite or crash marker found in $log" >&2
    exit 1
  fi
  timing_count="$(awk '$1 == "ASTR_GPU_RK_TIMING" {count++} END {print count+0}' "$log")"
  if [[ "$timing_count" -ne $((MAXSTEP + 1)) ]]; then
    echo "expected $((MAXSTEP + 1)) RK timings, found $timing_count in $log" >&2
    exit 1
  fi
  if [[ "$record" != "t" ]]; then
    return
  fi

  timing_values="$run_dir/rk_seconds.txt"
  awk -v discard="$DISCARD_STEPS" \
    '$1 == "ASTR_GPU_RK_TIMING" {seen++; if (seen > discard) print $5}' \
    "$log" > "$timing_values"
  retained_count="$(wc -l < "$timing_values")"
  wall="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$end")"
  max_memory="$(awk -F',' '{gsub(/ /,"",$1); if ($1+0>m) m=$1+0} END {print m+0}' "$monitor")"
  max_util="$(awk -F',' '{gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$monitor")"
  python3 - "$LABEL" "$repeat" "$retained_count" "$wall" "$max_memory" "$max_util" "$timing_values" >> "$TIMINGS" <<'PY'
import statistics
import sys
from pathlib import Path

label, repeat, count, wall, memory, utilization, path = sys.argv[1:]
values = [float(value) for value in Path(path).read_text(encoding="ascii").split()]
if len(values) != int(count) or not values:
    raise SystemExit("invalid retained RK timing sample count")
print(
    f"{label}\t{repeat}\t{count}\t{statistics.median(values):.12f}\t"
    f"{min(values):.12f}\t{max(values):.12f}\t{float(wall):.6f}\t"
    f"{memory}\t{utilization}"
)
PY
}

mkdir -p "$OUT_DIR"
prepare_case "$CASE_DIR"
printf 'label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent\n' > "$TIMINGS"

run_once warmup f
for repeat in $(seq 1 "$REPEATS"); do
  run_once "$repeat" t
done

python3 "$ROOT_DIR/tests/gpu_validation/summarize_tgv_performance.py" \
  --timings "$TIMINGS" --summary "$SUMMARY" --grid "$GRID" \
  --maxstep "$MAXSTEP" --discard-steps "$DISCARD_STEPS" \
  --sync-mode "$SYNC_MODE"
