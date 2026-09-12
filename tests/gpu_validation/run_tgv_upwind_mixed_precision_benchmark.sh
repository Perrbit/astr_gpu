#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_upwind_mixed_precision_benchmark}"
GRID="${GRID:-64,64,64}"
MAXSTEP="${MAXSTEP:-5}"
FEQCHKPT="${FEQCHKPT:-9999}"
REPEATS="${REPEATS:-5}"
DISCARD_STEPS="${DISCARD_STEPS:-1}"
RECON_SCHEM="${RECON_SCHEM:-1}"
GPU_ID="${GPU_ID:-0}"
CASE_DIR="$OUT_DIR/case"
FP64_TIMINGS="$OUT_DIR/fp64_timings.tsv"
MIXED_TIMINGS="$OUT_DIR/mixed_workspace_timings.tsv"

if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
if [[ -e "$OUT_DIR" ]]; then
  echo "refusing to overwrite benchmark evidence directory: $OUT_DIR" >&2
  exit 2
fi
if [[ "$REPEATS" -lt 5 ]]; then
  echo "REPEATS must be at least 5" >&2
  exit 2
fi
if [[ "$DISCARD_STEPS" -lt 0 || "$DISCARD_STEPS" -ge $((MAXSTEP + 1)) ]]; then
  echo "DISCARD_STEPS must retain at least one RK timing" >&2
  exit 2
fi
if [[ "$RECON_SCHEM" != "1" && "$RECON_SCHEM" != "3" ]]; then
  echo "RECON_SCHEM must be 1 (WENO7) or 3 (MP7)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
  --dst-case "$CASE_DIR" --input-name input.tgv --flowtype tgv \
  --homogeneous t,t,t --bctype 1,1,1,1,1,1 --use-gpu t \
  --grid "$GRID" --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" \
  --lfilter f --diffterm f --scheme 643e --conschm 543e \
  --difschm 643e --recon-schem "$RECON_SCHEM" --lchardecomp f \
  --deltat 1.d-4

header='label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent'
printf '%b\n' "$header" > "$FP64_TIMINGS"
printf '%b\n' "$header" > "$MIXED_TIMINGS"

run_once() {
  local mode="$1" repeat="$2" record="$3"
  local label timings run_dir log monitor stop monitor_pid start end status
  local timing_count timing_values retained_count wall max_memory max_util
  label="$mode"
  [[ "$mode" == "mixed_workspace" ]] && label="mixed"
  timings="$FP64_TIMINGS"
  [[ "$mode" == "mixed_workspace" ]] && timings="$MIXED_TIMINGS"
  run_dir="$OUT_DIR/${label}_run_${repeat}"
  log="$run_dir/run.log"
  monitor="$run_dir/gpu_monitor.csv"
  stop="$run_dir/.monitor_stop"
  mkdir -p "$run_dir"

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
      ASTR_GPU_PRECISION_MODE="$mode" ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      OMPI_MCA_coll_hcoll_enable=0 \
      mpirun -np 1 "$GPU_EXE" run datin/input.tgv > "$log" 2>&1
  )
  status=$?
  set -e
  end="$(date +%s.%N)"
  touch "$stop"
  wait "$monitor_pid"
  if [[ "$status" -ne 0 ]]; then
    echo "ASTR failed with status $status; see $log" >&2
    return "$status"
  fi

  rg -q 'The job is done!' "$log"
  rg -q "^ASTR_GPU_PRECISION_MODE=$mode$" "$log"
  timing_count="$(awk '$1 == "ASTR_GPU_RK_TIMING" {count++} END {print count+0}' "$log")"
  if [[ "$timing_count" -ne $((MAXSTEP + 1)) ]]; then
    echo "expected $((MAXSTEP + 1)) RK timings, found $timing_count in $log" >&2
    exit 1
  fi
  if [[ "$record" != "t" ]]; then
    return
  fi

  timing_values="$run_dir/rk_seconds.txt"
  python3 "$ROOT_DIR/tests/gpu_validation/summarize_rank_rk_timing.py" \
    --log "$log" --ranks 1 --steps "$((MAXSTEP + 1))" \
    --discard "$DISCARD_STEPS" > "$run_dir/rank_rk.tsv"
  awk 'NR>1 {print $2}' "$run_dir/rank_rk.tsv" > "$timing_values"
  retained_count="$(wc -l < "$timing_values")"
  wall="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$end")"
  max_memory="$(awk -F',' '{gsub(/ /,"",$1); if ($1+0>m) m=$1+0} END {print m+0}' "$monitor")"
  max_util="$(awk -F',' '{gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$monitor")"
  python3 - "$label" "$repeat" "$retained_count" "$wall" \
    "$max_memory" "$max_util" "$timing_values" >> "$timings" <<'PY'
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

run_once fp64 warmup f
run_once mixed_workspace warmup f
for repeat in $(seq 1 "$REPEATS"); do
  if (( repeat % 2 == 1 )); then
    run_once fp64 "$repeat" t
    run_once mixed_workspace "$repeat" t
  else
    run_once mixed_workspace "$repeat" t
    run_once fp64 "$repeat" t
  fi
done

fp64_bytes="$(awk -F= '/^ASTR_GPU_FLUX_WORKSPACE_BYTES=/ {print $2; exit}' "$OUT_DIR/fp64_run_1/run.log")"
mixed_bytes="$(awk -F= '/^ASTR_GPU_FLUX_WORKSPACE_BYTES=/ {print $2; exit}' "$OUT_DIR/mixed_run_1/run.log")"
python3 "$ROOT_DIR/tests/gpu_validation/summarize_mixed_precision_benchmark.py" \
  --fp64 "$FP64_TIMINGS" --mixed "$MIXED_TIMINGS" \
  --fp64-workspace-bytes "$fp64_bytes" \
  --mixed-workspace-bytes "$mixed_bytes" \
  --report "$OUT_DIR/summary.md"
