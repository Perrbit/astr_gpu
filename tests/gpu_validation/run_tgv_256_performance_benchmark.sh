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
GPU_IDS="${GPU_IDS:-$GPU_ID}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
SYNC_MODE="${SYNC_MODE:-explicit}"
PHASE_TIMING="${PHASE_TIMING:-0}"
HALO_TRANSPORT="${HALO_TRANSPORT:-pageable}"
FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"
FEQCHKPT="${FEQCHKPT:-9999}"
DELTAT="${DELTAT:-}"
TIMINGS="$OUT_DIR/${LABEL}_timings.tsv"
SUMMARY="$OUT_DIR/${LABEL}_summary.md"
PHASE_SUMMARY="$OUT_DIR/${LABEL}_phase_summary.tsv"
CASE_DIR="$OUT_DIR/${LABEL}_case"
MONITOR_PID=""
MONITOR_FILE=""
MONITOR_ERROR=""
SOLVER_PID=""
CLEANUP_POLL_ATTEMPTS=50
CLEANUP_POLL_SECONDS=0.02

validate_monitor_samples() {
  local monitor="$1"
  awk -F',' '
    function numeric(value) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      return value ~ /^[0-9]+([.][0-9]+)?$/
    }
    NF == 2 && numeric($1) && numeric($2) { valid++ ; next }
    NF > 0 { invalid++ }
    END { exit !(valid > 0 && invalid == 0) }
  ' "$monitor"
}

start_monitor() {
  local monitor="$1" error="$2" pid pgid
  : > "$monitor"
  : > "$error"
  setsid nvidia-smi --id="$GPU_IDS" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits --loop-ms=100 \
    > "$monitor" 2> "$error" &
  pid=$!
  MONITOR_PID="$pid"
  MONITOR_FILE="$monitor"
  MONITOR_ERROR="$error"

  for _ in $(seq 1 50); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$pgid" == "$pid" ]]; then
      return 0
    fi
    sleep 0.02
  done

  echo "GPU monitor failed to establish its process group; see $error" >&2
  stop_monitor f || true
  return 1
}

process_group_alive() {
  kill -0 -- "-$1" 2>/dev/null
}

terminate_process_group() {
  local pid="$1"
  local result_leader_was_alive=0 result_term_sent=0
  local result_kill_sent=0 result_wait_status=0
  local group_stopped=0 leader_stopped=0

  if kill -0 "$pid" 2>/dev/null; then
    result_leader_was_alive=1
  fi
  if process_group_alive "$pid"; then
    if kill -TERM -- "-$pid" 2>/dev/null; then
      result_term_sent=1
    fi
  fi

  for _ in $(seq 1 "$CLEANUP_POLL_ATTEMPTS"); do
    if ! process_group_alive "$pid"; then
      group_stopped=1
      break
    fi
    sleep "$CLEANUP_POLL_SECONDS"
  done
  if [[ "$group_stopped" -ne 1 ]]; then
    if kill -KILL -- "-$pid" 2>/dev/null; then
      result_kill_sent=1
    fi
    for _ in $(seq 1 "$CLEANUP_POLL_ATTEMPTS"); do
      if ! process_group_alive "$pid"; then
        group_stopped=1
        break
      fi
      sleep "$CLEANUP_POLL_SECONDS"
    done
  fi

  for _ in $(seq 1 "$CLEANUP_POLL_ATTEMPTS"); do
    if ! kill -0 "$pid" 2>/dev/null; then
      leader_stopped=1
      break
    fi
    sleep "$CLEANUP_POLL_SECONDS"
  done
  if [[ "$leader_stopped" -eq 1 ]]; then
    if wait "$pid" 2>/dev/null; then
      result_wait_status=0
    else
      result_wait_status=$?
    fi
  fi

  if [[ "$#" -eq 5 ]]; then
    printf -v "$2" '%s' "$result_leader_was_alive"
    printf -v "$3" '%s' "$result_term_sent"
    printf -v "$4" '%s' "$result_kill_sent"
    printf -v "$5" '%s' "$result_wait_status"
  fi

  [[ "$group_stopped" -eq 1 && "$leader_stopped" -eq 1 ]]
}

stop_monitor() {
  local strict="${1:-f}"
  local pid="${MONITOR_PID:-}" monitor="${MONITOR_FILE:-}"
  local error="${MONITOR_ERROR:-}" leader_was_alive term_sent kill_sent wait_status
  local cleanup_status=0
  if [[ -z "$pid" ]]; then
    return 0
  fi

  terminate_process_group "$pid" leader_was_alive term_sent kill_sent wait_status || \
    cleanup_status=$?
  MONITOR_PID=""
  MONITOR_FILE=""
  MONITOR_ERROR=""

  if [[ "$strict" != "t" ]]; then
    return "$cleanup_status"
  fi
  if [[ "$cleanup_status" -ne 0 ]]; then
    echo "GPU monitor process group could not be stopped; see $error" >&2
    return 1
  fi
  if [[ "$kill_sent" -eq 1 ]]; then
    echo "GPU monitor process group required SIGKILL during cleanup; see $error" >&2
    return 1
  fi
  if [[ "$leader_was_alive" -ne 1 || "$term_sent" -ne 1 ]]; then
    echo "GPU monitor exited unexpectedly with status $wait_status; see $error" >&2
    return 1
  fi
  if [[ "$wait_status" -ne 0 && "$wait_status" -ne 143 ]]; then
    echo "GPU monitor exited unexpectedly with status $wait_status; see $error" >&2
    return 1
  fi
  if ! validate_monitor_samples "$monitor"; then
    echo "GPU monitor produced no valid samples; see $monitor and $error" >&2
    return 1
  fi
}

stop_solver() {
  local pid="${SOLVER_PID:-}"
  local cleanup_status=0
  if [[ -z "$pid" ]]; then
    return 0
  fi
  terminate_process_group "$pid" || cleanup_status=$?
  SOLVER_PID=""
  return "$cleanup_status"
}

handle_exit() {
  local status=$?
  trap - EXIT INT TERM
  stop_solver || true
  stop_monitor f || true
  exit "$status"
}

handle_signal() {
  local status="$1"
  trap - EXIT INT TERM
  stop_solver || true
  stop_monitor f || true
  exit "$status"
}

trap 'handle_exit' EXIT
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

if [[ "$GPU_EXE" != /* ]]; then
  GPU_EXE="$ROOT_DIR/$GPU_EXE"
fi
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
  TIMINGS="$OUT_DIR/${LABEL}_timings.tsv"
  SUMMARY="$OUT_DIR/${LABEL}_summary.md"
  PHASE_SUMMARY="$OUT_DIR/${LABEL}_phase_summary.tsv"
  CASE_DIR="$OUT_DIR/${LABEL}_case"
fi

if [[ ! -x "$GPU_EXE" ]]; then
  echo "GPU executable not found: $GPU_EXE" >&2
  exit 2
fi
if ! command -v setsid >/dev/null 2>&1; then
  echo "setsid is required for GPU monitor process-group cleanup" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for GPU benchmark monitoring" >&2
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
if [[ "$SYNC_MODE" != "explicit" && "$SYNC_MODE" != "selective" && \
      "$SYNC_MODE" != "dependency" ]]; then
  echo "SYNC_MODE must be explicit, selective or dependency" >&2
  exit 2
fi
if [[ "$PHASE_TIMING" != "0" && "$PHASE_TIMING" != "1" ]]; then
  echo "PHASE_TIMING must be 0 or 1" >&2
  exit 2
fi
if [[ "$PHASE_TIMING" == "1" && "$SYNC_MODE" != "explicit" ]]; then
  echo "PHASE_TIMING=1 requires SYNC_MODE=explicit" >&2
  exit 2
fi
if [[ "$HALO_TRANSPORT" != "pageable" && "$HALO_TRANSPORT" != "pinned" && \
      "$HALO_TRANSPORT" != "pinned-overlap" && "$HALO_TRANSPORT" != "pinned-pipeline" && \
      "$HALO_TRANSPORT" != "device-aware" ]]; then
  echo "HALO_TRANSPORT must be pageable, pinned, pinned-overlap, pinned-pipeline or device-aware" >&2
  exit 2
fi
if [[ "$HALO_TRANSPORT" == "pinned-pipeline" && "$SYNC_MODE" == "selective" ]]; then
  echo "HALO_TRANSPORT=pinned-pipeline does not support SYNC_MODE=selective" >&2
  exit 2
fi
if [[ "$SYNC_MODE" == "dependency" && "$HALO_TRANSPORT" != "pinned-pipeline" ]]; then
  echo "SYNC_MODE=dependency requires HALO_TRANSPORT=pinned-pipeline" >&2
  exit 2
fi
if [[ "$FILTER_WORKSPACE" != "full" && "$FILTER_WORKSPACE" != "scalar" ]]; then
  echo "FILTER_WORKSPACE must be full or scalar" >&2
  exit 2
fi
python3 - "$NP" "$TOPOLOGY" "$GPU_IDS" <<'PY'
import math
import sys

np = int(sys.argv[1])
topology = tuple(map(int, sys.argv[2].split(',')))
devices = sys.argv[3].split(',')
if len(topology) != 3 or min(topology) < 1 or math.prod(topology) != np:
    raise SystemExit('topology must contain three positive factors with product NP')
if np < 1 or len(devices) != np or len(set(devices)) != np:
    raise SystemExit('performance requires one distinct visible GPU per rank')
PY

prepare_case() {
  local case_dir="$1"
  local args=(
    "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py"
    --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex"
    --dst-case "$case_dir" --use-gpu t --grid "$GRID"
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT"
    --lfilter t --diffterm t --scheme 643e
  )
  args+=(--lreadgrid f)
  if [[ -n "$DELTAT" ]]; then
    args+=(--deltat "$DELTAT")
  fi
  python3 "${args[@]}"
}

assert_no_field_hdf5() {
  local found
  found="$(find "$CASE_DIR" -type f \
    \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  if [[ -n "$found" ]]; then
    printf 'benchmark generated forbidden field HDF5:\n%s\n' "$found" >&2
    exit 1
  fi
}

run_once() {
  local repeat="$1" record="$2"
  local run_dir log monitor monitor_error start end wall
  local max_memory max_util timing_count retained_count timing_values run_status
  local monitor_status=0 solver_cleanup_status=0 solver_group_remaining=0
  assert_no_field_hdf5
  run_dir="$OUT_DIR/${LABEL}_run_${repeat}"
  log="$run_dir/run.log"
  monitor="$run_dir/gpu_monitor.csv"
  monitor_error="$run_dir/gpu_monitor.stderr"
  mkdir -p "$run_dir"
  start_monitor "$monitor" "$monitor_error"
  start="$(date +%s.%N)"
  (
    cd "$CASE_DIR"
    exec setsid env OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}" \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      ASTR_GPU_PHASE_TIMING="$PHASE_TIMING" \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 \
      ASTR_GPU_SYNC_MODE="$SYNC_MODE" ASTR_GPU_HALO_TRANSPORT="$HALO_TRANSPORT" \
      ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
      mpirun -np "$NP" "$GPU_EXE" \
      run datin/input.tgv
  ) > "$log" 2>&1 &
  SOLVER_PID=$!
  if wait "$SOLVER_PID"; then
    run_status=0
  else
    run_status=$?
  fi
  if process_group_alive "$SOLVER_PID"; then
    solver_group_remaining=1
    terminate_process_group "$SOLVER_PID" || solver_cleanup_status=$?
  fi
  if [[ "$solver_cleanup_status" -eq 0 ]]; then
    SOLVER_PID=""
  else
    echo "ASTR solver process group could not be stopped" >&2
  fi
  if [[ "$solver_group_remaining" -eq 1 && "$run_status" -eq 0 ]]; then
    echo "ASTR solver exited successfully but left process-group members" >&2
    run_status=1
  fi
  end="$(date +%s.%N)"
  if [[ "$run_status" -ne 0 ]]; then
    stop_monitor f || true
  elif ! stop_monitor t; then
    monitor_status=1
  fi
  assert_no_field_hdf5
  if [[ "$run_status" -ne 0 ]]; then
    echo "ASTR failed with status $run_status; see $log" >&2
    return "$run_status"
  fi
  if [[ "$monitor_status" -ne 0 ]]; then
    return 1
  fi

  grep -q 'The job is done!' "$log"
  grep -q 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log"
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
  python3 "$ROOT_DIR/tests/gpu_validation/summarize_rank_rk_timing.py" \
    --log "$log" --ranks "$NP" --steps "$((MAXSTEP + 1))" \
    --discard "$DISCARD_STEPS" > "$run_dir/rank_rk.tsv"
  awk 'NR>1 {print $2}' "$run_dir/rank_rk.tsv" > "$timing_values"
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
rm -f "$CASE_DIR/datin/grid.h5"
rm -f "$PHASE_SUMMARY"
assert_no_field_hdf5
printf 'np=%s\ntopology=%s\ngpu_ids=%s\nhalo_transport=%s\nfilter_workspace=%s\nbenchmark_no_field_io=1\nphase_timing=%s\n' \
  "$NP" "$TOPOLOGY" "$GPU_IDS" "$HALO_TRANSPORT" "$FILTER_WORKSPACE" "$PHASE_TIMING" \
  > "$OUT_DIR/${LABEL}_transport_metadata.txt"
printf 'label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent\n' > "$TIMINGS"

run_once warmup f
for repeat in $(seq 1 "$REPEATS"); do
  run_once "$repeat" t
done

if [[ "$PHASE_TIMING" == "1" ]]; then
  phase_args=()
  for repeat in $(seq 1 "$REPEATS"); do
    phase_args+=(--log "$OUT_DIR/${LABEL}_run_${repeat}/run.log")
  done
  python3 "$ROOT_DIR/tests/gpu_validation/summarize_gpu_phase_timing.py" \
    "${phase_args[@]}" --ranks "$NP" --steps "$((MAXSTEP + 1))" \
    --output "$PHASE_SUMMARY"
fi

python3 "$ROOT_DIR/tests/gpu_validation/summarize_tgv_performance.py" \
  --timings "$TIMINGS" --summary "$SUMMARY" --grid "$GRID" \
  --maxstep "$MAXSTEP" --discard-steps "$DISCARD_STEPS" \
  --sync-mode "$SYNC_MODE"
