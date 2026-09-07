#!/usr/bin/env bash
set -euo pipefail

P2_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$P2_ROOT_DIR/tests/gpu_validation/p2_shock_case_common.sh"

CASE="${CASE:-shuosher}"
p2_validate_case "$CASE"
GPU_EXE="${GPU_EXE:-$P2_ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$P2_ROOT_DIR/tests/gpu_validation/out/p2_${CASE}_performance}"
LABEL="${LABEL:-p2_${CASE}_baseline}"
GRID="${GRID:-$(p2_default_grid "$CASE")}"
MAXSTEP="${MAXSTEP:-20}"
REPEATS="${REPEATS:-5}"
DISCARD_STEPS="${DISCARD_STEPS:-1}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-$GPU_ID}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
FEQCHKPT="${FEQCHKPT:-9999}"
FEQLIST="${FEQLIST:-9999}"
SYNC_MODE="${SYNC_MODE:-explicit}"

if [[ "$GPU_EXE" != /* ]]; then GPU_EXE="$P2_ROOT_DIR/$GPU_EXE"; fi
if [[ "$OUT_DIR" != /* ]]; then OUT_DIR="$P2_ROOT_DIR/$OUT_DIR"; fi
[[ -x "$GPU_EXE" ]] || { printf 'GPU executable not found: %s\n' "$GPU_EXE" >&2; exit 2; }
[[ "$REPEATS" -ge 5 ]] || { printf 'REPEATS must be at least 5\n' >&2; exit 2; }
[[ "$MAXSTEP" -ge 1 ]] || { printf 'MAXSTEP must be positive\n' >&2; exit 2; }
[[ "$DISCARD_STEPS" -ge 0 && "$DISCARD_STEPS" -lt $((MAXSTEP + 1)) ]] || {
  printf 'DISCARD_STEPS must retain at least one RK sample\n' >&2; exit 2;
}
[[ "$FEQCHKPT" -gt "$MAXSTEP" ]] || { printf 'FEQCHKPT must exceed MAXSTEP\n' >&2; exit 2; }
[[ "$FEQLIST" -gt "$MAXSTEP" ]] || { printf 'FEQLIST must exceed MAXSTEP\n' >&2; exit 2; }
[[ "$SYNC_MODE" == "explicit" ]] || { printf 'P2 requires explicit synchronization\n' >&2; exit 2; }
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

TIMINGS="$OUT_DIR/${LABEL}_timings.tsv"
SUMMARY="$OUT_DIR/${LABEL}_summary.md"
CASE_DIR="$OUT_DIR/${LABEL}_case"
mkdir -p "$OUT_DIR" "$OUT_DIR/tmp"
[[ ! -e "$TIMINGS" ]] || { printf 'refusing to overwrite %s\n' "$TIMINGS" >&2; exit 2; }
printf 'np=%s\ntopology=%s\ngpu_ids=%s\n' "$NP" "$TOPOLOGY" "$GPU_IDS" > "$OUT_DIR/transport_metadata.txt"
sha256sum "$GPU_EXE" > "$OUT_DIR/executable.sha256"
p2_prepare_case "$CASE" "$CASE_DIR" "$GRID" "$MAXSTEP" "$FEQCHKPT" "$FEQLIST"

run_once() {
  local repeat="$1" record="$2"
  local run_dir log monitor stop monitor_pid start end status
  local timing_count retained_count timing_values wall max_memory max_util
  run_dir="$OUT_DIR/${LABEL}_run_${repeat}"
  log="$run_dir/run.log"
  monitor="$run_dir/gpu_monitor.csv"
  stop="$run_dir/.monitor_stop"
  mkdir -p "$run_dir"
  rm -f "$stop"
  (
    while [[ ! -e "$stop" ]]; do
      nvidia-smi --id="$GPU_IDS" --query-gpu=memory.used,utilization.gpu,index,timestamp \
        --format=csv,noheader,nounits
      sleep 0.1
    done
  ) > "$monitor" &
  monitor_pid=$!
  start="$(date +%s.%N)"
  set +e
  (
    cd "$CASE_DIR"
    env "${P2_RUNTIME_ENV[@]}" TMPDIR="$OUT_DIR/tmp" \
      OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      ASTR_GPU_SYNC_MODE="$SYNC_MODE" \
      mpirun -np "$NP" "$GPU_EXE" run "$P2_INPUT" > "$log" 2>&1
  )
  status=$?
  set -e
  end="$(date +%s.%N)"
  touch "$stop"
  wait "$monitor_pid"
  [[ "$status" -eq 0 ]] || { printf 'ASTR failed; see %s\n' "$log" >&2; return "$status"; }
  grep -q 'The job is done!' "$log"
  if grep -Eq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    printf 'non-finite or crash marker found in %s\n' "$log" >&2
    return 1
  fi
  timing_count="$(awk '$1 == "ASTR_GPU_RK_TIMING" {count++} END {print count+0}' "$log")"
  [[ "$timing_count" -eq $((MAXSTEP + 1)) ]] || {
    printf 'expected %d RK timings, found %d in %s\n' "$((MAXSTEP + 1))" "$timing_count" "$log" >&2
    return 1
  }
  timing_values="$run_dir/rk_seconds.txt"
  python3 "$P2_ROOT_DIR/tests/gpu_validation/summarize_rank_rk_timing.py" \
    --log "$log" --ranks "$NP" --steps "$((MAXSTEP + 1))" \
    --discard "$DISCARD_STEPS" > "$run_dir/rank_rk.tsv"
  awk 'NR>1 {print $2}' "$run_dir/rank_rk.tsv" > "$timing_values"
  [[ "$record" == t ]] || return 0
  retained_count="$(wc -l < "$timing_values")"
  wall="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$end")"
  max_memory="$(awk -F, '{gsub(/ /,"",$1); if ($1+0>m) m=$1+0} END {print m+0}' "$monitor")"
  max_util="$(awk -F, '{gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$monitor")"
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
    f"{min(values):.12f}\t{max(values):.12f}\t{float(wall):.6f}\t{memory}\t{utilization}"
)
PY
}

printf 'label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent\n' > "$TIMINGS"
run_once warmup f
for repeat in $(seq 1 "$REPEATS"); do run_once "$repeat" t; done

python3 "$P2_ROOT_DIR/tests/gpu_validation/summarize_gpu_rk_performance.py" \
  --timings "$TIMINGS" --summary "$SUMMARY" --case-name "$CASE" \
  --grid "$GRID" --maxstep "$MAXSTEP" --discard-steps "$DISCARD_STEPS" \
  --sync-mode "$SYNC_MODE"
