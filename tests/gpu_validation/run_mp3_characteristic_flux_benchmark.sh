#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp3_characteristic_flux_benchmark_$STAMP}"
[[ "$OUT_DIR" == /* ]] || OUT_DIR="$PWD/$OUT_DIR"
[[ "$GPU_EXE" == /* ]] || GPU_EXE="$PWD/$GPU_EXE"
GRID="${GRID:-400,16,16}"
MAXSTEP="${MAXSTEP:-5}"
REPEATS="${REPEATS:-5}"
DISCARD_STEPS="${DISCARD_STEPS:-1}"
GPU_ID="${GPU_ID:-0}"
CASE_DIR="$OUT_DIR/case"
FP64_TIMINGS="$OUT_DIR/fp64_timings.tsv"
CANDIDATE_TIMINGS="$OUT_DIR/characteristic_flux_timings.tsv"

[[ -x "$GPU_EXE" ]] || { echo "GPU_EXE is not executable: $GPU_EXE" >&2; exit 2; }
[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite benchmark evidence directory: $OUT_DIR" >&2; exit 2; }
(( REPEATS >= 5 )) || { echo "REPEATS must be at least 5" >&2; exit 2; }
(( DISCARD_STEPS >= 0 && DISCARD_STEPS < MAXSTEP + 1 )) || {
  echo "DISCARD_STEPS must retain at least one RK timing" >&2; exit 2;
}

mkdir -p "$OUT_DIR"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Shuosher" \
  --dst-case "$CASE_DIR" --input-name input.shuosher --flowtype shuosher \
  --homogeneous t,t,t --bctype 1,1,1,1,1,1 --use-gpu t \
  --grid "$GRID" --maxstep "$MAXSTEP" --feqchkpt 9999 \
  --lfilter f --diffterm f --scheme 643e --conschm 543e --difschm 643e \
  --recon-schem 3 --lchardecomp t --deltat 1.d-4

header='label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent'
printf '%b\n' "$header" > "$FP64_TIMINGS"
printf '%b\n' "$header" > "$CANDIDATE_TIMINGS"

run_once() {
  local mode="$1" candidate="$2" repeat="$3" record="$4"
  local label timings run_dir log monitor stop monitor_pid start end status
  local timing_count timing_values retained_count wall max_memory max_util
  label="$mode"
  timings="$FP64_TIMINGS"
  if [[ "$mode" == mixed_workspace ]]; then
    label=characteristic_flux
    timings="$CANDIDATE_TIMINGS"
  fi
  run_dir="$OUT_DIR/${label}_run_${repeat}"
  log="$run_dir/run.log"
  monitor="$run_dir/gpu_monitor.csv"
  stop="$run_dir/.monitor_stop"
  mkdir -p "$run_dir"
  (
    while [[ ! -e "$stop" ]]; do
      nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,utilization.gpu \
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
      ASTR_GPU_PRECISION_MODE="$mode" ASTR_GPU_MIXED_CANDIDATE="$candidate" \
      ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      OMPI_MCA_coll='^hcoll,ucc' OMPI_MCA_pml=ob1 \
      OMPI_MCA_btl=self OMPI_MCA_osc=pt2pt \
      mpirun -np 1 "$GPU_EXE" run datin/input.shuosher > "$log" 2>&1
  )
  status=$?
  set -e
  end="$(date +%s.%N)"
  touch "$stop"
  wait "$monitor_pid"
  (( status == 0 )) || { echo "ASTR failed with status $status; see $log" >&2; return "$status"; }
  rg -q 'The job is done!' "$log"
  timing_count="$(awk '$1 == "ASTR_GPU_RK_TIMING" {count++} END {print count+0}' "$log")"
  [[ "$timing_count" -eq $((MAXSTEP + 1)) ]] || {
    echo "unexpected RK timing count in $log: $timing_count" >&2; exit 1;
  }
  [[ "$record" == t ]] || return 0
  timing_values="$run_dir/rk_seconds.txt"
  python3 "$ROOT_DIR/tests/gpu_validation/summarize_rank_rk_timing.py" \
    --log "$log" --ranks 1 --steps "$((MAXSTEP + 1))" \
    --discard "$DISCARD_STEPS" > "$run_dir/rank_rk.tsv"
  awk 'NR>1 {print $2}' "$run_dir/rank_rk.tsv" > "$timing_values"
  retained_count="$(wc -l < "$timing_values")"
  wall="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$end")"
  max_memory="$(awk -F',' '{gsub(/ /,"",$1); if ($1+0>m) m=$1+0} END {print m+0}' "$monitor")"
  max_util="$(awk -F',' '{gsub(/ /,"",$2); if ($2+0>m) m=$2+0} END {print m+0}' "$monitor")"
  python3 - "$label" "$repeat" "$retained_count" "$wall" "$max_memory" "$max_util" "$timing_values" >> "$timings" <<'PY'
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

run_once fp64 flux warmup f
run_once mixed_workspace characteristic_flux warmup f
for repeat in $(seq 1 "$REPEATS"); do
  if (( repeat % 2 == 1 )); then
    run_once fp64 flux "$repeat" t
    run_once mixed_workspace characteristic_flux "$repeat" t
  else
    run_once mixed_workspace characteristic_flux "$repeat" t
    run_once fp64 flux "$repeat" t
  fi
done

mixed_bytes="$(awk -F= '/^ASTR_GPU_MIXED_WORKSPACE_BYTES=/ {print $2; exit}' "$OUT_DIR/characteristic_flux_run_1/run.log")"
fp64_bytes="$(awk -F= '/^ASTR_GPU_FP64_WORKSPACE_BYTES=/ {print $2; exit}' "$OUT_DIR/characteristic_flux_run_1/run.log")"
workspace_elements="$(python3 -c "i,j,k=map(int,'$GRID'.split(',')); h=5; print((i+2*h+1)*(j+2*h+1)*(k+2*h+1)*5)")"
expected_mixed_bytes="$((workspace_elements * 4))"
expected_fp64_bytes="$((workspace_elements * 8))"
python3 "$ROOT_DIR/tests/gpu_validation/summarize_mp2_benchmark.py" \
  --fp64 "$FP64_TIMINGS" --candidate "$CANDIDATE_TIMINGS" \
  --fp64-workspace-bytes "$fp64_bytes" --mixed-workspace-bytes "$mixed_bytes" \
  --expected-fp64-workspace-bytes "$expected_fp64_bytes" \
  --expected-mixed-workspace-bytes "$expected_mixed_bytes" \
  --phase MP3 --report "$OUT_DIR/summary.md"
