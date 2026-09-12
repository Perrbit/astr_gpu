#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/tgv_upwind_flux_pair_benchmark}"
GRID="${GRID:-128,128,128}"
MAXSTEP="${MAXSTEP:-5}"
FEQCHKPT="${FEQCHKPT:-9999}"
REPEATS="${REPEATS:-5}"
DISCARD_STEPS="${DISCARD_STEPS:-1}"
RECON_SCHEM="${RECON_SCHEM:-1}"
PRECISION_MODE="${PRECISION_MODE:-fp64}"
GPU_ID="${GPU_ID:-0}"
CASE_DIR="$OUT_DIR/case"
SPLIT_TIMINGS="$OUT_DIR/split_timings.tsv"
FUSED_TIMINGS="$OUT_DIR/fused_timings.tsv"

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
if [[ "$PRECISION_MODE" != "fp64" && "$PRECISION_MODE" != "mixed_workspace" ]]; then
  echo "PRECISION_MODE must be fp64 or mixed_workspace" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" --dst-case "$CASE_DIR" \
  --input-name input.tgv --flowtype tgv --homogeneous t,t,t \
  --bctype 1,1,1,1,1,1 --use-gpu t --grid "$GRID" \
  --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" --lfilter f --diffterm f \
  --scheme 643e --conschm 543e --difschm 643e \
  --recon-schem "$RECON_SCHEM" --lchardecomp f --deltat 1.d-4

header='label\trepeat\trk_samples\tmedian_rk_seconds\tmin_rk_seconds\tmax_rk_seconds\twall_seconds\tmax_memory_mib\tmax_utilization_percent'
printf '%b\n' "$header" > "$SPLIT_TIMINGS"
printf '%b\n' "$header" > "$FUSED_TIMINGS"

run_once() {
  local pair_mode="$1" repeat="$2" record="$3"
  local run_dir log timing_values timing_count retained_count start end wall timings status
  run_dir="$OUT_DIR/${pair_mode}_run_${repeat}"
  log="$run_dir/run.log"
  timings="$SPLIT_TIMINGS"
  [[ "$pair_mode" == "fused" ]] && timings="$FUSED_TIMINGS"
  mkdir -p "$run_dir"
  start="$(date +%s.%N)"
  set +e
  (
    cd "$CASE_DIR"
    env CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_coll_hcoll_enable=0 \
      ASTR_FORCE_MPI_TOPOLOGY=1,1,1 ASTR_GPU_PRECISION_MODE="$PRECISION_MODE" \
      ASTR_GPU_FLUX_PAIR_MODE="$pair_mode" ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      mpirun -np 1 "$GPU_EXE" run datin/input.tgv > "$log" 2>&1
  )
  status=$?
  set -e
  end="$(date +%s.%N)"
  if [[ "$status" -ne 0 ]]; then
    echo "ASTR failed with status $status; see $log" >&2
    return "$status"
  fi
  rg -q "^ASTR_GPU_FLUX_PAIR_MODE=$pair_mode$" "$log"
  timing_count="$(awk '$1 == "ASTR_GPU_RK_TIMING" {count++} END {print count+0}' "$log")"
  if [[ "$timing_count" -ne $((MAXSTEP + 1)) ]]; then
    echo "expected $((MAXSTEP + 1)) RK timings, found $timing_count in $log" >&2
    exit 1
  fi
  if [[ "$record" != "t" ]]; then
    return 0
  fi

  timing_values="$run_dir/rk_seconds.txt"
  python3 "$ROOT_DIR/tests/gpu_validation/summarize_rank_rk_timing.py" \
    --log "$log" --ranks 1 --steps "$((MAXSTEP + 1))" \
    --discard "$DISCARD_STEPS" > "$run_dir/rank_rk.tsv"
  awk 'NR>1 {print $2}' "$run_dir/rank_rk.tsv" > "$timing_values"
  retained_count="$(wc -l < "$timing_values")"
  wall="$(python3 -c 'import sys; print(float(sys.argv[2])-float(sys.argv[1]))' "$start" "$end")"
  python3 - "$pair_mode" "$repeat" "$retained_count" "$wall" "$timing_values" >> "$timings" <<'PY'
import statistics
import sys
from pathlib import Path

label, repeat, count, wall, path = sys.argv[1:]
values = [float(value) for value in Path(path).read_text(encoding="ascii").split()]
if len(values) != int(count) or not values:
    raise SystemExit("invalid retained RK timing sample count")
print(
    f"{label}\t{repeat}\t{count}\t{statistics.median(values):.12f}\t"
    f"{min(values):.12f}\t{max(values):.12f}\t{float(wall):.6f}\t0\t0"
)
PY
}

run_once split warmup f
run_once fused warmup f
for repeat in $(seq 1 "$REPEATS"); do
  if (( repeat % 2 == 1 )); then
    run_once split "$repeat" t
    run_once fused "$repeat" t
  else
    run_once fused "$repeat" t
    run_once split "$repeat" t
  fi
done

python3 "$ROOT_DIR/tests/gpu_validation/summarize_flux_pair_benchmark.py" \
  --split "$SPLIT_TIMINGS" --fused "$FUSED_TIMINGS" \
  --precision-mode "$PRECISION_MODE" --report "$OUT_DIR/summary.md"
