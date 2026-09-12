#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
GPU_ID="${GPU_ID:-0}"
GRID="${GRID:-64,64,32}"
TOPOLOGY="1,1,1"
FEQAVG=1
REPEATS="${REPEATS:-5}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
MEASURED_STEPS="${MEASURED_STEPS:-8}"
MAX_OVERHEAD_PERCENT="${MAX_OVERHEAD_PERCENT:-10.0}"
MODE=''
RESULT_DIR=''

usage() {
  printf 'Usage: %s --mode memcheck|residency|performance --result-dir NEW_PATH\n' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:?--mode requires a value}"
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

case "$MODE" in
  memcheck|residency|performance) ;;
  *)
    usage >&2
    exit 2
    ;;
esac
if [[ -z "$RESULT_DIR" ]]; then
  printf -- '--result-dir is required\n' >&2
  exit 2
fi
if [[ -e "$RESULT_DIR" ]]; then
  printf 'refusing to overwrite compact statistics profile directory: %s\n' "$RESULT_DIR" >&2
  exit 2
fi
if [[ ! -x "$GPU_EXE" ]]; then
  printf 'GPU executable not found: %s\n' "$GPU_EXE" >&2
  exit 2
fi
IFS=',' read -r IM JM KM <<< "$GRID"
if [[ -z "${KM:-}" || "$IM" -lt 16 || "$JM" -lt 16 || "$KM" -lt 8 ]]; then
  printf 'GRID must contain three comma-separated dimensions, at least 16,16,8\n' >&2
  exit 2
fi
mkdir -p "$RESULT_DIR"

GIT_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
EXE_SHA256="$(sha256sum "$GPU_EXE" | awk '{print $1}')"
GPU_MODEL="$(nvidia-smi --id="$GPU_ID" --query-gpu=name --format=csv,noheader | head -1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
{
  printf 'mode=%s\n' "$MODE"
  printf 'git_commit=%s\n' "$GIT_COMMIT"
  printf 'git_dirty=%s\n' "$(git -C "$ROOT_DIR" status --porcelain | wc -l)"
  printf 'executable=%s\n' "$GPU_EXE"
  printf 'executable_sha256=%s\n' "$EXE_SHA256"
  printf 'gpu_model=%s\n' "$GPU_MODEL"
  printf 'gpu_id=%s\n' "$GPU_ID"
  printf 'topology=%s\n' "$TOPOLOGY"
  printf 'grid=%s\n' "$GRID"
  printf 'feqavg=%s\n' "$FEQAVG"
  printf 'warmup_steps=%s\n' "$WARMUP_STEPS"
  printf 'measured_steps=%s\n' "$MEASURED_STEPS"
  printf 'repeats=%s\n' "$REPEATS"
} > "$RESULT_DIR/metadata.txt"

configure_controller() {
  local controller="$1" lavg="$2" maxstep="$3"
  if [[ "$lavg" == t ]]; then
    sed -i 's/^f,f,f,f$/f,f,t,f/' "$controller"
  fi
  sed -i "/^# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg$/{n;s/.*/${maxstep},9999,9999,9999,9999,${FEQAVG}/;}" "$controller"
}

prepare_case() {
  local case_dir="$1" lavg="$2" maxstep="$3"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$case_dir" --use-gpu t \
    --im "$IM" --jm "$JM" --km "$KM" --mach 0.3 \
    --conschm 543e --diffterm t --lfilter f --wall-temperature 1.4 \
    --isobaric-profile --ninit 0 --turbinf prof \
    --maxstep "$maxstep" --feqchkpt 9999 --feqlist 9999 --deltat 1.0e-5
  configure_controller "$case_dir/datin/controller" "$lavg" "$maxstep"
}

run_gpu() {
  local case_dir="$1" log="$2"
  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_sharedfp=individual \
      OMPI_MCA_pml=ob1 OMPI_MCA_osc=pt2pt OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      mpirun -np 1 "$GPU_EXE" run datin/input.flatplate > "$log" 2>&1
  )
  grep -q 'The job is done!' "$case_dir/$log"
}

run_memcheck() {
  local case_dir="$RESULT_DIR/case"
  command -v compute-sanitizer >/dev/null 2>&1 || {
    printf 'compute-sanitizer is required\n' >&2
    exit 127
  }
  prepare_case "$case_dir" t 1
  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_sharedfp=individual \
      OMPI_MCA_pml=ob1 OMPI_MCA_osc=pt2pt OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      mpirun -np 1 compute-sanitizer --tool memcheck --leak-check full \
        --error-exitcode 99 "$GPU_EXE" run datin/input.flatplate \
        > "$RESULT_DIR/memcheck.log" 2>&1
  )
  grep -q 'The job is done!' "$RESULT_DIR/memcheck.log"
  grep -q 'ERROR SUMMARY: 0 errors' "$RESULT_DIR/memcheck.log"
  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_sharedfp=individual \
      OMPI_MCA_pml=ob1 OMPI_MCA_osc=pt2pt OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      mpirun -np 1 compute-sanitizer --tool racecheck --error-exitcode 99 \
        "$GPU_EXE" run datin/input.flatplate > "$RESULT_DIR/racecheck.log" 2>&1
  )
  grep -q 'The job is done!' "$RESULT_DIR/racecheck.log"
  grep -Fq 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' \
    "$RESULT_DIR/racecheck.log"
  printf 'COMPACT_STATISTICS_MEMCHECK_PASS\n'
}

run_residency() {
  local case_dir="$RESULT_DIR/case"
  local report_base="$RESULT_DIR/compact_statistics_residency"
  command -v nsys >/dev/null 2>&1 || {
    printf 'nsys is required\n' >&2
    exit 127
  }
  prepare_case "$case_dir" t 3
  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_sharedfp=individual \
      OMPI_MCA_pml=ob1 OMPI_MCA_osc=pt2pt OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
        --force-overwrite=true -o "$report_base" \
        mpirun -np 1 "$GPU_EXE" run datin/input.flatplate \
        > "$RESULT_DIR/residency.log" 2>&1
  )
  grep -q 'The job is done!' "$RESULT_DIR/residency.log"
  nsys export --type=sqlite --force-overwrite=true \
    --output="$RESULT_DIR/compact_statistics_residency.sqlite" \
    "$report_base.nsys-rep"
  python3 "$ROOT_DIR/tests/gpu_validation/analyze_nsys_rk_residency.py" \
    --input "$RESULT_DIR/compact_statistics_residency.sqlite" \
    --start-kernel compact_plane_moment_kernel --large-transfer-bytes 65536 \
    --report "$RESULT_DIR/residency_report.txt"
  nsys stats --force-export=true \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum --format table \
    "$report_base.nsys-rep" > "$RESULT_DIR/nsys_summary.txt"
  printf 'COMPACT_STATISTICS_RESIDENCY_PASS\n'
}

timed_run() {
  local label="$1" repeat="$2" case_dir="$3" record="$4"
  local log="$RESULT_DIR/${label}_${repeat}.log"
  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMPI_MCA_sharedfp=individual \
      OMPI_MCA_pml=ob1 OMPI_MCA_osc=pt2pt OMPI_MCA_btl=self,vader,tcp \
      OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_opal_cuda_support=0 UCX_MEMTYPE_CACHE=n \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_GPU_COMPLETE_STEP_TIMING=1 \
      mpirun -np 1 "$GPU_EXE" run datin/input.flatplate > "$log" 2>&1
  )
  grep -q 'The job is done!' "$log"
  if [[ "$record" == t ]]; then
    python3 - "$label" "$repeat" "$WARMUP_STEPS" "$MEASURED_STEPS" "$log" >> "$RESULT_DIR/timings.tsv" <<'PY'
import statistics
import sys
from pathlib import Path

label, repeat, discard, expected_last, log = sys.argv[1:]
discard = int(discard)
expected_last = int(expected_last)
values = []
for line in Path(log).read_text(encoding="utf-8", errors="replace").splitlines():
    fields = line.split()
    if fields and fields[0] == "ASTR_GPU_COMPLETE_STEP_TIMING":
        step = int(fields[1])
        if step >= discard:
            values.append(float(fields[2]))
if len(values) != expected_last - discard + 1:
    raise SystemExit(f"unexpected complete-step timing count: {len(values)}")
print(f"{label}\t{repeat}\t{len(values)}\t{statistics.median(values):.16e}")
PY
  fi
}

run_performance() {
  local off_dir="$RESULT_DIR/statistics_off" on_dir="$RESULT_DIR/statistics_on"
  if [[ "$REPEATS" -lt 5 ]]; then
    printf 'REPEATS must be at least 5\n' >&2
    exit 2
  fi
  if [[ "$WARMUP_STEPS" -lt 1 || "$WARMUP_STEPS" -gt "$MEASURED_STEPS" ]]; then
    printf 'WARMUP_STEPS must lie in [1, MEASURED_STEPS]\n' >&2
    exit 2
  fi
  prepare_case "$off_dir" f "$MEASURED_STEPS"
  prepare_case "$on_dir" t "$MEASURED_STEPS"
  printf 'label\trepeat\tstep_samples\tmedian_complete_step_seconds\n' > "$RESULT_DIR/timings.tsv"
  timed_run statistics_off warmup "$off_dir" f
  timed_run statistics_on warmup "$on_dir" f
  for repeat in $(seq 1 "$REPEATS"); do
    timed_run statistics_off "$repeat" "$off_dir" t
    timed_run statistics_on "$repeat" "$on_dir" t
  done
  python3 - "$RESULT_DIR/timings.tsv" "$RESULT_DIR/performance_report.txt" "$MAX_OVERHEAD_PERCENT" <<'PY'
import statistics
import sys
from pathlib import Path

timings_path, report_path, limit = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
groups = {"statistics_off": [], "statistics_on": []}
for line in timings_path.read_text(encoding="ascii").splitlines()[1:]:
    label, _, _, value = line.split("\t")
    groups[label].append(float(value))
off = statistics.median(groups["statistics_off"])
on = statistics.median(groups["statistics_on"])
overhead = 100.0 * (on / off - 1.0)
passed = overhead <= limit
lines = [
    f"status: {'pass' if passed else 'fail'}",
    f"statistics_off_median_complete_step_seconds: {off:.16e}",
    f"statistics_on_median_complete_step_seconds: {on:.16e}",
    f"overhead_percent: {overhead:.8f}",
    f"maximum_overhead_percent: {limit:.8f}",
]
report_path.write_text("\n".join(lines) + "\n", encoding="ascii")
print("\n".join(lines))
raise SystemExit(0 if passed else 1)
PY
  printf 'COMPACT_STATISTICS_PERFORMANCE_PASS\n'
}

case "$MODE" in
  memcheck) run_memcheck ;;
  residency) run_residency ;;
  performance) run_performance ;;
esac
