#!/bin/bash
#SBATCH --job-name=astr-a800-tgv-all
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-tgv-all-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-tgv-all-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_tgv_campaign}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-$WORK_ROOT/build_cpu_tgv_campaign}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$WORK_ROOT/build_gpu_tgv_campaign}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_tgv_campaign_${SLURM_JOB_ID:-dryrun}}"
DLR_REFERENCE="$SRC_DIR/documents/reference_data/tgv/spectral_Re1600_512.gdiag"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
DRY_RUN="${ASTR_CAMPAIGN_DRY_RUN:-0}"

case_labels=(
  t0_cfl_memory_preflight
  t1_256_cpu_np1
  t1_256_gpu_np1
  t2_512_gpu_np1
  t3_512_gpu_np2_z
  t4_512_gpu_np4_z
  t5_512_gpu_np4_pageable
  t6_512_gpu_np4_nsys
  t7_512_gpu_np4_t20
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'TGV campaign dry-run\n'
  printf 'work_root=%s\nsource=%s\nrun_root=%s\n' "$WORK_ROOT" "$SRC_DIR" "$RUN_ROOT"
  printf 'target_cfl=0.50\ntarget_time=20.0\nsegment_steps=2000\n'
  printf 'case=%s\n' "${case_labels[@]}"
  exit 0
fi

for path in "$SRC_DIR" "$CPU_BUILD_DIR" "$GPU_BUILD_DIR" "$HDF5_ROOT" "$RUN_ROOT"; do
  case "$path/" in
    "$WORK_ROOT"/*) ;;
    *) printf 'path escapes approved work root: %s\n' "$path" >&2; exit 2 ;;
  esac
done

module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"

CPU_EXE="$CPU_BUILD_DIR/bin/astr"
GPU_EXE="$GPU_BUILD_DIR/bin/astr"
for required in "$CPU_EXE" "$GPU_EXE" "$DLR_REFERENCE" "$GPU_BUILD_DIR/compile_commands.json" "$PYTHON_EXE"; do
  [[ -f "$required" ]] || { printf 'missing campaign input: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$CPU_EXE" && -x "$GPU_EXE" ]] || { printf 'ASTR executables are not executable\n' >&2; exit 2; }
[[ -d "$SRC_DIR/.git" ]] || { printf 'source checkout not found: %s\n' "$SRC_DIR" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite run root: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"
mkdir -p "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"
printf 'case\tstatus\texit_code\tstarted\tfinished\tdetail\n' > "$STATUS_FILE"

visible="${CUDA_VISIBLE_DEVICES:?SLURM did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs, got: %s\n' "$visible" >&2; exit 2; }
gpu1="${gpu[0]}"
gpu2="${gpu[0]},${gpu[1]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

{
  printf 'source_commit: %s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  printf 'source_status:\n'
  git -C "$SRC_DIR" status --short
  printf 'cpu_executable_sha256: '
  sha256sum "$CPU_EXE"
  printf 'gpu_executable_sha256: '
  sha256sum "$GPU_EXE"
  printf 'dlr_reference_sha256: '
  sha256sum "$DLR_REFERENCE"
  printf 'date: %s\n' "$(date -Iseconds)"
  printf 'hostname: %s\n' "$(hostname)"
  printf 'slurm_job_id: %s\n' "$SLURM_JOB_ID"
  printf 'slurm_job_nodelist: %s\n' "$SLURM_JOB_NODELIST"
  printf 'cuda_visible_devices: %s\n' "$visible"
  printf '\nloaded_modules:\n'
  module -t list 2>&1
  printf '\nnvidia_smi:\n'
  nvidia-smi
  printf '\nnvidia_topology:\n'
  nvidia-smi topo -m
  printf '\nnvfortran:\n'
  nvfortran --version
  printf '\nmpirun:\n'
  mpirun --version
  printf '\nhdf5:\n'
  h5pfc -showconfig
} > "$RUN_ROOT/environment.txt"

if grep -E -- '(--use_fast_math|-Mfprelaxed|(^|[[:space:]])-fast([[:space:]]|$))' \
  "$GPU_BUILD_DIR/compile_commands.json" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi

record_case() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" >> "$STATUS_FILE"
}

run_case() {
  local label="$1"
  shift
  local started finished status
  started="$(date -Iseconds)"
  printf 'CASE_START %s %s\n' "$label" "$started"
  "$@"
  status=$?
  finished="$(date -Iseconds)"
  if (( status == 0 )); then
    record_case "$label" PASS 0 "$started" "$finished" "$RUN_ROOT/$label"
  else
    record_case "$label" FAIL "$status" "$started" "$finished" "$RUN_ROOT/$label"
  fi
  printf 'CASE_END %s status=%s %s\n' "$label" "$status" "$finished"
  return "$status"
}

skip_case() {
  local label="$1" reason="$2" now
  now="$(date -Iseconds)"
  record_case "$label" SKIP 0 "$now" "$now" "$reason"
  printf 'CASE_SKIP %s reason=%s\n' "$label" "$reason"
}

validate_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log" || return 1
  ! grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"
}

run_cfl_preflight() {
  local out="$RUN_ROOT/t0_cfl_memory_preflight" case_dir="$RUN_ROOT/t0_cfl_memory_preflight/case"
  local log="$out/run.log" monitor="$out/gpu_monitor.csv" stop="$out/.monitor_stop" monitor_pid
  mkdir -p "$out"
  "$PYTHON_EXE" "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid 512,512,512 --maxstep 0 --feqchkpt 9999 --feqlist 1 \
    --deltat 1.0e-4 --lfilter t --diffterm t --scheme 643e || return 2
  (
    while [[ ! -e "$stop" ]]; do
      nvidia-smi --id="$gpu1" --query-gpu=timestamp,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 0.1
    done
  ) > "$monitor" &
  monitor_pid=$!
  (
    cd "$case_dir" || exit 2
    CUDA_VISIBLE_DEVICES="$gpu1" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_HALO_TRANSPORT=pinned-overlap \
      mpirun -np 1 "$GPU_EXE" run datin/input.tgv > "$log" 2>&1
  )
  status=$?
  touch "$stop"
  wait "$monitor_pid"
  (( status == 0 )) || return "$status"
  validate_log "$log" || return 1
  awk '/time step for CFL=1:/ {value=$NF} END {if (value+0 > 0) print value; else exit 1}' \
    "$log" > "$out/cfl_dt_one.txt" || return 1
  awk '/current CFL:/ {seen=1; if ($NF+0 >= 1.0 || $NF+0 <= 0.0) bad=1} END {exit !(seen && !bad)}' "$log"
}

run_end_to_end_repeats() {
  local label="$1" use_gpu="$2" exe="$3" gpu_ids="$4" dt="$5"
  local out="$RUN_ROOT/$label" timings="$RUN_ROOT/$label/wall_timings.tsv"
  local repeat case_dir start end status
  mkdir -p "$out"
  printf 'repeat\twall_seconds\n' > "$timings"
  for repeat in 1 2 3 4 5; do
    case_dir="$out/run_$repeat"
    "$PYTHON_EXE" "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
      --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
      --use-gpu "$use_gpu" --grid 256,256,256 --maxstep 9 --feqchkpt 9999 \
      --feqlist 9999 --deltat "$dt" --lfilter t --diffterm t --scheme 643e || return 2
    start="$(date +%s.%N)"
    (
      cd "$case_dir" || exit 2
      CUDA_VISIBLE_DEVICES="$gpu_ids" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
      ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_HALO_TRANSPORT=pinned-overlap \
        mpirun -np 1 "$exe" run datin/input.tgv > run.log 2>&1
    )
    status=$?
    end="$(date +%s.%N)"
    (( status == 0 )) || return "$status"
    validate_log "$case_dir/run.log" || return 1
    "$PYTHON_EXE" - "$repeat" "$start" "$end" >> "$timings" <<'PY'
import sys
print(f"{sys.argv[1]}\t{float(sys.argv[3]) - float(sys.argv[2]):.9f}")
PY
  done
}

run_gpu_timing() {
  local label="$1" np="$2" topology="$3" gpu_ids="$4" transport="$5" dt="$6"
  GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/$label" LABEL="$label" \
    GRID=512,512,512 MAXSTEP=21 DISCARD_STEPS=2 REPEATS=5 DELTAT="$dt" \
    NP="$np" TOPOLOGY="$topology" GPU_IDS="$gpu_ids" \
    HALO_TRANSPORT="$transport" SYNC_MODE=explicit FEQCHKPT=9999 \
    "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
}

run_nsys() {
  local out="$RUN_ROOT/t6_512_gpu_np4_nsys" case_dir="$RUN_ROOT/t6_512_gpu_np4_nsys/case" dt="$1"
  command -v nsys >/dev/null 2>&1 || return 127
  mkdir -p "$out"
  "$PYTHON_EXE" "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid 512,512,512 --maxstep 4 --feqchkpt 9999 --feqlist 9999 \
    --deltat "$dt" --lfilter t --diffterm t --scheme 643e || return 2
  (
    cd "$case_dir" || exit 2
    CUDA_VISIBLE_DEVICES="$gpu4" ASTR_FORCE_MPI_TOPOLOGY=1,1,4 \
    ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_HALO_TRANSPORT=pinned-overlap \
      nsys profile --trace=cuda,mpi --stats=true --force-overwrite=true \
      -o "$out/tgv_512_np4" mpirun -np 4 "$GPU_EXE" run datin/input.tgv \
      > "$out/nsys.log" 2>&1
  )
  status=$?
  (( status == 0 )) || return "$status"
  validate_log "$out/nsys.log"
}

cfl_ready=f
if run_case t0_cfl_memory_preflight run_cfl_preflight; then
  cfl_ready=t
  cfl_dt_one="$(<"$RUN_ROOT/t0_cfl_memory_preflight/cfl_dt_one.txt")"
  dt512="$("$PYTHON_EXE" -c 'import math,sys; print(f"{math.floor(float(sys.argv[1])*0.5*1e12)/1e12:.12g}")' "$cfl_dt_one")"
  dt256="$("$PYTHON_EXE" -c 'import sys; print(f"{2.0*float(sys.argv[1]):.12g}")' "$dt512")"
fi

if [[ "$cfl_ready" == t ]]; then
  run_case t1_256_cpu_np1 run_end_to_end_repeats t1_256_cpu_np1 f "$CPU_EXE" "$gpu1" "$dt256"
  run_case t1_256_gpu_np1 run_end_to_end_repeats t1_256_gpu_np1 t "$GPU_EXE" "$gpu1" "$dt256"
  run_case t2_512_gpu_np1 run_gpu_timing t2_512_gpu_np1 1 1,1,1 "$gpu1" pinned-overlap "$dt512"
  run_case t3_512_gpu_np2_z run_gpu_timing t3_512_gpu_np2_z 2 1,1,2 "$gpu2" pinned-overlap "$dt512"
  run_case t4_512_gpu_np4_z run_gpu_timing t4_512_gpu_np4_z 4 1,1,4 "$gpu4" pinned-overlap "$dt512"
  run_case t5_512_gpu_np4_pageable run_gpu_timing t5_512_gpu_np4_pageable 4 1,1,4 "$gpu4" pageable "$dt512"
  run_case t6_512_gpu_np4_nsys run_nsys "$dt512"
  run_case t7_512_gpu_np4_t20 env \
    ROOT_DIR="$SRC_DIR" GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/t7_512_gpu_np4_t20" \
    PYTHON_EXE="$PYTHON_EXE" \
    CFL_DT_ONE_FILE="$RUN_ROOT/t0_cfl_memory_preflight/cfl_dt_one.txt" \
    DLR_REFERENCE="$DLR_REFERENCE" GPU_IDS="$gpu4" NP=4 TOPOLOGY=1,1,4 \
    TARGET_TIME=20.0 TARGET_CFL=0.50 SEGMENT_STEPS=2000 MAX_RETRIES=1 \
    SYNC_MODE=explicit HALO_TRANSPORT=pinned-overlap \
    "$SRC_DIR/tests/gpu_validation/run_tgv_512_production.sh"
else
  for label in "${case_labels[@]:1}"; do
    skip_case "$label" t0_cfl_memory_preflight_failed
  done
fi

"$PYTHON_EXE" - "$STATUS_FILE" "$RUN_ROOT/job_status.txt" <<'PY'
import csv
import sys
from pathlib import Path

with Path(sys.argv[1]).open(encoding="ascii", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
counts = {status: sum(row["status"] == status for row in rows) for status in ("PASS", "FAIL", "SKIP")}
overall = "PASS" if counts["FAIL"] == 0 and counts["SKIP"] == 0 else "PARTIAL"
Path(sys.argv[2]).write_text(
    f"status={overall}\npass={counts['PASS']}\nfail={counts['FAIL']}\nskip={counts['SKIP']}\n",
    encoding="ascii",
)
PY

if awk -F '\t' 'NR > 1 && ($2 == "FAIL" || $2 == "SKIP") {found=1} END {exit !found}' "$STATUS_FILE"; then
  exit 1
fi
