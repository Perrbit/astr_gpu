#!/bin/bash
#SBATCH --job-name=astr-a800-tgv-scale
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-tgv-scale-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-tgv-scale-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
EXPECTED_SOURCE_COMMIT=27447959e3d50ea93ad10d4dc708e7f6bec6a138
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_scaling_2744795}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$WORK_ROOT/build_gpu_scaling_2744795}"
BENCH_DIR="${BENCH_DIR:-$WORK_ROOT/a800_tgv_scaling_matrix_20260913}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_tgv_scaling_matrix_${SLURM_JOB_ID:-dryrun}}"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
MANIFEST="$RUN_ROOT/matrix_manifest.tsv"
FILTER_WORKSPACE=full
HALO_TRANSPORT=pinned-overlap
SYNC_MODE=explicit
MAXSTEP=21
DISCARD_STEPS=2
REPEATS=5
DELTAT=1.0e-4
DRY_RUN="${ASTR_SCALING_DRY_RUN:-0}"

matrix=(
  "strong_np1_111 1 1,1,1 512,512,512 512,512,512 strong"
  "strong_np2_211 2 2,1,1 512,512,512 256,512,512 strong"
  "strong_np2_121 2 1,2,1 512,512,512 512,256,512 strong"
  "strong_np2_112 2 1,1,2 512,512,512 512,512,256 strong"
  "strong_np4_411 4 4,1,1 512,512,512 128,512,512 strong"
  "strong_np4_141 4 1,4,1 512,512,512 512,128,512 strong"
  "strong_np4_114 4 1,1,4 512,512,512 512,512,128 strong"
  "strong_np4_221 4 2,2,1 512,512,512 256,256,512 strong"
  "strong_np4_212 4 2,1,2 512,512,512 256,512,256 strong"
  "strong_np4_122 4 1,2,2 512,512,512 512,256,256 strong"
  "weak_np1_111 1 1,1,1 256,256,256 256,256,256 weak"
  "weak_np2_211 2 2,1,1 512,256,256 256,256,256 weak"
  "weak_np2_121 2 1,2,1 256,512,256 256,256,256 weak"
  "weak_np2_112 2 1,1,2 256,256,512 256,256,256 weak"
  "weak_np4_411 4 4,1,1 1024,256,256 256,256,256 weak"
  "weak_np4_141 4 1,4,1 256,1024,256 256,256,256 weak"
  "weak_np4_114 4 1,1,4 256,256,1024 256,256,256 weak"
  "weak_np4_221 4 2,2,1 512,512,256 256,256,256 weak"
  "weak_np4_212 4 2,1,2 512,256,512 256,256,256 weak"
  "weak_np4_122 4 1,2,2 256,512,512 256,256,256 weak"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'A800 TGV scaling matrix dry-run\n'
  printf 'source=%s\nbuild=%s\nbenchmark_control=%s\nrun_root=%s\n' \
    "$SRC_DIR" "$GPU_BUILD_DIR" "$BENCH_DIR" "$RUN_ROOT"
  printf 'filter_workspace=%s\nhalo_transport=%s\nsync_mode=%s\n' \
    "$FILTER_WORKSPACE" "$HALO_TRANSPORT" "$SYNC_MODE"
  printf 'maxstep=%s\ndiscard_steps=%s\nrepeats=%s\ndeltat=%s\n' \
    "$MAXSTEP" "$DISCARD_STEPS" "$REPEATS" "$DELTAT"
  printf 'case=%s\n' "${matrix[@]}"
  exit 0
fi

for path in "$SRC_DIR" "$GPU_BUILD_DIR" "$BENCH_DIR" "$HDF5_ROOT" "$PYTHON_EXE" "$RUN_ROOT"; do
  case "$path/" in
    "$WORK_ROOT"/*) ;;
    *) printf 'path escapes approved work root: %s\n' "$path" >&2; exit 2 ;;
  esac
done

module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$(dirname "$PYTHON_EXE"):$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_coll='^hcoll,ucc'
export OMPI_MCA_opal_cuda_support=0
export UCX_MEMTYPE_CACHE=n

GPU_EXE="$GPU_BUILD_DIR/bin/astr"
for required in "$GPU_EXE" "$GPU_BUILD_DIR/compile_commands.json" "$PYTHON_EXE" \
  "$BENCH_DIR/summarize_tgv_scaling_matrix.py"; do
  [[ -f "$required" ]] || { printf 'missing benchmark input: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$GPU_EXE" && -x "$PYTHON_EXE" ]] || { printf 'required executable is not executable\n' >&2; exit 2; }
[[ -d "$SRC_DIR/.git" ]] || { printf 'source checkout not found: %s\n' "$SRC_DIR" >&2; exit 2; }
[[ "$(git -C "$SRC_DIR" rev-parse HEAD)" == "$EXPECTED_SOURCE_COMMIT" ]] || {
  printf 'source commit does not match benchmark contract\n' >&2
  exit 2
}
[[ -z "$(git -C "$SRC_DIR" status --short)" ]] || {
  printf 'source checkout is not clean\n' >&2
  git -C "$SRC_DIR" status --short >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite run root: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/tmp"
export TMPDIR="$RUN_ROOT/tmp"

ldd "$GPU_EXE" > "$RUN_ROOT/gpu_dependencies.txt" 2>&1
if grep -q 'not found' "$RUN_ROOT/gpu_dependencies.txt"; then
  printf 'unresolved compute-node dynamic library dependency\n' >&2
  exit 2
fi
if grep -E -- '(--use_fast_math|-Mfprelaxed|(^|[[:space:]])-fast([[:space:]]|$))' \
  "$GPU_BUILD_DIR/compile_commands.json" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs, got: %s\n' "$visible" >&2; exit 2; }
gpu1="${gpu[0]}"
gpu2="${gpu[0]},${gpu[1]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

{
  printf 'source_commit: %s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  printf 'source_status: clean\n'
  printf 'gpu_executable_sha256: '
  sha256sum "$GPU_EXE"
  printf 'batch_script_sha256: '
  sha256sum "$BENCH_DIR/run_zhongke_a800_tgv_scaling_matrix.sbatch"
  printf 'summary_script_sha256: '
  sha256sum "$BENCH_DIR/summarize_tgv_scaling_matrix.py"
  printf 'date: %s\n' "$(date -Iseconds)"
  printf 'hostname: %s\n' "$(hostname)"
  printf 'slurm_job_id: %s\n' "$SLURM_JOB_ID"
  printf 'slurm_job_nodelist: %s\n' "$SLURM_JOB_NODELIST"
  printf 'cuda_visible_devices: %s\n' "$visible"
  printf 'precision: FP64\nfilter_workspace: %s\n' "$FILTER_WORKSPACE"
  printf 'halo_transport: %s\nsync_mode: %s\n' "$HALO_TRANSPORT" "$SYNC_MODE"
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
  printf '\npython:\n'
  "$PYTHON_EXE" --version
} > "$RUN_ROOT/environment.txt"

printf 'case\tstatus\texit_code\tstarted\tfinished\tdetail\n' > "$STATUS_FILE"
printf 'label\tkind\tnp\ttopology\tgrid\tlocal_grid\n' > "$MANIFEST"
for entry in "${matrix[@]}"; do
  read -r label np topology grid local_grid kind <<< "$entry"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$kind" "$np" "$topology" "$grid" "$local_grid" >> "$MANIFEST"
done

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
  return 0
}

gpu_ids_for_np() {
  case "$1" in
    1) printf '%s\n' "$gpu1" ;;
    2) printf '%s\n' "$gpu2" ;;
    4) printf '%s\n' "$gpu4" ;;
    *) return 2 ;;
  esac
}

verify_full_workspace() {
  local out="$1" count=0 log
  while IFS= read -r -d '' log; do
    count=$((count + 1))
    grep -aq 'GPU filter workspace storage: full' "$log" || {
      printf 'expected full GPU filter workspace in %s\n' "$log" >&2
      return 1
    }
  done < <(find "$out" -type f -name run.log -print0)
  [[ "$count" -eq 6 ]] || {
    printf 'expected six run logs under %s, found %s\n' "$out" "$count" >&2
    return 1
  }
}

run_timing() {
  local label="$1" np="$2" topology="$3" grid="$4" gpu_ids
  gpu_ids="$(gpu_ids_for_np "$np")" || return 2
  PYTHON_EXE="$PYTHON_EXE" GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/$label" LABEL="$label" \
    GRID="$grid" MAXSTEP="$MAXSTEP" DISCARD_STEPS="$DISCARD_STEPS" REPEATS="$REPEATS" \
    DELTAT="$DELTAT" NP="$np" TOPOLOGY="$topology" GPU_IDS="$gpu_ids" \
    FILTER_WORKSPACE="$FILTER_WORKSPACE" HALO_TRANSPORT="$HALO_TRANSPORT" \
    SYNC_MODE="$SYNC_MODE" FEQCHKPT=9999 \
    "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh" || return $?
  verify_full_workspace "$RUN_ROOT/$label"
}

for entry in "${matrix[@]}"; do
  read -r label np topology grid local_grid kind <<< "$entry"
  run_case "$label" run_timing "$label" "$np" "$topology" "$grid"
done

"$PYTHON_EXE" "$BENCH_DIR/summarize_tgv_scaling_matrix.py" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT" \
  --output-tsv "$RUN_ROOT/scaling_results.tsv" \
  --summary "$RUN_ROOT/scaling_summary.md" > "$RUN_ROOT/scaling_summary.stdout" 2>&1
summary_status=$?

pass_count="$(awk -F '\t' 'NR>1 && $2=="PASS" {count++} END {print count+0}' "$STATUS_FILE")"
fail_count="$(awk -F '\t' 'NR>1 && $2=="FAIL" {count++} END {print count+0}' "$STATUS_FILE")"
{
  printf 'summary_status=%s\n' "$summary_status"
  printf 'expected_cases=%s\n' "${#matrix[@]}"
  printf 'passed_cases=%s\n' "$pass_count"
  printf 'failed_cases=%s\n' "$fail_count"
  if [[ "$summary_status" -eq 0 && "$pass_count" -eq "${#matrix[@]}" ]]; then
    printf 'overall_status=PASS\n'
  else
    printf 'overall_status=PARTIAL\n'
  fi
} > "$RUN_ROOT/job_status.txt"

printf 'A800 TGV scaling matrix complete: pass=%s fail=%s summary=%s root=%s\n' \
  "$pass_count" "$fail_count" "$summary_status" "$RUN_ROOT"
exit 0
