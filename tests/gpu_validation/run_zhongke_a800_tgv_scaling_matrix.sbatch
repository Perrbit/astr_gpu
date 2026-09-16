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
EXPECTED_SOURCE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_p4_c801eeb}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$WORK_ROOT/build_gpu_p4_c801eeb}"
BENCH_DIR="${BENCH_DIR:-$WORK_ROOT/a800_tgv_gpu_scaling_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_tgv_gpu_scaling_${SLURM_JOB_ID:-dryrun}}"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
MANIFEST="$RUN_ROOT/matrix_manifest.tsv"
FILTER_WORKSPACE=full
PRIMARY_HALO_TRANSPORT=pinned-pipeline
PRIMARY_SYNC_MODE=explicit
MAXSTEP=21
DISCARD_STEPS=2
REPEATS=5
DELTAT=1.0e-4
DRY_RUN="${ASTR_SCALING_DRY_RUN:-0}"
EXPECTED_CASES=17

main_matrix=(
  "strong_np1_111 1 1,1,1 512,512,512 512,512,512 strong pinned explicit"
  "strong_np2_211 2 2,1,1 512,512,512 256,512,512 strong pinned-pipeline explicit"
  "strong_np2_121 2 1,2,1 512,512,512 512,256,512 strong pinned-pipeline explicit"
  "strong_np2_112 2 1,1,2 512,512,512 512,512,256 strong pinned-pipeline explicit"
  "strong_np4_411 4 4,1,1 512,512,512 128,512,512 strong pinned-pipeline explicit"
  "strong_np4_141 4 1,4,1 512,512,512 512,128,512 strong pinned-pipeline explicit"
  "strong_np4_114 4 1,1,4 512,512,512 512,512,128 strong pinned-pipeline explicit"
  "strong_np4_221 4 2,2,1 512,512,512 256,256,512 strong pinned-pipeline explicit"
  "strong_np4_212 4 2,1,2 512,512,512 256,512,256 strong pinned-pipeline explicit"
  "strong_np4_122 4 1,2,2 512,512,512 512,256,256 strong pinned-pipeline explicit"
  "weak_np1_111 1 1,1,1 256,256,256 256,256,256 weak pinned explicit"
  "weak_np2_211 2 2,1,1 512,256,256 256,256,256 weak pinned-pipeline explicit"
  "weak_np4_221 4 2,2,1 512,512,256 256,256,256 weak pinned-pipeline explicit"
)

if [[ "$DRY_RUN" == 1 ]]; then
  printf 'A800 TGV scaling matrix dry-run\n'
  printf 'source=%s\nbuild=%s\nbenchmark_control=%s\nrun_root=%s\n' \
    "$SRC_DIR" "$GPU_BUILD_DIR" "$BENCH_DIR" "$RUN_ROOT"
  printf 'filter_workspace=%s\nprimary_halo_transport=%s\nprimary_sync_mode=%s\n' \
    "$FILTER_WORKSPACE" "$PRIMARY_HALO_TRANSPORT" "$PRIMARY_SYNC_MODE"
  printf 'maxstep=%s\ndiscard_steps=%s\nrepeats=%s\ndeltat=%s\n' \
    "$MAXSTEP" "$DISCARD_STEPS" "$REPEATS" "$DELTAT"
  printf 'main_case=%s\n' "${main_matrix[@]}"
  printf 'control_case=control_np2_pinned_overlap_explicit fastest-strong-np2\n'
  printf 'control_case=control_np2_pipeline_dependency fastest-strong-np2\n'
  printf 'control_case=control_np4_pinned_overlap_explicit fastest-strong-np4\n'
  printf 'control_case=control_np4_pipeline_dependency fastest-strong-np4\n'
  printf 'ASTR_GPU_BENCHMARK_NO_FIELD_IO=1\n'
  printf 'statistics output: compact TGV statistics and timing logs only\n'
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
  "$BENCH_DIR/summarize_tgv_scaling_matrix.py" \
  "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"; do
  [[ -f "$required" ]] || { printf 'missing benchmark input: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$GPU_EXE" && -x "$PYTHON_EXE" && -x "$HDF5_ROOT/bin/h5pfc" ]] || {
  printf 'required executable is not executable\n' >&2
  exit 2
}
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

if ! ldd "$GPU_EXE" > "$RUN_ROOT/gpu_dependencies.txt" 2>&1; then
  printf 'ldd failed for GPU executable\n' >&2
  exit 2
fi
if grep -q 'not found' "$RUN_ROOT/gpu_dependencies.txt"; then
  printf 'unresolved compute-node dynamic library dependency\n' >&2
  exit 2
fi
if ! grep -Fq "$HDF5_ROOT/lib/libhdf5_fortran" "$RUN_ROOT/gpu_dependencies.txt"; then
  printf 'GPU executable does not use the approved HDF5 Fortran runtime\n' >&2
  exit 2
fi
if grep -E -- '(--use_fast_math|-ffast-math|-Ofast|-Mfprelaxed|-Mfpapprox|-gpu=[^[:space:]]*fastmath|(^|[[:space:]])-fast([[:space:]]|$))' \
  "$GPU_BUILD_DIR/compile_commands.json" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi
if ! grep -q -- '-cuda' "$GPU_BUILD_DIR/compile_commands.json" || \
   ! grep -q -- '-O2' "$GPU_BUILD_DIR/compile_commands.json"; then
  printf 'expected CUDA RELEASE compiler options are absent\n' >&2
  exit 2
fi
for input_file in \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/input.tgv" \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/controller" \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/parallel.info"; do
  if LC_ALL=C grep -q $'\r' "$input_file"; then
    printf 'CRLF or bare carriage return found in %s\n' "$input_file" >&2
    exit 2
  fi
done
grep -q 'ASTR_GPU_BENCHMARK_NO_FIELD_IO=1' \
  "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh" || {
  printf 'benchmark driver does not enforce the no-field-I/O contract\n' >&2
  exit 2
}

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs, got: %s\n' "$visible" >&2; exit 2; }
gpu1="${gpu[0]}"
gpu2="${gpu[0]},${gpu[1]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

run_compute_preflight() {
  local out="$RUN_ROOT/compute_preflight" case_dir="$RUN_ROOT/compute_preflight/case"
  local log="$RUN_ROOT/compute_preflight/run.log" status found
  mkdir -p "$out"
  "$PYTHON_EXE" "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid 32,32,32 --maxstep 0 --feqchkpt 9999 --feqlist 9999 \
    --deltat "$DELTAT" --lfilter t --diffterm t --scheme 643e --lreadgrid f || return 2
  rm -f "$case_dir/datin/grid.h5"
  (
    cd "$case_dir" || exit 2
    env OMPI_MCA_sharedfp=individual CUDA_VISIBLE_DEVICES="$gpu4" \
      ASTR_FORCE_MPI_TOPOLOGY=2,2,1 ASTR_GPU_RK_TIMING=1 \
      ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=pinned-pipeline ASTR_GPU_FILTER_WORKSPACE=full \
      mpirun -np 4 "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log" || return 1
  if grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    return 1
  fi
  found="$(find "$case_dir" -type f -name '*.h5' -print)"
  if [[ -n "$found" ]]; then
    printf 'compute preflight generated forbidden HDF5 output:\n%s\n' "$found" >&2
    return 1
  fi
}

if ! run_compute_preflight; then
  printf 'compute-node preflight failed; performance matrix will not run\n' >&2
  exit 2
fi

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
  printf 'primary_halo_transport: %s\nprimary_sync_mode: %s\n' \
    "$PRIMARY_HALO_TRANSPORT" "$PRIMARY_SYNC_MODE"
  printf 'field_hdf5: disabled\nstatistics output: enabled\n'
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
printf 'label\tkind\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\tsync_mode\n' > "$MANIFEST"
for entry in "${main_matrix[@]}"; do
  read -r label np topology grid local_grid kind halo_transport sync_mode <<< "$entry"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$kind" "$np" "$topology" "$grid" "$local_grid" \
    "$halo_transport" "$sync_mode" >> "$MANIFEST"
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
  local label="$1" np="$2" topology="$3" grid="$4" halo_transport="$5"
  local sync_mode="$6" gpu_ids found
  gpu_ids="$(gpu_ids_for_np "$np")" || return 2
  PYTHON_EXE="$PYTHON_EXE" GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/$label" LABEL="$label" \
    GRID="$grid" MAXSTEP="$MAXSTEP" DISCARD_STEPS="$DISCARD_STEPS" REPEATS="$REPEATS" \
    DELTAT="$DELTAT" NP="$np" TOPOLOGY="$topology" GPU_IDS="$gpu_ids" \
    FILTER_WORKSPACE="$FILTER_WORKSPACE" HALO_TRANSPORT="$halo_transport" \
    SYNC_MODE="$sync_mode" FEQCHKPT=9999 \
    "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh" || return $?
  verify_full_workspace "$RUN_ROOT/$label" || return $?
  found="$(find "$RUN_ROOT/$label" -type f -name '*.h5' -print)"
  if [[ -n "$found" ]]; then
    printf 'benchmark generated forbidden HDF5 output:\n%s\n' "$found" >&2
    return 1
  fi
}

for entry in "${main_matrix[@]}"; do
  read -r label np topology grid local_grid kind halo_transport sync_mode <<< "$entry"
  run_case "$label" run_timing "$label" "$np" "$topology" "$grid" \
    "$halo_transport" "$sync_mode"
done

"$PYTHON_EXE" "$BENCH_DIR/summarize_tgv_scaling_matrix.py" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT" \
  --output-tsv "$RUN_ROOT/preliminary_scaling_results.tsv" \
  --summary "$RUN_ROOT/preliminary_scaling_summary.md" \
  > "$RUN_ROOT/preliminary_scaling_summary.stdout" 2>&1
preliminary_status=$?

select_fastest_topology() {
  local np="$1"
  (( preliminary_status == 0 )) || return 1
  "$PYTHON_EXE" - "$RUN_ROOT/preliminary_scaling_results.tsv" "$np" <<'PY'
import csv
import sys

path, target_np = sys.argv[1], int(sys.argv[2])
with open(path, encoding="ascii", newline="") as handle:
    rows = [
        row for row in csv.DictReader(handle, delimiter="\t")
        if row["kind"] == "strong"
        and int(row["np"]) == target_np
        and row["halo_transport"] == "pinned-pipeline"
        and row["sync_mode"] == "explicit"
    ]
if not rows:
    raise SystemExit(1)
print(min(rows, key=lambda row: float(row["median_rk_seconds"]))["topology"])
PY
}

local_grid_for_topology() {
  local np="$1" topology="$2"
  awk -F '\t' -v np="$np" -v topology="$topology" \
    'NR > 1 && $2 == "strong" && $3 == np && $4 == topology {print $6; exit}' "$MANIFEST"
}

append_control_manifest() {
  printf '%s\tcontrol\t%s\t%s\t512,512,512\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" >> "$MANIFEST"
}

skip_control() {
  local label="$1" np="$2" halo_transport="$3" sync_mode="$4" now
  now="$(date -Iseconds)"
  append_control_manifest "$label" "$np" unavailable unavailable "$halo_transport" "$sync_mode"
  record_case "$label" SKIP 3 "$now" "$now" 'no completed primary topology'
}

run_controls_for_np() {
  local np="$1" topology local_grid label
  topology="$(select_fastest_topology "$np")" || topology=''
  if [[ -z "$topology" ]]; then
    skip_control "control_np${np}_pinned_overlap_explicit" "$np" pinned-overlap explicit
    skip_control "control_np${np}_pipeline_dependency" "$np" pinned-pipeline dependency
    return
  fi
  local_grid="$(local_grid_for_topology "$np" "$topology")"

  label="control_np${np}_pinned_overlap_explicit"
  append_control_manifest "$label" "$np" "$topology" "$local_grid" pinned-overlap explicit
  run_case "$label" run_timing "$label" "$np" "$topology" 512,512,512 \
    pinned-overlap explicit

  label="control_np${np}_pipeline_dependency"
  append_control_manifest "$label" "$np" "$topology" "$local_grid" pinned-pipeline dependency
  run_case "$label" run_timing "$label" "$np" "$topology" 512,512,512 \
    pinned-pipeline dependency
}

run_controls_for_np 2
run_controls_for_np 4

"$PYTHON_EXE" "$BENCH_DIR/summarize_tgv_scaling_matrix.py" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT" \
  --output-tsv "$RUN_ROOT/scaling_results.tsv" \
  --summary "$RUN_ROOT/scaling_summary.md" > "$RUN_ROOT/scaling_summary.stdout" 2>&1
summary_status=$?

pass_count="$(awk -F '\t' 'NR>1 && $2=="PASS" {count++} END {print count+0}' "$STATUS_FILE")"
fail_count="$(awk -F '\t' 'NR>1 && $2=="FAIL" {count++} END {print count+0}' "$STATUS_FILE")"
skip_count="$(awk -F '\t' 'NR>1 && $2=="SKIP" {count++} END {print count+0}' "$STATUS_FILE")"
unstable_count=0
very_unstable_count=0
if [[ -f "$RUN_ROOT/scaling_results.tsv" ]]; then
  unstable_count="$(awk -F '\t' 'NR>1 && $11+0 > 0.03 {count++} END {print count+0}' \
    "$RUN_ROOT/scaling_results.tsv")"
  very_unstable_count="$(awk -F '\t' 'NR>1 && $11+0 > 0.05 {count++} END {print count+0}' \
    "$RUN_ROOT/scaling_results.tsv")"
fi
{
  printf 'preliminary_summary_status=%s\n' "$preliminary_status"
  printf 'summary_status=%s\n' "$summary_status"
  printf 'expected_cases=%s\n' "$EXPECTED_CASES"
  printf 'passed_cases=%s\n' "$pass_count"
  printf 'failed_cases=%s\n' "$fail_count"
  printf 'skipped_cases=%s\n' "$skip_count"
  printf 'spread_over_3_percent=%s\n' "$unstable_count"
  printf 'spread_over_5_percent=%s\n' "$very_unstable_count"
  if [[ "$summary_status" -eq 0 && "$pass_count" -eq "$EXPECTED_CASES" && \
        "$unstable_count" -eq 0 ]]; then
    printf 'overall_status=PASS\n'
  else
    printf 'overall_status=PARTIAL\n'
  fi
} > "$RUN_ROOT/job_status.txt"

printf 'A800 TGV scaling matrix complete: pass=%s fail=%s skip=%s summary=%s root=%s\n' \
  "$pass_count" "$fail_count" "$skip_count" "$summary_status" "$RUN_ROOT"
exit 0
