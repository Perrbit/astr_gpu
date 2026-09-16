#!/bin/bash
#SBATCH --job-name=astr-a800-tgv-diag
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-tgv-diag-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-tgv-diag-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
EXPECTED_SOURCE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7
EXPECTED_GPU_EXE_SHA256=7ed0e326e907143d9cb5b7abb045ea8cea6d0256c0f63cdd02f91ceed0b06755
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_p4_c801eeb}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$WORK_ROOT/build_gpu_p4_c801eeb}"
BENCH_DIR="${BENCH_DIR:-$WORK_ROOT/a800_tgv_performance_diagnostics_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
NSYS_EXE="${NSYS_EXE:-/data/apps/nvhpc/24.11/Linux_x86_64/24.11/compilers/bin/nsys}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_tgv_performance_diagnostics_${SLURM_JOB_ID:-dryrun}}"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
MANIFEST="$RUN_ROOT/matrix_manifest.tsv"
MAXSTEP=21
DISCARD_STEPS=2
REPEATS=5
DELTAT=1.0e-4
FILTER_WORKSPACE=full
EXPECTED_PHASE_CASES=15

phase_matrix=(
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
  "control_np2_overlap 2 2,1,1 512,512,512 256,512,512 control pinned-overlap explicit"
  "control_np4_overlap 4 1,2,2 512,512,512 512,256,256 control pinned-overlap explicit"
)

nsys_matrix=(
  "nsys_np1_111 1 1,1,1 512,512,512 pinned explicit"
  "nsys_np2_211 2 2,1,1 512,512,512 pinned-pipeline explicit"
  "nsys_np4_122 4 1,2,2 512,512,512 pinned-pipeline explicit"
)

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
PREPARE_CASE="$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py"
BENCHMARK_DRIVER="$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
OVERLAP_SUMMARY="$SRC_DIR/tests/gpu_validation/summarize_mpi_kernel_overlap.py"
for required in "$GPU_EXE" "$GPU_BUILD_DIR/compile_commands.json" "$PYTHON_EXE" \
  "$HDF5_ROOT/bin/h5pfc" "$NSYS_EXE" "$PREPARE_CASE" "$BENCHMARK_DRIVER" \
  "$OVERLAP_SUMMARY" "$BENCH_DIR/summarize_tgv_scaling_matrix.py" \
  "$BENCH_DIR/summarize_tgv_phase_matrix.py"; do
  [[ -f "$required" ]] || { printf 'missing diagnostic dependency: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$GPU_EXE" && -x "$PYTHON_EXE" && -x "$NSYS_EXE" ]] || {
  printf 'required executable is not executable\n' >&2
  exit 2
}
[[ -d "$SRC_DIR/.git" ]] || { printf 'source checkout is absent\n' >&2; exit 2; }
[[ "$(git -C "$SRC_DIR" rev-parse HEAD)" == "$EXPECTED_SOURCE_COMMIT" ]] || {
  printf 'source commit does not match the frozen scaling run\n' >&2
  exit 2
}
[[ -z "$(git -C "$SRC_DIR" status --short)" ]] || {
  printf 'source checkout is not clean\n' >&2
  git -C "$SRC_DIR" status --short >&2
  exit 2
}
[[ "$(sha256sum "$GPU_EXE" | awk '{print $1}')" == "$EXPECTED_GPU_EXE_SHA256" ]] || {
  printf 'GPU executable hash does not match the frozen scaling run\n' >&2
  exit 2
}
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite run root: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/nsys"
export TMPDIR="$RUN_ROOT/tmp"

if ! ldd "$GPU_EXE" > "$RUN_ROOT/gpu_dependencies.txt" 2>&1 || \
   grep -q 'not found' "$RUN_ROOT/gpu_dependencies.txt"; then
  printf 'unresolved compute-node dynamic library dependency\n' >&2
  exit 2
fi
grep -Fq "$HDF5_ROOT/lib/libhdf5_fortran" "$RUN_ROOT/gpu_dependencies.txt" || {
  printf 'GPU executable does not use the approved HDF5 Fortran runtime\n' >&2
  exit 2
}
if grep -E -- '(--use_fast_math|-ffast-math|-Ofast|-Mfprelaxed|-Mfpapprox|-gpu=[^[:space:]]*fastmath|(^|[[:space:]])-fast([[:space:]]|$))' \
  "$GPU_BUILD_DIR/compile_commands.json" > "$RUN_ROOT/prohibited_math_flags.txt"; then
  printf 'prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi
for input_file in \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/input.tgv" \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/controller" \
  "$SRC_DIR/examples/Taylor_Green_Vortex/datin/parallel.info"; do
  [[ -f "$input_file" ]] || continue
  if LC_ALL=C grep -q $'\r' "$input_file"; then
    printf 'CRLF or bare carriage return found in %s\n' "$input_file" >&2
    exit 2
  fi
done

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs, got %s\n' "$visible" >&2; exit 2; }
gpu1="${gpu[0]}"
gpu2="${gpu[0]},${gpu[1]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

gpu_ids_for_np() {
  case "$1" in
    1) printf '%s\n' "$gpu1" ;;
    2) printf '%s\n' "$gpu2" ;;
    4) printf '%s\n' "$gpu4" ;;
    *) return 2 ;;
  esac
}

prepare_case() {
  local case_dir="$1" grid="$2" maxstep="$3"
  "$PYTHON_EXE" "$PREPARE_CASE" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid "$grid" --maxstep "$maxstep" --feqchkpt 9999 --feqlist 9999 \
    --deltat "$DELTAT" --lfilter t --diffterm t --scheme 643e --lreadgrid f || return $?
  rm -f "$case_dir/datin/grid.h5"
}

assert_no_field_hdf5() {
  local root="$1" found
  found="$(find "$root" -type f \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  [[ -z "$found" ]] || { printf 'forbidden field HDF5 output:\n%s\n' "$found" >&2; return 1; }
}

scan_solver_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log" || return 1
  ! grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"
}

run_compute_preflight() {
  local out="$RUN_ROOT/compute_preflight" case_dir="$RUN_ROOT/compute_preflight/case"
  local log="$RUN_ROOT/compute_preflight/run.log" status
  mkdir -p "$out"
  prepare_case "$case_dir" 32,32,32 0 || return $?
  (
    cd "$case_dir" || exit 2
    env OMPI_MCA_sharedfp=individual CUDA_VISIBLE_DEVICES="$gpu4" \
      ASTR_FORCE_MPI_TOPOLOGY=2,2,1 ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      ASTR_GPU_PHASE_TIMING=0 ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 \
      ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_HALO_TRANSPORT=pinned-pipeline \
      ASTR_GPU_FILTER_WORKSPACE=full mpirun -np 4 "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" && assert_no_field_hdf5 "$case_dir"
}

run_compute_preflight || {
  printf 'compute-node preflight failed; diagnostics will not run\n' >&2
  exit 2
}

{
  printf 'source_commit: %s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  printf 'gpu_executable_sha256: '; sha256sum "$GPU_EXE"
  printf 'date: %s\nhostname: %s\nslurm_job_id: %s\n' "$(date -Iseconds)" "$(hostname)" "$SLURM_JOB_ID"
  printf 'cuda_visible_devices: %s\nprecision: FP64\nfilter_workspace: full\n' "$visible"
  printf 'field_hdf5: disabled\nstatistics output: enabled\nphase synchronization: explicit\n'
  printf '\nloaded_modules:\n'; module -t list 2>&1
  printf '\nnvidia_smi:\n'; nvidia-smi
  printf '\nnvidia_topology:\n'; nvidia-smi topo -m
  printf '\nnsys:\n'; "$NSYS_EXE" --version
  printf '\nnvfortran:\n'; nvfortran --version
  printf '\nmpirun:\n'; mpirun --version
  printf '\ndriver_parameters:\n'; grep -E 'RmProfilingAdminOnly|RestrictProfiling' /proc/driver/nvidia/params 2>/dev/null || true
} > "$RUN_ROOT/environment.txt"

printf 'tool\tcase\tstatus\texit_code\tstarted\tfinished\tdetail\n' > "$STATUS_FILE"
printf 'label\tkind\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\tsync_mode\n' > "$MANIFEST"
for entry in "${phase_matrix[@]}"; do
  read -r label np topology grid local_grid kind halo sync <<< "$entry"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$kind" "$np" "$topology" "$grid" "$local_grid" "$halo" "$sync" >> "$MANIFEST"
done

record_case() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$STATUS_FILE"
}

run_case() {
  local tool="$1" label="$2"
  shift 2
  local started finished status
  started="$(date -Iseconds)"
  printf 'CASE_START %s %s %s\n' "$tool" "$label" "$started"
  "$@"
  status=$?
  finished="$(date -Iseconds)"
  if (( status == 0 )); then
    record_case "$tool" "$label" PASS 0 "$started" "$finished" "$RUN_ROOT/$tool/$label"
  else
    record_case "$tool" "$label" FAIL "$status" "$started" "$finished" "$RUN_ROOT/$tool/$label"
  fi
  printf 'CASE_END %s %s status=%s %s\n' "$tool" "$label" "$status" "$finished"
  return 0
}

nsys_report_has_cuda() {
  local report="$1" sqlite="${report%.nsys-rep}.sqlite"
  "$NSYS_EXE" stats --force-export=true --report cuda_gpu_kern_sum "$report" \
    > "${report%.nsys-rep}_cuda_gpu_kern_sum.txt" 2>&1 || return $?
  "$PYTHON_EXE" - "$sqlite" <<'PY'
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as connection:
    tables = {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        raise SystemExit(1)
    count = connection.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    raise SystemExit(0 if count > 0 else 1)
PY
}

run_nsys_profile() {
  local label="$1" np="$2" topology="$3" grid="$4" halo="$5" sync="$6"
  local out="$RUN_ROOT/nsys/$label" case_dir="$RUN_ROOT/nsys/$label/case"
  local log="$RUN_ROOT/nsys/$label/solver.log" gpu_ids report status
  local reports=()
  mkdir -p "$out/tmp"
  prepare_case "$case_dir" "$grid" 2 || return $?
  gpu_ids="$(gpu_ids_for_np "$np")" || return $?
  (
    cd "$case_dir" || exit 2
    env TMPDIR="$out/tmp" OMPI_MCA_sharedfp=individual CUDA_VISIBLE_DEVICES="$gpu_ids" \
      ASTR_FORCE_MPI_TOPOLOGY="$topology" ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 \
      ASTR_GPU_PHASE_TIMING=0 ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 \
      ASTR_GPU_SYNC_MODE="$sync" ASTR_GPU_HALO_TRANSPORT="$halo" \
      ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
      mpirun -np "$np" "$NSYS_EXE" profile --trace=cuda,nvtx,osrt,mpi \
        --mpi-impl=openmpi --sample=none --cpuctxsw=none --stats=false \
        --force-overwrite=true -o "$out/trace_rank_%q{OMPI_COMM_WORLD_RANK}" \
        "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$case_dir" || return 1
  mapfile -t reports < <(find "$out" -maxdepth 1 -type f -name 'trace_rank_*.nsys-rep' -print | sort)
  [[ "${#reports[@]}" -eq "$np" ]] || return 1
  for report in "${reports[@]}"; do
    nsys_report_has_cuda "$report" || return 4
    "$NSYS_EXE" stats --force-export=true --report cuda_api_sum "$report" \
      > "${report%.nsys-rep}_cuda_api_sum.txt" 2>&1 || return $?
    "$NSYS_EXE" stats --force-export=true --report mpi_event_sum "$report" \
      > "${report%.nsys-rep}_mpi_event_sum.txt" 2>&1 || return $?
    "$PYTHON_EXE" "$OVERLAP_SUMMARY" "${report%.nsys-rep}.sqlite" \
      > "${report%.nsys-rep}_mpi_kernel_overlap.json" || return $?
  done
}

nsys_preflight_start="$(date -Iseconds)"
run_nsys_profile nsys_preflight 1 1,1,1 32,32,32 pinned explicit
nsys_preflight_status=$?
nsys_preflight_finish="$(date -Iseconds)"
if (( nsys_preflight_status == 0 )); then
  record_case nsys nsys_preflight PASS 0 "$nsys_preflight_start" "$nsys_preflight_finish" "$RUN_ROOT/nsys/nsys_preflight"
  for entry in "${nsys_matrix[@]}"; do
    read -r label np topology grid halo sync <<< "$entry"
    run_case nsys "$label" run_nsys_profile "$label" "$np" "$topology" "$grid" "$halo" "$sync"
  done
else
  record_case nsys nsys_preflight BLOCKED "$nsys_preflight_status" \
    "$nsys_preflight_start" "$nsys_preflight_finish" 'CUDA events unavailable; internal timing continues'
  printf 'NSYS_BLOCKED status=%s; internal phase timing continues\n' "$nsys_preflight_status"
fi

verify_full_workspace() {
  local out="$1" count=0 log
  while IFS= read -r -d '' log; do
    count=$((count + 1))
    grep -aq 'GPU filter workspace storage: full' "$log" || return 1
  done < <(find "$out" -type f -name run.log -print0)
  [[ "$count" -eq 6 ]]
}

run_phase_timing() {
  local label="$1" np="$2" topology="$3" grid="$4" halo="$5" sync="$6"
  local gpu_ids found
  gpu_ids="$(gpu_ids_for_np "$np")" || return $?
  GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/phase/$label" LABEL="$label" \
    GRID="$grid" MAXSTEP="$MAXSTEP" DISCARD_STEPS="$DISCARD_STEPS" REPEATS="$REPEATS" \
    DELTAT="$DELTAT" NP="$np" TOPOLOGY="$topology" GPU_IDS="$gpu_ids" \
    FILTER_WORKSPACE="$FILTER_WORKSPACE" HALO_TRANSPORT="$halo" SYNC_MODE="$sync" \
    PHASE_TIMING=1 FEQCHKPT=9999 "$BENCHMARK_DRIVER" || return $?
  verify_full_workspace "$RUN_ROOT/phase/$label" || return 1
  [[ -s "$RUN_ROOT/phase/$label/${label}_phase_summary.tsv" ]] || return 1
  found="$(find "$RUN_ROOT/phase/$label" -type f -name '*.h5' -print)"
  [[ -z "$found" ]] || return 1
}

mkdir -p "$RUN_ROOT/phase"
for entry in "${phase_matrix[@]}"; do
  read -r label np topology grid local_grid kind halo sync <<< "$entry"
  run_case phase "$label" run_phase_timing "$label" "$np" "$topology" "$grid" "$halo" "$sync"
done

"$PYTHON_EXE" "$BENCH_DIR/summarize_tgv_scaling_matrix.py" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT/phase" \
  --output-tsv "$RUN_ROOT/instrumented_scaling_results.tsv" \
  --summary "$RUN_ROOT/instrumented_scaling_summary.md" \
  > "$RUN_ROOT/instrumented_scaling_summary.stdout" 2>&1
scaling_summary_status=$?
"$PYTHON_EXE" "$BENCH_DIR/summarize_tgv_phase_matrix.py" \
  --manifest "$MANIFEST" --run-root "$RUN_ROOT/phase" \
  --output-tsv "$RUN_ROOT/phase_results.tsv" --summary "$RUN_ROOT/phase_summary.md" \
  > "$RUN_ROOT/phase_summary.stdout" 2>&1
phase_summary_status=$?

phase_pass="$(awk -F '\t' '$1=="phase" && $3=="PASS" {n++} END {print n+0}' "$STATUS_FILE")"
phase_fail="$(awk -F '\t' '$1=="phase" && $3=="FAIL" {n++} END {print n+0}' "$STATUS_FILE")"
nsys_pass="$(awk -F '\t' '$1=="nsys" && $3=="PASS" {n++} END {print n+0}' "$STATUS_FILE")"
nsys_blocked="$(awk -F '\t' '$1=="nsys" && $3=="BLOCKED" {n++} END {print n+0}' "$STATUS_FILE")"
{
  printf 'expected_phase_cases=%s\nphase_passed=%s\nphase_failed=%s\n' \
    "$EXPECTED_PHASE_CASES" "$phase_pass" "$phase_fail"
  printf 'nsys_passed=%s\nnsys_blocked=%s\n' "$nsys_pass" "$nsys_blocked"
  printf 'scaling_summary_status=%s\nphase_summary_status=%s\n' \
    "$scaling_summary_status" "$phase_summary_status"
  if [[ "$phase_pass" -eq "$EXPECTED_PHASE_CASES" && "$phase_fail" -eq 0 && \
        "$scaling_summary_status" -eq 0 && "$phase_summary_status" -eq 0 ]]; then
    printf 'internal_timing_status=PASS\n'
  else
    printf 'internal_timing_status=PARTIAL\n'
  fi
  if [[ "$nsys_blocked" -gt 0 ]]; then
    printf 'nsys_status=BLOCKED_BY_PLATFORM\n'
  elif [[ "$nsys_pass" -eq 4 ]]; then
    printf 'nsys_status=PASS\n'
  else
    printf 'nsys_status=PARTIAL\n'
  fi
} > "$RUN_ROOT/job_status.txt"

printf 'A800 diagnostics complete: nsys_pass=%s nsys_blocked=%s phase_pass=%s phase_fail=%s root=%s\n' \
  "$nsys_pass" "$nsys_blocked" "$phase_pass" "$phase_fail" "$RUN_ROOT"
exit 0
