#!/bin/bash
#SBATCH --job-name=astr-a800-tgv-prof
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-tgv-prof-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-tgv-prof-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
EXPECTED_SOURCE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7
EXPECTED_GPU_EXE_SHA256=7ed0e326e907143d9cb5b7abb045ea8cea6d0256c0f63cdd02f91ceed0b06755
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_p4_c801eeb}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$WORK_ROOT/build_gpu_p4_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
NSYS_EXE="${NSYS_EXE:-/data/apps/nvhpc/24.11/Linux_x86_64/24.11/compilers/bin/nsys}"
NCU_EXE="${NCU_EXE:-/data/apps/nvhpc/25.5/Linux_x86_64/25.5/compilers/bin/ncu}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_tgv_gpu_diagnostics_${SLURM_JOB_ID:-dryrun}}"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
NSYS_MANIFEST="$RUN_ROOT/nsys_manifest.tsv"
NCU_MANIFEST="$RUN_ROOT/ncu_manifest.tsv"
NSYS_MAXSTEP=2
NCU_MAXSTEP=0
DELTAT=1.0e-4
FILTER_WORKSPACE=full

nsys_matrix=(
  "strong_np1_111 1 1,1,1 512,512,512 pinned explicit"
  "strong_np2_211 2 2,1,1 512,512,512 pinned-pipeline explicit"
  "strong_np2_121 2 1,2,1 512,512,512 pinned-pipeline explicit"
  "strong_np2_112 2 1,1,2 512,512,512 pinned-pipeline explicit"
  "strong_np4_411 4 4,1,1 512,512,512 pinned-pipeline explicit"
  "strong_np4_141 4 1,4,1 512,512,512 pinned-pipeline explicit"
  "strong_np4_114 4 1,1,4 512,512,512 pinned-pipeline explicit"
  "strong_np4_221 4 2,2,1 512,512,512 pinned-pipeline explicit"
  "strong_np4_212 4 2,1,2 512,512,512 pinned-pipeline explicit"
  "strong_np4_122 4 1,2,2 512,512,512 pinned-pipeline explicit"
  "weak_np1_111 1 1,1,1 256,256,256 pinned explicit"
  "weak_np2_211 2 2,1,1 512,256,256 pinned-pipeline explicit"
  "weak_np4_221 4 2,2,1 512,512,256 pinned-pipeline explicit"
  "control_np2_overlap 2 2,1,1 512,512,512 pinned-overlap explicit"
  "control_np2_dependency 2 2,1,1 512,512,512 pinned-pipeline dependency"
  "control_np4_overlap 4 1,2,2 512,512,512 pinned-overlap explicit"
  "control_np4_dependency 4 1,2,2 512,512,512 pinned-pipeline dependency"
)

ncu_matrix=(
  "primitive solver_gpu_q_to_primitive_kernel_"
  "filter_x solver_gpu_filter_x_periodic_interior_global_kernel_"
  "filter_y solver_gpu_filter_y_pong_to_q_halo_global_kernel_"
  "filter_z solver_gpu_filter_z_halo_global_kernel_"
  "gradcal gradcal_gpu_gradcal_dvel_kernel_"
  "conv_x solver_gpu_convective_rhs_x_global_kernel_"
  "conv_y solver_gpu_convective_rhs_y_global_kernel_"
  "conv_z solver_gpu_convective_rhs_z_global_kernel_"
  "diff_flux solver_gpu_diffusion_flux_global_kernel_"
  "diff_rhs_x solver_gpu_diffusion_rhs_x_stored_global_kernel_"
  "diff_rhs_y solver_gpu_diffusion_rhs_y_stored_global_kernel_"
  "diff_rhs_z solver_gpu_diffusion_rhs_z_stored_global_kernel_"
)

for path in "$SRC_DIR" "$GPU_BUILD_DIR" "$HDF5_ROOT" "$PYTHON_EXE" "$RUN_ROOT"; do
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
OVERLAP_SUMMARY="$SRC_DIR/tests/gpu_validation/summarize_mpi_kernel_overlap.py"

for required in "$GPU_EXE" "$GPU_BUILD_DIR/compile_commands.json" "$PYTHON_EXE" \
  "$HDF5_ROOT/bin/h5pfc" "$PREPARE_CASE" "$OVERLAP_SUMMARY" \
  "$NSYS_EXE" "$NCU_EXE"; do
  [[ -f "$required" ]] || { printf 'missing diagnostic dependency: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$GPU_EXE" && -x "$PYTHON_EXE" ]] || {
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
mkdir -p "$RUN_ROOT/tmp" "$RUN_ROOT/nsys" "$RUN_ROOT/ncu"
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

prepare_tgv_case() {
  local case_dir="$1" grid="$2" maxstep="$3"
  "$PYTHON_EXE" "$PREPARE_CASE" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid "$grid" --maxstep "$maxstep" \
    --feqchkpt 9999 --feqlist 9999 --deltat "$DELTAT" \
    --lfilter t --diffterm t --scheme 643e --lreadgrid f || return $?
  rm -f "$case_dir/datin/grid.h5"
}

assert_no_field_hdf5() {
  local root="$1" found
  found="$(find "$root" -type f \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  if [[ -n "$found" ]]; then
    printf 'forbidden field HDF5 output:\n%s\n' "$found" >&2
    return 1
  fi
}

scan_solver_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log" || return 1
  if grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    return 1
  fi
}

run_compute_preflight() {
  local out="$RUN_ROOT/compute_preflight" case_dir="$RUN_ROOT/compute_preflight/case"
  local log="$RUN_ROOT/compute_preflight/run.log" status
  mkdir -p "$out"
  prepare_tgv_case "$case_dir" 32,32,32 0 || return $?
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
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$case_dir"
}

if ! run_compute_preflight; then
  printf 'compute-node preflight failed; profiling matrix will not run\n' >&2
  exit 2
fi

run_profiler_preflight() {
  local out="$RUN_ROOT/profiler_preflight" case_dir="$RUN_ROOT/profiler_preflight/case"
  local log="$RUN_ROOT/profiler_preflight/solver.log" report sqlite status
  local reports=()
  mkdir -p "$out/tmp"
  prepare_tgv_case "$case_dir" 32,32,32 0 || return $?
  (
    cd "$case_dir" || exit 2
    env TMPDIR="$out/tmp" OMPI_MCA_sharedfp=individual CUDA_VISIBLE_DEVICES="$gpu2" \
      ASTR_FORCE_MPI_TOPOLOGY=2,1,1 ASTR_GPU_RK_TIMING=1 \
      ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=pinned-pipeline ASTR_GPU_FILTER_WORKSPACE=full \
      mpirun -np 2 "$NSYS_EXE" profile --trace=cuda,mpi,nvtx --mpi-impl=openmpi \
        --sample=none --cpuctxsw=none --stats=false --force-overwrite=true \
        -o "$out/trace_rank_%q{OMPI_COMM_WORLD_RANK}" \
        "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$case_dir" || return 1
  mapfile -t reports < <(find "$out" -maxdepth 1 -type f -name 'trace_rank_*.nsys-rep' -print | sort)
  [[ "${#reports[@]}" -eq 2 ]] || {
    printf 'profiler preflight expected two rank reports, found %s\n' "${#reports[@]}" >&2
    return 1
  }
  for report in "${reports[@]}"; do
    sqlite="${report%.nsys-rep}.sqlite"
    "$NSYS_EXE" stats --force-export=true --report cuda_gpu_kern_sum "$report" \
      > "${report%.nsys-rep}_cuda_gpu_kern_sum.txt" 2>&1 || return $?
    "$NSYS_EXE" stats --force-export=true --report mpi_event_sum "$report" \
      > "${report%.nsys-rep}_mpi_event_sum.txt" 2>&1 || return $?
    "$PYTHON_EXE" - "$sqlite" <<'PY' || return $?
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    has_cuda = "CUPTI_ACTIVITY_KIND_KERNEL" in tables
    has_mpi = bool(
        {"MPI_START_WAIT_EVENTS", "MPI_OTHER_EVENTS", "MPI_P2P_EVENTS"} & tables
    )
    if not (has_cuda and has_mpi):
        raise SystemExit("profiler preflight report lacks CUDA or MPI trace data")
PY
  done
}

if ! run_profiler_preflight; then
  printf 'profiler preflight failed; profiling matrix will not run\n' >&2
  exit 2
fi

{
  printf 'source_commit: %s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  printf 'gpu_executable_sha256: '; sha256sum "$GPU_EXE"
  printf 'date: %s\n' "$(date -Iseconds)"
  printf 'hostname: %s\n' "$(hostname)"
  printf 'slurm_job_id: %s\n' "$SLURM_JOB_ID"
  printf 'cuda_visible_devices: %s\n' "$visible"
  printf 'precision: FP64\nfilter_workspace: %s\n' "$FILTER_WORKSPACE"
  printf 'field_hdf5: disabled\nstatistics output: enabled\n'
  printf '\nloaded_modules:\n'; module -t list 2>&1
  printf '\nnvidia_smi:\n'; nvidia-smi
  printf '\nnvidia_topology:\n'; nvidia-smi topo -m
  printf '\nnsys:\n'; "$NSYS_EXE" --version
  printf '\nncu:\n'; "$NCU_EXE" --version
  printf '\nnvfortran:\n'; nvfortran --version
  printf '\nmpirun:\n'; mpirun --version
} > "$RUN_ROOT/environment.txt"

printf 'tool\tcase\tstatus\texit_code\tstarted\tfinished\tdetail\n' > "$STATUS_FILE"
printf 'label\tnp\ttopology\tgrid\thalo_transport\tsync_mode\n' > "$NSYS_MANIFEST"
printf 'label\tkernel_regex\tgrid\tnp\n' > "$NCU_MANIFEST"

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

run_nsys_profile() {
  local label="$1" np="$2" topology="$3" grid="$4" halo="$5" sync="$6"
  local out="$RUN_ROOT/nsys/$label" case_dir="$RUN_ROOT/nsys/$label/case"
  local log="$RUN_ROOT/nsys/$label/solver.log" report sqlite gpu_ids status report_name stem
  local reports=()
  mkdir -p "$out/tmp"
  prepare_tgv_case "$case_dir" "$grid" "$NSYS_MAXSTEP" || return $?
  gpu_ids="$(gpu_ids_for_np "$np")" || return $?
  (
    cd "$case_dir" || exit 2
    env TMPDIR="$out/tmp" OMPI_MCA_sharedfp=individual CUDA_VISIBLE_DEVICES="$gpu_ids" \
      ASTR_FORCE_MPI_TOPOLOGY="$topology" ASTR_GPU_RK_TIMING=1 \
      ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE="$sync" \
      ASTR_GPU_HALO_TRANSPORT="$halo" ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
      mpirun -np "$np" "$NSYS_EXE" profile --trace=cuda,mpi,nvtx --mpi-impl=openmpi \
        --sample=none --cpuctxsw=none --stats=false --force-overwrite=true \
        -o "$out/trace_rank_%q{OMPI_COMM_WORLD_RANK}" \
        "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$case_dir" || return 1
  mapfile -t reports < <(find "$out" -maxdepth 1 -type f -name 'trace_rank_*.nsys-rep' -print | sort)
  [[ "${#reports[@]}" -eq "$np" ]] || {
    printf 'expected %s Nsight Systems rank reports for %s, found %s\n' \
      "$np" "$label" "${#reports[@]}" >&2
    return 1
  }
  for report in "${reports[@]}"; do
    sqlite="${report%.nsys-rep}.sqlite"
    stem="$(basename "${report%.nsys-rep}")"
    for report_name in cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum mpi_event_sum mpi_msg_size_sum; do
      if ! "$NSYS_EXE" stats --force-export=true --report "$report_name" "$report" \
        > "$out/${stem}_${report_name}.txt" 2> "$out/${stem}_${report_name}.stderr"; then
        printf 'nsys stats failed for %s %s %s\n' "$label" "$stem" "$report_name" >&2
        return 1
      fi
    done
    [[ -s "$sqlite" ]] || { printf 'missing Nsight SQLite export: %s\n' "$sqlite" >&2; return 1; }
    "$PYTHON_EXE" "$OVERLAP_SUMMARY" "$sqlite" \
      > "$out/${stem}_mpi_kernel_overlap.json" || return $?
  done
}

for entry in "${nsys_matrix[@]}"; do
  read -r label np topology grid halo sync <<< "$entry"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$np" "$topology" "$grid" "$halo" "$sync" >> "$NSYS_MANIFEST"
  run_case nsys "$label" run_nsys_profile "$label" "$np" "$topology" "$grid" "$halo" "$sync"
done

NCU_CASE="$RUN_ROOT/ncu/case"
ncu_case_status=0
prepare_tgv_case "$NCU_CASE" 256,256,256 "$NCU_MAXSTEP" || ncu_case_status=$?

run_ncu_profile() {
  local label="$1" kernel_regex="$2"
  local out="$RUN_ROOT/ncu/$label" log="$RUN_ROOT/ncu/$label/solver.log"
  local report status
  (( ncu_case_status == 0 )) || return "$ncu_case_status"
  mkdir -p "$out/tmp"
  (
    cd "$NCU_CASE" || exit 2
    env TMPDIR="$out/tmp" OMPI_MCA_sharedfp=individual CUDA_VISIBLE_DEVICES="$gpu1" \
      ASTR_FORCE_MPI_TOPOLOGY=1,1,1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=pinned ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
      mpirun -np 1 "$NCU_EXE" --target-processes all --target-processes-filter 'regex:^astr$' \
        --set full --kernel-name "regex:${kernel_regex}" \
        --launch-skip 0 --launch-count 1 --clock-control none \
        --force-overwrite --export "$out/kernel" \
        "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  report="$out/kernel.ncu-rep"
  [[ -s "$report" ]] || { printf 'missing Nsight Compute report: %s\n' "$report" >&2; return 1; }
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$NCU_CASE" || return 1
  "$NCU_EXE" --import "$report" --page details --print-details all > "$out/details.txt" 2> "$out/details.stderr" || return $?
  "$NCU_EXE" --import "$report" --page raw --csv --print-units base > "$out/raw.csv" 2> "$out/raw.stderr" || return $?
}

for entry in "${ncu_matrix[@]}"; do
  read -r label kernel_regex <<< "$entry"
  printf '%s\t%s\t256,256,256\t1\n' "$label" "$kernel_regex" >> "$NCU_MANIFEST"
  run_case ncu "$label" run_ncu_profile "$label" "$kernel_regex"
done

assert_no_field_hdf5 "$RUN_ROOT" || true
pass_count="$(awk -F'\t' 'NR>1 && $3=="PASS" {count++} END {print count+0}' "$STATUS_FILE")"
fail_count="$(awk -F'\t' 'NR>1 && $3=="FAIL" {count++} END {print count+0}' "$STATUS_FILE")"
{
  printf 'job_id=%s\n' "$SLURM_JOB_ID"
  printf 'source_commit=%s\n' "$EXPECTED_SOURCE_COMMIT"
  printf 'pass=%s\nfail=%s\n' "$pass_count" "$fail_count"
  printf 'nsys_expected=%s\nncu_expected=%s\n' "${#nsys_matrix[@]}" "${#ncu_matrix[@]}"
  printf 'result_root=%s\n' "$RUN_ROOT"
  printf 'finished=%s\n' "$(date -Iseconds)"
} > "$RUN_ROOT/job_status.txt"

printf 'A800 TGV diagnostics complete: pass=%s fail=%s root=%s\n' \
  "$pass_count" "$fail_count" "$RUN_ROOT"
(( fail_count == 0 )) || exit 1
