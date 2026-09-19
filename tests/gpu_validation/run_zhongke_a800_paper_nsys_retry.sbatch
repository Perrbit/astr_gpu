#!/bin/bash
#SBATCH --job-name=astr-a800-nsys-retry
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-nsys-retry-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-nsys-retry-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SOURCE_PROVENANCE_COMMIT=8aefd6c5e50e6e804e4fe47b5ffa7bdd78a386e4
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cuda_aware_qualification_c801eeb}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_cuda_aware_qualification_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
NSYS_EXE="${NSYS_EXE:-/data/apps/nvhpc/25.5/Linux_x86_64/25.5/compilers/bin/nsys}"
QUALIFICATION_JSON="$WORK_ROOT/results/a800_cuda_aware_qualification_460370/qualification.json"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_paper_nsys_retry_${SLURM_JOB_ID:-dryrun}}"
GPU_EXE="$BUILD_DIR/bin/astr"
PREPARE_CASE="$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py"
OVERLAP_SUMMARY="$SRC_DIR/tests/gpu_validation/summarize_mpi_kernel_overlap.py"

EXPECTED_GPU_EXE_SHA256=c19d05a2b496161f267c19d493c8303cef8f8b0cc838231f61bdce120d7f1fbc
EXPECTED_PREPARE_SHA256=16dd45b851e86d9f580f41f4701f7972f1926091764fff8c2b0d4b57bd5d099a
EXPECTED_OVERLAP_SHA256=f4ce5a95132aabe13d7153d3d2a3f3987766592064cd0cedd9e6d4f37ffa7098
DELTAT=1.0e-4
FILTER_WORKSPACE=full
TOPOLOGY=1,2,2
NP=4

for path in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$PYTHON_EXE" "$NSYS_EXE" \
  "$QUALIFICATION_JSON" "$RUN_ROOT"; do
  case "$path/" in
    "$WORK_ROOT"/*|/data/apps/*) ;;
    *) printf 'path escapes approved roots: %s\n' "$path" >&2; exit 2 ;;
  esac
done

module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$(dirname "$PYTHON_EXE"):$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll='^hcoll,ucc,cuda'
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_sharedfp=individual
export UCX_MEMTYPE_CACHE=n
export UCX_CUDA_COPY_ENABLE_FABRIC=no
export UCX_CUDA_COPY_DMABUF=no
export UCX_CUDA_IPC_ENABLE_MNNVL=no
export UCX_TLS=self,sm,cuda_copy,cuda_ipc
export UCX_PROTO_INFO=n

for required in "$GPU_EXE" "$PYTHON_EXE" "$NSYS_EXE" "$QUALIFICATION_JSON" \
  "$PREPARE_CASE" "$OVERLAP_SUMMARY"; do
  [[ -f "$required" ]] || { printf 'missing profiler dependency: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$GPU_EXE" && -x "$PYTHON_EXE" && -x "$NSYS_EXE" ]] || {
  printf 'required executable is not executable\n' >&2
  exit 2
}

check_sha256() {
  local expected="$1" file="$2" actual
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    printf 'hash mismatch for %s: expected=%s actual=%s\n' "$file" "$expected" "$actual" >&2
    exit 2
  }
}

check_sha256 "$EXPECTED_GPU_EXE_SHA256" "$GPU_EXE"
check_sha256 "$EXPECTED_PREPARE_SHA256" "$PREPARE_CASE"
check_sha256 "$EXPECTED_OVERLAP_SHA256" "$OVERLAP_SUMMARY"

"$PYTHON_EXE" - "$QUALIFICATION_JSON" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
if data.get("production_qualified") is not True:
    raise SystemExit("A800 CUDA-aware MPI probe is not production_qualified")
PY

[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/preflight/per_rank" "$RUN_ROOT/preflight/outer" "$RUN_ROOT/main"
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$TMPDIR"

if ! ldd "$GPU_EXE" > "$RUN_ROOT/gpu_dependencies.txt" 2>&1 || \
   grep -q 'not found' "$RUN_ROOT/gpu_dependencies.txt"; then
  printf 'unresolved compute-node ASTR dependency\n' >&2
  exit 2
fi
grep -Fq "$HDF5_ROOT/lib/libhdf5_fortran" "$RUN_ROOT/gpu_dependencies.txt" || {
  printf 'ASTR does not use the approved HDF5 Fortran runtime\n' >&2
  exit 2
}

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || { printf 'expected four visible GPUs: %s\n' "$visible" >&2; exit 2; }

assert_no_field_hdf5() {
  local root="$1" found
  found="$(find "$root" -type f \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  [[ -z "$found" ]] || {
    printf 'forbidden field HDF5 output:\n%s\n' "$found" >&2
    return 1
  }
}

scan_solver_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log" || return 1
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log" || return 1
  grep -aq 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$log" || return 1
  ! grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"
}

prepare_case() {
  local case_dir="$1" grid="$2" maxstep="$3"
  "$PYTHON_EXE" "$PREPARE_CASE" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" --dst-case "$case_dir" \
    --use-gpu t --grid "$grid" --maxstep "$maxstep" --feqchkpt 9999 --feqlist 9999 \
    --deltat "$DELTAT" --lfilter t --diffterm t --scheme 643e --lreadgrid f || return $?
  rm -f "$case_dir/datin/grid.h5"
}

report_has_cuda_and_mpi() {
  local report="$1" stem="${report%.nsys-rep}"
  "$NSYS_EXE" stats --force-export=true --report cuda_gpu_kern_sum "$report" \
    > "${stem}_cuda_gpu_kern_sum.txt" 2>&1 || return $?
  "$PYTHON_EXE" - "${stem}.sqlite" <<'PY'
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    if not required.issubset(tables):
        raise SystemExit("Nsight export lacks CUDA kernel tables")
    kernels = connection.execute(
        "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()[0]
    mpi_tables = {
        "MPI_START_WAIT_EVENTS", "MPI_OTHER_EVENTS", "MPI_P2P_EVENTS"
    } & tables
    mpi_events = sum(
        connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        for table in mpi_tables
    )
    if kernels <= 0 or mpi_events <= 0:
        raise SystemExit(
            f"Nsight export lacks events: kernels={kernels} mpi_events={mpi_events}"
        )
    print(f"kernels={kernels} mpi_events={mpi_events}")
PY
}

summarize_report() {
  local report="$1" stem="${report%.nsys-rep}"
  "$NSYS_EXE" stats --force-export=true --report cuda_api_sum "$report" \
    > "${stem}_cuda_api_sum.txt" 2>&1 || return $?
  "$NSYS_EXE" stats --force-export=true --report mpi_event_sum "$report" \
    > "${stem}_mpi_event_sum.txt" 2>&1 || return $?
  "$NSYS_EXE" stats --force-export=true --report mpi_msg_size_sum "$report" \
    > "${stem}_mpi_msg_size_sum.txt" 2>&1 || return $?
  "$PYTHON_EXE" "$OVERLAP_SUMMARY" "${stem}.sqlite" \
    > "${stem}_mpi_kernel_overlap.json" || return $?
}

profile_per_rank() {
  local out="$1" case_dir="$2" log="$3" grid="$4" maxstep="$5"
  local report status
  local reports=()
  mkdir -p "$out"
  prepare_case "$case_dir" "$grid" "$maxstep" || return $?
  (
    cd "$case_dir" || exit 2
    export NSYS_SYSTEM_ID=ASTR_A800_SINGLE_NODE
    env CUDA_VISIBLE_DEVICES="$visible" ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=device-aware ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
      mpirun -np "$NP" "$NSYS_EXE" profile --trace=cuda,nvtx,osrt,mpi \
        --mpi-impl=openmpi --sample=none --cpuctxsw=none --stats=false \
        --force-overwrite=true -o "$out/trace_rank_%q{OMPI_COMM_WORLD_RANK}" \
        "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$case_dir" || return 1
  mapfile -t reports < <(find "$out" -maxdepth 1 -type f -name 'trace_rank_*.nsys-rep' -print | sort)
  [[ "${#reports[@]}" -eq 4 ]] || return 1
  for report in "${reports[@]}"; do
    report_has_cuda_and_mpi "$report" || return 1
  done
}

profile_outer() {
  local out="$1" case_dir="$2" log="$3" grid="$4" maxstep="$5"
  local report="$out/trace_all_ranks.nsys-rep" status
  mkdir -p "$out"
  prepare_case "$case_dir" "$grid" "$maxstep" || return $?
  (
    cd "$case_dir" || exit 2
    env CUDA_VISIBLE_DEVICES="$visible" ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=device-aware ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
      "$NSYS_EXE" profile --trace=cuda,nvtx,osrt,mpi --mpi-impl=openmpi \
        --sample=none --cpuctxsw=none --stats=false --wait=all \
        --force-overwrite=true -o "$out/trace_all_ranks" \
        mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  assert_no_field_hdf5 "$case_dir" || return 1
  [[ -s "$report" ]] || return 1
  report_has_cuda_and_mpi "$report"
}

{
  printf 'date=%s\nhostname=%s\nslurm_job_id=%s\n' \
    "$(date -Iseconds)" "$(hostname)" "$SLURM_JOB_ID"
  printf 'source_provenance_commit=%s\n' "$SOURCE_PROVENANCE_COMMIT"
  printf 'cuda_visible_devices=%s\nucx_tls=%s\n' "$visible" "$UCX_TLS"
  printf 'precision=FP64\nfilter_workspace=full\nsync_mode=explicit\n'
  printf 'field_hdf5=disabled\nstatistics_output=enabled\n'
  sha256sum "$GPU_EXE" "$PREPARE_CASE" "$OVERLAP_SUMMARY"
  module -t list 2>&1
  mpirun --version
  "$NSYS_EXE" --version
  nvidia-smi
  nvidia-smi topo -m
  ldd "$GPU_EXE"
} > "$RUN_ROOT/environment.txt" 2>&1

mode=''
if profile_per_rank "$RUN_ROOT/preflight/per_rank" \
  "$RUN_ROOT/preflight/per_rank_case" "$RUN_ROOT/preflight/per_rank.log" 64,64,64 1; then
  mode=per_rank
  printf 'per_rank preflight passed\n'
else
  printf 'per_rank preflight failed; trying outer profiler mode\n' >&2
  if profile_outer "$RUN_ROOT/preflight/outer" \
    "$RUN_ROOT/preflight/outer_case" "$RUN_ROOT/preflight/outer.log" 64,64,64 1; then
    mode=outer
    printf 'outer preflight passed\n'
  else
    printf 'both Nsight Systems preflight modes failed\n' >&2
    printf 'preflight_status=FAIL\n' > "$RUN_ROOT/job_status.txt"
    exit 2
  fi
fi

printf '%s\n' "$mode" > "$RUN_ROOT/profile_mode.txt"

main_status=0
if [[ "$mode" == per_rank ]]; then
  profile_per_rank "$RUN_ROOT/main/per_rank" "$RUN_ROOT/main/case" \
    "$RUN_ROOT/main/solver.log" 512,512,512 2 || main_status=$?
  if (( main_status == 0 )); then
    for report in "$RUN_ROOT"/main/per_rank/trace_rank_*.nsys-rep; do
      summarize_report "$report" || main_status=$?
    done
  fi
else
  profile_outer "$RUN_ROOT/main/outer" "$RUN_ROOT/main/case" \
    "$RUN_ROOT/main/solver.log" 512,512,512 2 || main_status=$?
  if (( main_status == 0 )); then
    summarize_report "$RUN_ROOT/main/outer/trace_all_ranks.nsys-rep" || main_status=$?
  fi
fi

if (( main_status == 0 )); then
  printf 'preflight_status=PASS\nprofile_mode=%s\nmain_status=PASS\n' "$mode" \
    > "$RUN_ROOT/job_status.txt"
  printf 'A800_NSYS_RETRY=PASS mode=%s root=%s\n' "$mode" "$RUN_ROOT"
  exit 0
fi

printf 'preflight_status=PASS\nprofile_mode=%s\nmain_status=FAIL\nexit_code=%s\n' \
  "$mode" "$main_status" > "$RUN_ROOT/job_status.txt"
printf 'A800_NSYS_RETRY=FAIL mode=%s status=%s root=%s\n' \
  "$mode" "$main_status" "$RUN_ROOT" >&2
exit "$main_status"
