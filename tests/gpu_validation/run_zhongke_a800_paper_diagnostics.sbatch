#!/bin/bash
#SBATCH --job-name=astr-a800-paperdiag
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-paperdiag-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-paperdiag-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SOURCE_PROVENANCE_COMMIT=8aefd6c5e50e6e804e4fe47b5ffa7bdd78a386e4
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cuda_aware_qualification_c801eeb}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_cuda_aware_qualification_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
NSYS_EXE="${NSYS_EXE:-/data/apps/nvhpc/24.11/Linux_x86_64/24.11/compilers/bin/nsys}"
QUALIFICATION_JSON="$WORK_ROOT/results/a800_cuda_aware_qualification_460370/qualification.json"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_paper_diagnostics_${SLURM_JOB_ID:-dryrun}}"
GPU_EXE="$BUILD_DIR/bin/astr"
BENCHMARK_DRIVER="$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
PREPARE_CASE="$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py"
OVERLAP_SUMMARY="$SRC_DIR/tests/gpu_validation/summarize_mpi_kernel_overlap.py"
PHASE_SUMMARY_DRIVER="$SRC_DIR/tests/gpu_validation/summarize_gpu_phase_timing.py"
STATUS_FILE="$RUN_ROOT/case_status.tsv"
MANIFEST="$RUN_ROOT/case_manifest.tsv"

EXPECTED_GPU_EXE_SHA256=c19d05a2b496161f267c19d493c8303cef8f8b0cc838231f61bdce120d7f1fbc
EXPECTED_BENCHMARK_SHA256=930e30d309adeb311be0e4a2adab4128300875aa38d0bc8049925a9c362f62b6
EXPECTED_PREPARE_SHA256=16dd45b851e86d9f580f41f4701f7972f1926091764fff8c2b0d4b57bd5d099a
EXPECTED_OVERLAP_SHA256=f4ce5a95132aabe13d7153d3d2a3f3987766592064cd0cedd9e6d4f37ffa7098
EXPECTED_PHASE_SUMMARY_SHA256=23ea5f3b9092af8ce00d072770e6c425c2d0a87ac2f094ee61490dd72385a730
EXPECTED_ASTR_MAIN_SHA256=5a084c7ff04609fb892eefcb0e974e8b78d37bf4104e0e6b836c80155fa4f483
EXPECTED_SRC_CMAKE_SHA256=5a4dab8f2e37243ca33cbe2d3d6a3d5053d95f8a9c67aa985e7e00ba9f9fee59
EXPECTED_DEVICE_MPI_SHA256=5aec9b95071bdaf6556ddc525e2a90484775c5aa618f3ee75472b28dded1640c
EXPECTED_DEVICE_RUNTIME_SHA256=3a69d8f81ea0c4e296ba39ed16ef9bdc44a291d352c9f042eb5958669a00239c
EXPECTED_GPU_RUNTIME_SHA256=69287b8c714407050dcf8637dab9b75bd876a2484d492507f300dc5a92f8e8ec
EXPECTED_HALO_EXCHANGE_SHA256=1472e8c80fb4f5cee5a5f4b37b2f860f621c1528ed590b4a2976e1b5b9f33b9e
EXPECTED_HALO_TRANSPORT_SHA256=de00edff372a9dc634fdd03bf0345d866cff672a36a0798ec06e644da4567e02

MAXSTEP=21
DISCARD_STEPS=2
REPEATS=5
DELTAT=1.0e-4
FILTER_WORKSPACE=full
NSYS_MAXSTEP=2

weak_matrix=(
  "weak_np1_111 1 1,1,1 256,256,256 256,256,256 pinned"
  "weak_np2_112_device_aware 2 1,1,2 256,256,512 256,256,256 device-aware"
  "weak_np4_122_device_aware 4 1,2,2 256,512,512 256,256,256 device-aware"
)

phase_matrix=(
  "phase_np1_111 1 1,1,1 512,512,512 512,512,512 pinned"
  "phase_np4_122_device_aware 4 1,2,2 512,512,512 512,256,256 device-aware"
)

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
  "$BENCHMARK_DRIVER" "$PREPARE_CASE" "$OVERLAP_SUMMARY" "$PHASE_SUMMARY_DRIVER"; do
  [[ -f "$required" ]] || { printf 'missing diagnostic dependency: %s\n' "$required" >&2; exit 2; }
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
check_sha256 "$EXPECTED_BENCHMARK_SHA256" "$BENCHMARK_DRIVER"
check_sha256 "$EXPECTED_PREPARE_SHA256" "$PREPARE_CASE"
check_sha256 "$EXPECTED_OVERLAP_SHA256" "$OVERLAP_SUMMARY"
check_sha256 "$EXPECTED_PHASE_SUMMARY_SHA256" "$PHASE_SUMMARY_DRIVER"
check_sha256 "$EXPECTED_ASTR_MAIN_SHA256" "$SRC_DIR/src/astr.F90"
check_sha256 "$EXPECTED_SRC_CMAKE_SHA256" "$SRC_DIR/src/CMakeLists.txt"
check_sha256 "$EXPECTED_DEVICE_MPI_SHA256" "$SRC_DIR/src_gpu/device_mpi_transport_gpu.cuf"
check_sha256 "$EXPECTED_DEVICE_RUNTIME_SHA256" "$SRC_DIR/src_gpu/device_runtime_gpu.cuf"
check_sha256 "$EXPECTED_GPU_RUNTIME_SHA256" "$SRC_DIR/src_gpu/gpu_runtime.cuf"
check_sha256 "$EXPECTED_HALO_EXCHANGE_SHA256" "$SRC_DIR/src_gpu/halo_exchange_gpu.cuf"
check_sha256 "$EXPECTED_HALO_TRANSPORT_SHA256" "$SRC_DIR/src_gpu/halo_transport_gpu.cuf"

"$PYTHON_EXE" - "$QUALIFICATION_JSON" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
if data.get("production_qualified") is not True:
    raise SystemExit("A800 CUDA-aware MPI probe is not production_qualified")
PY

[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT/preflight" "$RUN_ROOT/weak" "$RUN_ROOT/phase" "$RUN_ROOT/nsys"

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

run_compute_preflight() {
  local case_dir="$RUN_ROOT/preflight/case" log="$RUN_ROOT/preflight/run.log" status
  prepare_case "$case_dir" 64,64,64 1 || return $?
  (
    cd "$case_dir" || exit 2
    env CUDA_VISIBLE_DEVICES="$gpu4" UCX_PROTO_INFO=y \
      ASTR_FORCE_MPI_TOPOLOGY=2,2,1 ASTR_GPU_RK_TIMING=1 \
      ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=device-aware ASTR_GPU_FILTER_WORKSPACE=full \
      mpirun -np 4 "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  grep -aq 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$log" || return 1
  grep -aq 'cuda_ipc/cuda' "$log" || return 1
  [[ "$(grep -ac 'ASTR_GPU_PRE_MPI_DEVICE=' "$log")" -eq 4 ]] || return 1
  assert_no_field_hdf5 "$case_dir"
}

run_compute_preflight || {
  printf 'device-aware compute-node preflight failed; diagnostics will not run\n' >&2
  exit 2
}

{
  printf 'date=%s\nhostname=%s\nslurm_job_id=%s\n' \
    "$(date -Iseconds)" "$(hostname)" "$SLURM_JOB_ID"
  printf 'source_provenance_commit=%s\n' "$SOURCE_PROVENANCE_COMMIT"
  printf 'cuda_visible_devices=%s\nucx_tls=%s\n' "$visible" "$UCX_TLS"
  printf 'precision=FP64\nfilter_workspace=full\nsync_mode=explicit\n'
  printf 'field_hdf5=disabled\nstatistics_output=enabled\n'
  sha256sum "$GPU_EXE" "$BENCHMARK_DRIVER" "$PREPARE_CASE" \
    "$OVERLAP_SUMMARY" "$PHASE_SUMMARY_DRIVER" \
    "$SRC_DIR/src/astr.F90" "$SRC_DIR/src/CMakeLists.txt" \
    "$SRC_DIR/src_gpu/device_mpi_transport_gpu.cuf" \
    "$SRC_DIR/src_gpu/device_runtime_gpu.cuf" "$SRC_DIR/src_gpu/gpu_runtime.cuf" \
    "$SRC_DIR/src_gpu/halo_exchange_gpu.cuf" "$SRC_DIR/src_gpu/halo_transport_gpu.cuf"
  module -t list 2>&1
  mpirun --version
  "$NSYS_EXE" --version
  nvidia-smi
  nvidia-smi topo -m
  ldd "$GPU_EXE"
} > "$RUN_ROOT/environment.txt" 2>&1

printf 'tool\tcase\tstatus\texit_code\tstarted\tfinished\tdetail\n' > "$STATUS_FILE"
printf 'tool\tlabel\tnp\ttopology\tgrid\tlocal_grid\thalo_transport\n' > "$MANIFEST"
for entry in "${weak_matrix[@]}"; do
  read -r label np topology grid local_grid backend <<< "$entry"
  printf 'weak\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$np" "$topology" "$grid" "$local_grid" "$backend" >> "$MANIFEST"
done
for entry in "${phase_matrix[@]}"; do
  read -r label np topology grid local_grid backend <<< "$entry"
  printf 'phase\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$np" "$topology" "$grid" "$local_grid" "$backend" >> "$MANIFEST"
done
printf 'nsys\tnsys_np4_122_device_aware\t4\t1,2,2\t512,512,512\t512,256,256\tdevice-aware\n' >> "$MANIFEST"

record_case() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$STATUS_FILE"
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

run_benchmark() {
  local tool="$1" label="$2" np="$3" topology="$4" grid="$5" backend="$6" phase_timing="$7"
  local gpu_ids out count=0 log
  gpu_ids="$(gpu_ids_for_np "$np")" || return $?
  out="$RUN_ROOT/$tool/$label"
  GPU_EXE="$GPU_EXE" OUT_DIR="$out" LABEL="$label" \
    GRID="$grid" MAXSTEP="$MAXSTEP" DISCARD_STEPS="$DISCARD_STEPS" \
    REPEATS="$REPEATS" DELTAT="$DELTAT" NP="$np" TOPOLOGY="$topology" \
    GPU_IDS="$gpu_ids" FILTER_WORKSPACE="$FILTER_WORKSPACE" \
    HALO_TRANSPORT="$backend" SYNC_MODE=explicit PHASE_TIMING="$phase_timing" FEQCHKPT=9999 \
    "$BENCHMARK_DRIVER" || return $?
  [[ -s "$out/${label}_timings.tsv" ]] || return 1
  while IFS= read -r -d '' log; do
    count=$((count + 1))
    grep -aq 'GPU filter workspace storage: full' "$log" || return 1
    grep -aq "ASTR_GPU_HALO_TRANSPORT_SELECTED=$backend" "$log" || return 1
  done < <(find "$out" -type f -name run.log -print0)
  [[ "$count" -eq $((REPEATS + 1)) ]] || return 1
  if [[ "$phase_timing" == 1 ]]; then
    [[ -s "$out/${label}_phase_summary.tsv" ]] || return 1
  fi
  assert_no_field_hdf5 "$out"
}

run_nsys() {
  local label=nsys_np4_122_device_aware
  local out="$RUN_ROOT/nsys/$label" case_dir="$RUN_ROOT/nsys/$label/case"
  local log="$RUN_ROOT/nsys/$label/solver.log" report stem status
  local reports=()
  mkdir -p "$out/tmp"
  prepare_case "$case_dir" 512,512,512 "$NSYS_MAXSTEP" || return $?
  (
    cd "$case_dir" || exit 2
    env TMPDIR="$out/tmp" CUDA_VISIBLE_DEVICES="$gpu4" \
      ASTR_FORCE_MPI_TOPOLOGY=1,2,2 ASTR_GPU_RK_TIMING=1 \
      ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_PHASE_TIMING=0 \
      ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_SYNC_MODE=explicit \
      ASTR_GPU_HALO_TRANSPORT=device-aware ASTR_GPU_FILTER_WORKSPACE=full \
      mpirun -np 4 "$NSYS_EXE" profile --trace=cuda,nvtx,osrt,mpi \
        --mpi-impl=openmpi --sample=none --cpuctxsw=none --stats=false \
        --force-overwrite=true -o "$out/trace_rank_%q{OMPI_COMM_WORLD_RANK}" \
        "$GPU_EXE" run datin/input.tgv
  ) > "$log" 2>&1
  status=$?
  (( status == 0 )) || return "$status"
  scan_solver_log "$log" || return 1
  grep -aq 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$log" || return 1
  assert_no_field_hdf5 "$case_dir" || return 1
  mapfile -t reports < <(find "$out" -maxdepth 1 -type f -name 'trace_rank_*.nsys-rep' -print | sort)
  [[ "${#reports[@]}" -eq 4 ]] || return 1
  for report in "${reports[@]}"; do
    stem="${report%.nsys-rep}"
    "$NSYS_EXE" stats --force-export=true --report cuda_gpu_kern_sum "$report" \
      > "${stem}_cuda_gpu_kern_sum.txt" 2>&1 || return $?
    "$NSYS_EXE" stats --force-export=true --report cuda_api_sum "$report" \
      > "${stem}_cuda_api_sum.txt" 2>&1 || return $?
    "$NSYS_EXE" stats --force-export=true --report mpi_event_sum "$report" \
      > "${stem}_mpi_event_sum.txt" 2>&1 || return $?
    [[ -s "${stem}.sqlite" ]] || return 1
    "$PYTHON_EXE" "$OVERLAP_SUMMARY" "${stem}.sqlite" \
      > "${stem}_mpi_kernel_overlap.json" || return $?
  done
}

for entry in "${weak_matrix[@]}"; do
  read -r label np topology grid local_grid backend <<< "$entry"
  run_case weak "$label" run_benchmark weak "$label" "$np" "$topology" "$grid" "$backend" 0
done

for entry in "${phase_matrix[@]}"; do
  read -r label np topology grid local_grid backend <<< "$entry"
  run_case phase "$label" run_benchmark phase "$label" "$np" "$topology" "$grid" "$backend" 1
done

run_case nsys nsys_np4_122_device_aware run_nsys

pass_count="$(awk -F '\t' 'NR>1 && $3=="PASS" {n++} END {print n+0}' "$STATUS_FILE")"
fail_count="$(awk -F '\t' 'NR>1 && $3=="FAIL" {n++} END {print n+0}' "$STATUS_FILE")"
{
  printf 'expected_cases=6\npassed=%s\nfailed=%s\n' "$pass_count" "$fail_count"
  if [[ "$pass_count" -eq 6 && "$fail_count" -eq 0 ]]; then
    printf 'overall_status=PASS\n'
  else
    printf 'overall_status=PARTIAL\n'
  fi
} > "$RUN_ROOT/job_status.txt"

printf 'A800 paper diagnostics complete: pass=%s fail=%s root=%s\n' \
  "$pass_count" "$fail_count" "$RUN_ROOT"
[[ "$pass_count" -eq 6 && "$fail_count" -eq 0 ]]
