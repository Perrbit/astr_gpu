#!/bin/bash
#SBATCH --job-name=astr-a800-devmpi
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-devmpi-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-devmpi-%j.err

set -euo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cuda_aware_qualification_c801eeb}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_cuda_aware_qualification_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_device_aware_solver_admission_${SLURM_JOB_ID:-dryrun}}"
MPI_NP="${MPI_NP:-2}"
QUALIFICATION_JSON="$WORK_ROOT/results/a800_cuda_aware_qualification_460370/qualification.json"
EXE="$BUILD_DIR/bin/astr"
SUPPRESSIONS="$SRC_DIR/tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml"

for path in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$RUN_ROOT" "$QUALIFICATION_JSON"; do
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
export OMPI_MCA_pml=ucx
export OMPI_MCA_coll='^hcoll,ucc,cuda'
export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_sharedfp=individual
export UCX_MEMTYPE_CACHE=n
export UCX_CUDA_COPY_ENABLE_FABRIC=no
export UCX_CUDA_COPY_DMABUF=no
export UCX_CUDA_IPC_ENABLE_MNNVL=no
export UCX_TLS=self,sm,cuda_copy,cuda_ipc

for required in "$EXE" "$PYTHON_EXE" "$SUPPRESSIONS" "$QUALIFICATION_JSON" \
  "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  "$SRC_DIR/tests/gpu_validation/compare_tgv_flowstate.py"; do
  [[ -f "$required" ]] || { printf 'missing solver admission input: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$EXE" ]] || { printf 'ASTR is not executable: %s\n' "$EXE" >&2; exit 2; }
[[ -x "$PYTHON_EXE" ]] || { printf 'Python is not executable: %s\n' "$PYTHON_EXE" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { printf 'refusing to overwrite: %s\n' "$RUN_ROOT" >&2; exit 2; }
if ldd "$EXE" | grep -q 'not found'; then
  printf 'unresolved compute-node ASTR dependency\n' >&2
  ldd "$EXE" >&2
  exit 2
fi
mapfile -t flag_files < <(find "$BUILD_DIR" -type f \( -name 'flags.make' -o -name 'link.txt' \) -print)
[[ "${#flag_files[@]}" -gt 0 ]] || { printf 'missing CMake compiler flag manifests\n' >&2; exit 2; }
if grep -H -E -- '(--use_fast_math|-ffast-math|-Ofast|-Mfprelaxed|-Mfpapprox|-gpu=[^[:space:]]*fastmath|(^|[[:space:]])-fast([[:space:]]|$))' \
  "${flag_files[@]}"; then
  printf 'prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi

"$PYTHON_EXE" - "$QUALIFICATION_JSON" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1], encoding="ascii"))
if data.get("production_qualified") is not True:
    raise SystemExit("A800 CUDA-aware MPI qualification is not production_qualified")
PY

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq "$MPI_NP" ]] || {
  printf 'expected %s Slurm-visible GPUs: %s\n' "$MPI_NP" "$visible" >&2
  exit 2
}
case "$MPI_NP" in
  2)
    topology_specs=('x 2,1,1' 'y 1,2,1' 'z 1,1,2')
    sanitizer_topology=2,1,1
    ;;
  4)
    topology_specs=('xy 2,2,1' 'xz 2,1,2' 'yz 1,2,2')
    sanitizer_topology=2,2,1
    ;;
  *)
    printf 'solver admission supports MPI_NP=2 or 4, got %s\n' "$MPI_NP" >&2
    exit 2
    ;;
esac
mkdir -p "$RUN_ROOT"
{
  date -Iseconds
  hostname
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$visible"
  printf 'MPI_NP=%s\n' "$MPI_NP"
  printf 'UCX_TLS=%s\n' "$UCX_TLS"
  sha256sum "$EXE" "$SUPPRESSIONS"
  mpirun --version
  nvidia-smi
  nvidia-smi topo -m
  ldd "$EXE"
} > "$RUN_ROOT/environment.txt" 2>&1

prepare_case() {
  local destination="$1"
  local maxstep="${2:-5}"
  "$PYTHON_EXE" "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" \
    --dst-case "$destination" --use-gpu t --maxstep "$maxstep" \
    --feqchkpt 9999 --feqlist 1 --lfilter t --diffterm t \
    --scheme 643e --grid 128,128,128 --lreadgrid f
  rm -f "$destination"/datin/grid.h5
}

assert_no_field_hdf5() {
  local destination="$1"
  local found
  found="$(find "$destination" -type f \( -name 'grid*.h5' -o -name 'flowfield*.h5' \) -print)"
  if [[ -n "$found" ]]; then
    printf 'forbidden field HDF5 output:\n%s\n' "$found" >&2
    return 1
  fi
}

scan_solver_log() {
  local log="$1"
  grep -aq 'The job is done!' "$log"
  grep -aq 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$log"
  if grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    return 1
  fi
}

run_case() {
  local destination="$1"
  local topology="$2"
  local backend="$3"
  local protocol_info="$4"
  prepare_case "$destination"
  (
    cd "$destination"
    export ASTR_FORCE_MPI_TOPOLOGY="$topology"
    export ASTR_GPU_HALO_TRANSPORT="$backend"
    export ASTR_GPU_SYNC_MODE=explicit
    export ASTR_GPU_RK_TIMING=1
    export ASTR_GPU_BENCHMARK_NO_FIELD_IO=1
    export UCX_PROTO_INFO="$protocol_info"
    mpirun -np "$MPI_NP" "$EXE" run datin/input.tgv > run.log 2>&1
  )
  scan_solver_log "$destination/run.log"
  assert_no_field_hdf5 "$destination"
}

for spec in "${topology_specs[@]}"; do
  read -r axis topology <<< "$spec"
  reference="$RUN_ROOT/${axis}_pinned"
  candidate="$RUN_ROOT/${axis}_device_aware"
  run_case "$reference" "$topology" pinned n
  run_case "$candidate" "$topology" device-aware y
  grep -q 'ASTR_GPU_PRE_MPI_DEVICE=' "$candidate/run.log"
  grep -q 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$candidate/run.log"
  grep -q 'cuda_ipc/cuda' "$candidate/run.log"
  "$PYTHON_EXE" "$SRC_DIR/tests/gpu_validation/compare_tgv_flowstate.py" \
    --reference "$reference" --candidate "$candidate" \
    --report "$RUN_ROOT/${axis}_flowstate_compare.txt" \
    --min-rows 5 --atol 1e-10 --rtol 1e-10
done

sanitizer_case="$RUN_ROOT/x_device_aware_sanitizer"
prepare_case "$sanitizer_case" 1
(
  cd "$sanitizer_case"
  export ASTR_FORCE_MPI_TOPOLOGY="$sanitizer_topology"
  export ASTR_GPU_HALO_TRANSPORT=device-aware
  export ASTR_GPU_SYNC_MODE=explicit
  export ASTR_GPU_RK_TIMING=1
  export ASTR_GPU_BENCHMARK_NO_FIELD_IO=1
  export UCX_PROTO_INFO=n
  compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes all \
    --suppressions "$SUPPRESSIONS" \
    mpirun -np "$MPI_NP" "$EXE" run datin/input.tgv > sanitizer.log 2>&1
)
grep -q 'ERROR SUMMARY: 0 errors' "$sanitizer_case/sanitizer.log"
grep -q 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$sanitizer_case/sanitizer.log"
grep -q 'ASTR_GPU_BENCHMARK_NO_FIELD_IO enabled' "$sanitizer_case/sanitizer.log"
assert_no_field_hdf5 "$sanitizer_case"

printf 'A800_DEVICE_AWARE_SOLVER_ADMISSION=PASS\n'
