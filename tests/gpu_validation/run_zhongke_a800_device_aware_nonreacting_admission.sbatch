#!/bin/bash
#SBATCH --job-name=astr-a800-devmpi-nr
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-devmpi-nr-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-devmpi-nr-%j.err

set -euo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_gpu_probe}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_device_aware_nonreacting_${SLURM_JOB_ID:-dryrun}}"
QUALIFICATION_JSON="${QUALIFICATION_JSON:-$WORK_ROOT/results/a800_cuda_aware_qualification_460370/qualification.json}"
EXE="$BUILD_DIR/bin/astr"
MATRIX_DRIVER="$SRC_DIR/tests/gpu_validation/run_device_aware_nonreacting_matrix.sh"

for path in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$PYTHON_EXE" "$RUN_ROOT" "$QUALIFICATION_JSON"; do
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

for required in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$PYTHON_EXE" \
  "$QUALIFICATION_JSON" "$EXE" "$MATRIX_DRIVER"; do
  [[ -e "$required" ]] || { printf 'missing nonreacting admission input: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$PYTHON_EXE" ]] || { printf 'Python is not executable: %s\n' "$PYTHON_EXE" >&2; exit 2; }
[[ -x "$EXE" ]] || { printf 'ASTR is not executable: %s\n' "$EXE" >&2; exit 2; }
[[ -x "$MATRIX_DRIVER" ]] || { printf 'matrix driver is not executable: %s\n' "$MATRIX_DRIVER" >&2; exit 2; }
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
[[ "${#gpu[@]}" -eq 2 ]] || {
  printf 'expected two Slurm-visible GPUs: %s\n' "$visible" >&2
  exit 2
}

mkdir -p "$(dirname "$RUN_ROOT")"
environment_record="${RUN_ROOT}.environment.txt"
{
  date -Iseconds
  hostname
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$visible"
  printf 'UCX_TLS=%s\n' "$UCX_TLS"
  sha256sum "$EXE" "$MATRIX_DRIVER"
  mpirun --version
  nvidia-smi
  nvidia-smi topo -m
  ldd "$EXE"
} > "$environment_record" 2>&1

CPU_EXE="$EXE" GPU_EXE="$EXE" OUT_DIR="$RUN_ROOT" \
GPU_HALO_TRANSPORT=device-aware GPU_UCX_PROTO_INFO=y GPU_SYNC_MODE=explicit \
REQUIRE_CUDA_IPC=t \
  "$MATRIX_DRIVER"

grep -qx 'NONREACTING_DEVICE_AWARE_MATRIX=PASS' "$RUN_ROOT/matrix_summary.txt"
printf 'A800_DEVICE_AWARE_NONREACTING_ADMISSION=PASS\n'
