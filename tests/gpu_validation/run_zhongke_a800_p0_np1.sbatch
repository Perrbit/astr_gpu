#!/bin/bash
#SBATCH --job-name=astr-a800-p0-np1
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-p0-np1-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-p0-np1-%j.err

set -euo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR=$WORK_ROOT/astr_gpu_0d684a8
BUILD_DIR=$WORK_ROOT/build_gpu_a800_nvhpc_hdf5_0d684a8
HDF5_ROOT=$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5
RUN_ROOT=$WORK_ROOT/results/a800_p0_np1_0d684a8_nvhpc_hdf5

cd "$WORK_ROOT"
module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:$LD_LIBRARY_PATH"

mkdir -p "$RUN_ROOT"

GPU_EXE="$BUILD_DIR/bin/astr"
test -x "$GPU_EXE"

{
  printf 'source_commit: '
  cat "$SRC_DIR/SOURCE_COMMIT"
  printf 'date: %s\n' "$(date -Iseconds)"
  printf 'hostname: %s\n' "$(hostname)"
  printf 'slurm_job_id: %s\n' "$SLURM_JOB_ID"
  printf 'slurm_job_nodelist: %s\n' "$SLURM_JOB_NODELIST"
  printf 'cuda_visible_devices: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf '\nloaded_modules:\n'
  module -t list 2>&1
  printf '\nnvidia_smi:\n'
  nvidia-smi
  printf '\nnvfortran:\n'
  nvfortran --version
  printf '\nmpirun:\n'
  mpirun --version
  printf '\nhdf5:\n'
  h5pfc -showconfig
  printf '\ncmake:\n'
  cmake --version
} > "$RUN_ROOT/environment.txt"

if grep -E -- '(--use_fast_math|-Mfprelaxed|(^|[[:space:]])-fast([[:space:]]|$))' \
  "$BUILD_DIR/compile_commands.json"; then
  printf '%s\n' 'prohibited relaxed-math compiler option found' >&2
  exit 2
fi

ALLOCATED_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
GPU_EXE="$GPU_EXE" \
OUT_DIR="$RUN_ROOT/benchmark" \
LABEL=a800_np1_0d684a8 \
GRID=256,256,256 \
MAXSTEP=10 \
REPEATS=5 \
DISCARD_STEPS=1 \
GPU_ID="$ALLOCATED_GPU" \
SYNC_MODE=explicit \
FEQCHKPT=9999 \
  "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"

printf 'status: pass\n' > "$RUN_ROOT/job_status.txt"
