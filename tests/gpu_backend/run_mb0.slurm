#!/bin/bash
#SBATCH --job-name=astr-backend-mb0
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:05:00
#SBATCH --output=mb0-%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
work=$PWD
mkdir -p "$work/home" "$work/tmp/$SLURM_JOB_ID" "$work/cache/comgr" "$work/results"
export HOME="$work/home" TMPDIR="$work/tmp/$SLURM_JOB_ID"
export XDG_CACHE_HOME="$work/cache" AMD_COMGR_CACHE_DIR="$work/cache/comgr"
export LC_ALL=C LANG=C OMP_NUM_THREADS=1
dtk=/public/software/compiler/rocm/dtk-26.04
mpi=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1
hdf5=/public/software/mathlib/hdf5/1.12.2/gnu_fortran_parallel
source "$dtk/env.sh"
source "$dtk/cuda/env.sh"
export PATH="$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$mpi/lib:$hdf5/lib:$LD_LIBRARY_PATH"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt
export OMPI_MCA_coll=^cuda,hcoll,ucc OMPI_MCA_opal_cuda_support=0
date -Is
hostname
printf 'MB0 job=%s allocated=%s CUDA=%s HIP=%s ROCR=%s\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}" \
  "${HIP_VISIBLE_DEVICES:-unset}" "${ROCR_VISIBLE_DEVICES:-unset}"
if "$dtk/bin/rocm-smi" --showdriverversion; then
  printf 'MB0 driver-monitor status=0\n'
else
  printf 'MB0 driver-monitor status=%s; driver metadata unavailable\n' "$?"
fi
ldd "$work/login09/build/bin/astr_backend_mb0"
"$mpi/bin/mpiexec" --bind-to none -n 1 "$work/login09/build/bin/astr_backend_mb0" \
  "$work/results/mb0-$SLURM_JOB_ID.h5"
date -Is
