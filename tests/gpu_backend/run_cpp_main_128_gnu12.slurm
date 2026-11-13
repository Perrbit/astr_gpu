#!/bin/bash
#SBATCH --job-name=astr-cpp-main128
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=dcu:Hygon:1
#SBATCH --time=00:10:00
#SBATCH --output=cpp-main128-%j.log
set -e -o pipefail
cd "$SLURM_SUBMIT_DIR"
work=$PWD
gcc=/public/software/compiler/gcc-12.2.0
mpi=/public/software/mpi/openmpi/openmpi-4.1.5
dtk=/public/software/compiler/rocm/dtk-26.04
hdf5=$(cd ../hdf5-1.14.6 && pwd)
mkdir -p "$work/home" "$work/tmp/$SLURM_JOB_ID" "$work/cache/comgr" "$work/results"
export HOME="$work/home" TMPDIR="$work/tmp/$SLURM_JOB_ID"
export XDG_CACHE_HOME="$work/cache" AMD_COMGR_CACHE_DIR="$work/cache/comgr"
export LC_ALL=C LANG=C OMP_NUM_THREADS=1
unset CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:$hdf5/lib:/public/software/mpi/openmpi/hwloc/lib:/public/software/mathlib/zlib/1.2.8/gcc/lib"
source "$dtk/env.sh"
source "$dtk/cuda/env.sh"
export OPAL_PREFIX="$mpi"
export OMPI_CC="$gcc/bin/gcc" OMPI_CXX="$gcc/bin/g++" OMPI_FC="$gcc/bin/gfortran"
export PATH="$gcc/bin:$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:$hdf5/lib:$LD_LIBRARY_PATH"
export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt
export OMPI_MCA_coll=^cuda,hcoll,ucc OMPI_MCA_opal_cuda_support=0
export ASTR_GPU_SYNC_MODE=dependency ASTR_GPU_FILTER_WORKSPACE=full
export ASTR_FORCE_MPI_TOPOLOGY=1,1,1 ASTR_CFL_PROFILE_Y=1
export ASTR_VALIDATION_RHS_PREFIX=outdat/rhs ASTR_VALIDATION_RHS_STEP=20
date -Is
hostname
printf 'CPP-MAIN128 job=%s allocated=%s CUDA=%s HIP=%s ROCR=%s\n' \
  "$SLURM_JOB_ID" "${SLURM_JOB_GPUS:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}" \
  "${HIP_VISIBLE_DEVICES:-unset}" "${ROCR_VISIBLE_DEVICES:-unset}"
ldd "$work/build/bin/astr"
"$mpi/bin/mpiexec" --bind-to none -n 1 "$work/build/bin/astr_backend_fluid" \
  2>&1 | tee "$work/results/main128-diagnostics-$SLURM_JOB_ID.log"
for mode in fixed automatic; do
  printf 'CPP-MAIN128 starting %s\n' "$mode"
  (cd "$work/dcu_cases_128_20/$mode"
   "$mpi/bin/mpiexec" --bind-to none -n 1 "$work/build/bin/astr" run) \
    2>&1 | tee "$work/dcu_cases_128_20/$mode/run.log"
done
date -Is
