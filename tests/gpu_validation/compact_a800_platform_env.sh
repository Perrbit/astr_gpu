#!/usr/bin/env bash
# Source only after DEPLOY_DIR points to this campaign's private directory.
WORK_ROOT=/data/user/hd56000/weiph
case "$DEPLOY_DIR/" in "$WORK_ROOT"/*) ;; *) printf 'Deployment escapes work root\n' >&2; return 2 ;; esac
module purge
module load nvhpc/25.5 gcc/13.3 cmake/3.27.9
export CUDA_ROOT=/data/apps/nvhpc/25.5/Linux_x86_64/25.5/cuda/12.9
export NVHPC_CUDA_HOME="$CUDA_ROOT"
export NVCOMPILER_MATH_LIBS_HOME="$NVHPC_ROOT/math_libs/12.9"
export MPI_ROOT=/data/apps/nvhpc/25.5/Linux_x86_64/25.5/comm_libs/12.9/hpcx/hpcx-2.22.1/ompi
export HDF5_ROOT="$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip"
export PYTHON_EXE="$WORK_ROOT/venv_tgv_campaign_py311/bin/python"
export PATH="$MPI_ROOT/bin:$CUDA_ROOT/bin:$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:$MPI_ROOT/lib:$NVCOMPILER_MATH_LIBS_HOME/lib64:$CUDA_ROOT/lib64:/data/apps/gcc/13.3/lib64:${LD_LIBRARY_PATH:-}"
export OMPI_CC=gcc OMPI_CXX=g++ OMPI_FC=nvfortran
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$DEPLOY_DIR/tmp/${SLURM_JOB_ID:-login}"
export CUDA_CACHE_PATH="$TMPDIR/cuda-cache" XDG_CACHE_HOME="$TMPDIR/cache"
export PYTHONPYCACHEPREFIX="$TMPDIR/python-cache" OMPI_MCA_orte_tmpdir_base="$TMPDIR"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX"
unset PYTHONPATH
