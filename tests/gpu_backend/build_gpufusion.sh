#!/bin/bash
set -e
work=$1
dtk=$2
mpi=$3
hdf5_compiler=${4:?HDF5 Fortran compiler wrapper is required}
hdf5=$(cd "$(dirname "$hdf5_compiler")/.." && pwd)
cmake=$5
target=$6
gcc=${7:?GNU compiler root is required}
source_root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$work/home" "$work/tmp" "$work/cache/comgr" "$work/results"
export HOME="$work/home" TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache"
export AMD_COMGR_CACHE_DIR="$work/cache/comgr" LC_ALL=C LANG=C
unset CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:$hdf5/lib:/public/software/mathlib/zlib/1.2.8/gcc/lib"
source "$dtk/env.sh"
source "$dtk/cuda/env.sh"
export OPAL_PREFIX="$mpi"
export OMPI_CC="$gcc/bin/gcc" OMPI_CXX="$gcc/bin/g++" OMPI_FC="$gcc/bin/gfortran"
export PATH="$gcc/bin:$mpi/bin:$PATH"
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:$hdf5/lib:$LD_LIBRARY_PATH"
"$mpi/bin/mpifort" --showme
"$cmake" -S "$source_root" -B "$work/build" \
  -DCMAKE_TOOLCHAIN_FILE="$source_root/tests/gpu_backend/gpufusion.cmake" \
  -DCMAKE_Fortran_COMPILER="$mpi/bin/mpifort" \
  -DCMAKE_C_COMPILER="$gcc/bin/gcc" -DCMAKE_CXX_COMPILER="$gcc/bin/g++" \
  -DMPI_C_COMPILER="$mpi/bin/mpicc" \
  -DCMAKE_CUDA_FLAGS="--offload-arch=$target" \
  -DHDF5_Fortran_COMPILER_EXECUTABLE="$hdf5_compiler" \
  -DASTR_BUILD_GPU_BACKEND_PROBES=ON -DASTR_WITH_CUDA=OFF \
  -DASTR_GPU_CPP_BACKEND=gpufusion \
  -DASTR_WITH_CATALYST=OFF -DASTR_WITH_AIR5_CHEMISTRY=OFF -DBUILD_TESTING=ON
"$cmake" --build "$work/build" --target astr astr_backend_mb0 astr_backend_mb1 astr_backend_fluid astr_backend_rhs --parallel 4 --verbose
ldd "$work/build/bin/astr_backend_mb0"
