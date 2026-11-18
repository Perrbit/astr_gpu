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
  -DMPIEXEC_EXECUTABLE="$mpi/bin/mpiexec" \
  -DCMAKE_CUDA_FLAGS="--offload-arch=$target" \
  -DHDF5_Fortran_COMPILER_EXECUTABLE="$hdf5_compiler" \
  -DASTR_BUILD_GPU_BACKEND_PROBES=ON -DASTR_WITH_CUDA=OFF \
  -DASTR_GPU_CPP_BACKEND=gpufusion \
  -DASTR_WITH_CATALYST=OFF -DASTR_WITH_AIR5_CHEMISTRY=OFF -DBUILD_TESTING=ON
"$cmake" --build "$work/build" --target astr astr_backend_mb0 astr_backend_mb1 astr_backend_fluid astr_backend_rhs \
  astr_backend_halo astr_backend_compare astr_backend_compact_line astr_backend_compact_distributed \
  astr_backend_exterior_initialization astr_backend_shock_sensor astr_backend_shock_mask_halo \
  astr_backend_upwind astr_backend_profile_boundary astr_backend_nscbc_outlet astr_backend_nscbc_inlet \
  astr_backend_nscbc_filter astr_backend_nscbc_top_filter astr_backend_nscbc_top astr_backend_nscbc_top_viscous \
  astr_backend_flat_statistics astr_backend_slip_wall astr_backend_rti astr_backend_adaptive_energy \
  astr_backend_native_curve_gradient \
  astr_backend_crashfix astr_backend_mp_positivity_math astr_backend_mp_positivity_stage \
  astr_backend_critical_flux astr_backend_critical_history astr_backend_critical_shared_flux \
  astr_backend_mixed_viscous astr_backend_mixed_derivative astr_backend_production_statistics \
  astr_backend_keep_host_guard astr_backend_keep astr_backend_me6 astr_backend_wall41 \
  astr_backend_boundary_gradient astr_backend_wall41_step astr_backend_channel astr_backend_boundary_filter \
  astr_backend_boundary_flux astr_backend_insitu_sampling astr_backend_insitu_index \
  astr_backend_insitu_vectors astr_backend_insitu_moments astr_backend_insitu_selected --parallel 4 --verbose
ldd "$work/build/bin/astr_backend_mb0"
