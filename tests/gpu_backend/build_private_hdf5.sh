#!/bin/bash
set -e
work=$1
gcc=$2
mpi=$3
cmake=$4
archive=$5
mkdir -p "$work/home" "$work/tmp" "$work/cache" "$work/source"
export HOME="$work/home" TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache"
export LC_ALL=C LANG=C
unset CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export PATH="$gcc/bin:$mpi/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:/public/software/mpi/openmpi/hwloc/lib:/public/software/mathlib/zlib/1.2.8/gcc/lib"
export OPAL_PREFIX="$mpi"
export OMPI_CC="$gcc/bin/gcc" OMPI_CXX="$gcc/bin/g++" OMPI_FC="$gcc/bin/gfortran"
"$mpi/bin/mpifort" --showme
tar -xzf "$archive" -C "$work/source"
"$cmake" -S "$work/source/hdf5-1.14.6" -B "$work/build_hdf5" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$work/hdf5-1.14.6" \
  -DHDF5_INSTALL_LIB_DIR=lib \
  -DCMAKE_C_COMPILER="$mpi/bin/mpicc" -DCMAKE_Fortran_COMPILER="$mpi/bin/mpifort" \
  -DMPI_C_COMPILER="$mpi/bin/mpicc" -DMPI_Fortran_COMPILER="$mpi/bin/mpifort" \
  -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
  -DHDF5_ENABLE_PARALLEL=ON -DHDF5_BUILD_FORTRAN=ON \
  -DHDF5_BUILD_HL_LIB=ON -DHDF5_BUILD_CPP_LIB=OFF \
  -DBUILD_TESTING=OFF -DHDF5_BUILD_EXAMPLES=OFF -DHDF5_BUILD_TOOLS=OFF \
  -DHDF5_ENABLE_Z_LIB_SUPPORT=ON -DZLIB_ROOT=/public/software/mathlib/zlib/1.2.8/gcc \
  -DHDF5_ENABLE_SZIP_SUPPORT=OFF -DHDF5_ALLOW_EXTERNAL_SUPPORT=NO
"$cmake" --build "$work/build_hdf5" --parallel 4
"$cmake" --install "$work/build_hdf5"
cat "$work/hdf5-1.14.6/lib/libhdf5.settings"
