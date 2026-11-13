#!/bin/bash
set -e -o pipefail
work=$1
dtk=$2
gcc=$3
ucx_archive=$(readlink -f "$4")
mpi_archive=$(readlink -f "$5")
thunk_libdir=${6:?ROCt shared-library directory is required}
mkdir -p "$work/home" "$work/tmp" "$work/cache" "$work/source" "$work/build_ucx" "$work/build_mpi"
work=$(cd "$work" && pwd)
export HOME="$work/home" TMPDIR="$work/tmp" XDG_CACHE_HOME="$work/cache"
export LC_ALL=C LANG=C
unset CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OPAL_PREFIX
unset OMPI_CC OMPI_CXX OMPI_FC OMPI_MCA_mca_base_component_path OMPI_MCA_mca_base_param_files
unset OMPI_MCA_pml OMPI_MCA_btl OMPI_MCA_coll UCX_MODULE_DIR UCX_CONFIG_FILE UCX_TLS
source "$dtk/env.sh"
export PATH="$gcc/bin:$PATH"
export CC="$gcc/bin/gcc" CXX="$gcc/bin/g++" FC="$gcc/bin/gfortran" F77="$gcc/bin/gfortran"
# DTK's outer HSA include links target /opt/hyhal on compute images.
# Use its shipped headers explicitly when compiling on login nodes.
export CPPFLAGS="-I$dtk/.hyhal/include/hsa -I$dtk/hip/include"
export LD_LIBRARY_PATH="$gcc/lib64:$dtk/.hyhal/lib:$dtk/.hyhal/lib64:$dtk/hip/lib:$dtk/hsa/lib:$dtk/lib64:$dtk/lib:$thunk_libdir"
export LDFLAGS="-L$thunk_libdir -Wl,-rpath,$dtk/.hyhal/lib -Wl,-rpath,$thunk_libdir -Wl,-rpath,$gcc/lib64"
export CFLAGS='-O3 -g' CXXFLAGS='-O3 -g'
export FCFLAGS='-O2 -g -fallow-argument-mismatch' FFLAGS="$FCFLAGS"
ucx="$work/ucx-1.18.1"
mpi="$work/openmpi-4.1.5"
tar -xzf "$ucx_archive" -C "$work/source"
tar -xzf "$mpi_archive" -C "$work/source"
cd "$work/build_ucx"
"$work/source/ucx-1.18.1/configure" --prefix="$ucx" --with-rocm="$dtk" \
  --without-cuda --without-java --without-knem --without-xpmem \
  --enable-mt --enable-optimizations --disable-debug --disable-assertions --disable-static
grep '^#define HAVE_ROCM 1' config.h
grep '^#define HAVE_HIP 1' config.h
make -j4
make install
export LD_LIBRARY_PATH="$ucx/lib:$LD_LIBRARY_PATH"
"$ucx/bin/ucx_info" -v
"$ucx/bin/ucx_info" -b
export PKG_CONFIG_PATH="$ucx/lib/pkgconfig:$dtk/.hyhal/share/pkgconfig"
export LDFLAGS="-L$ucx/lib -Wl,-rpath,$ucx/lib"
cd "$work/build_mpi"
"$work/source/openmpi-4.1.5/configure" --prefix="$mpi" --with-ucx="$ucx" \
  --without-cuda --without-verbs --without-hcoll --with-hwloc=/public/software/mpi/openmpi/hwloc \
  --with-libevent=internal --with-pmix=internal --with-slurm \
  --enable-mpi-fortran=all --disable-oshmem --disable-static
make -j4
make install
export OPAL_PREFIX="$mpi"
export LD_LIBRARY_PATH="$mpi/lib:$LD_LIBRARY_PATH"
"$mpi/bin/ompi_info" --parsable --all
ldd "$mpi/lib/openmpi/mca_pml_ucx.so"
find "$ucx/lib" -name '*rocm*.so*' -print
