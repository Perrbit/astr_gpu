#!/bin/bash
# Build the two variants in one dedicated, incremental root-CMake directory.
set -eo pipefail
work=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
cd "$work"
source "$work/stack.sh"
sha256sum -c "$diag/ucx122_stack_pair/private_mpi.sha256"
sha256sum -c "$diag/ucx122_private/private_ucx.sha256"
mkdir -p bin

publish() {
    local name=$1
    install -m 0555 build/bin/astr "bin/astr_$name"
    ldd -r "bin/astr_$name" "$mpi/lib/openmpi/mca_pml_ucx.so" "$ucx/lib/ucx/libuct_rocm.so" > "$name.ldd" 2>&1
    if grep -Eq 'not found|undefined symbol' "$name.ldd"; then
        cat "$name.ldd"
        exit 1
    fi
    awk -v lib="$ucx/lib/" '$1 ~ /^libu(cp|ct|cs|cm)\.so/ {n++; if (index($3,lib)!=1) bad=1}
        END{exit(n<4||bad)}' "$name.ldd"
    sha256sum "bin/astr_$name" | tee "$name.sha256"
}

(cd source && git apply --check "$work/baseline_wall_timer.patch" && git apply "$work/baseline_wall_timer.patch")
"$cmake" -S source -B build \
    -DCMAKE_TOOLCHAIN_FILE="$work/source/tests/gpu_backend/gpufusion.cmake" \
    -DCMAKE_Fortran_COMPILER="$mpi/bin/mpifort" -DCMAKE_C_COMPILER="$gcc/bin/gcc" \
    -DCMAKE_CXX_COMPILER="$gcc/bin/g++" -DMPI_C_COMPILER="$mpi/bin/mpicc" \
    -DMPIEXEC_EXECUTABLE="$mpi/bin/mpiexec" -DCMAKE_CUDA_FLAGS=--offload-arch=gfx906 \
    -DHDF5_Fortran_COMPILER_EXECUTABLE="$hdf/bin/h5hlfc" -DCMAKE_BUILD_TYPE=Release \
    -DASTR_BUILD_GPU_BACKEND_PROBES=OFF -DASTR_WITH_CUDA=OFF \
    -DASTR_GPU_CPP_BACKEND=gpufusion -DASTR_WITH_CATALYST=OFF \
    -DASTR_WITH_AIR5_CHEMISTRY=ON -DASTR_NUMERICAL_PRECISION=fp64 \
    -DBUILD_TESTING=ON -DASTR_MB8_TEST_NUMERICS=OFF > configure.log 2>&1
"$cmake" --build build --target astr --parallel 4 > baseline_build.log 2>&1
publish baseline

(cd source && git apply --check "$work/regional_only.patch" && git apply "$work/regional_only.patch")
"$cmake" --build build --target astr --parallel 4 > regional_build.log 2>&1
publish regional
