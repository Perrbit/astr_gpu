#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  printf 'Usage: bash build_platform_solver.sh DEPLOYMENT_ENVIRONMENT\n' >&2
  exit 2
fi
source "$1"
: "${DEPLOY_ROOT:?Set the new deployment directory}"
: "${ASTR_PLATFORM_WORK_ROOT:?Set the permitted remote work root}"
: "${INSITU_DEPENDENCY_ROOT:?Set the existing validated device dependency directory}"
: "${CUDA_ROOT:?Set the CUDA toolkit}"
: "${MPI_ROOT:?Set the matching MPI installation}"
: "${HDF5_ROOT:?Set the parallel Fortran HDF5 installation}"
ROOT=$(realpath "$DEPLOY_ROOT")
WORK_ROOT=$(realpath "$ASTR_PLATFORM_WORK_ROOT")
case "$ROOT/" in "$WORK_ROOT"/*) ;; *) exit 2 ;; esac
cd "$ROOT"
JOBS=${BUILD_JOBS:-8}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { printf 'BUILD_JOBS must be positive\n' >&2; exit 2; }
PV="$INSITU_DEPENDENCY_ROOT/build/paraview-device"
common=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_Fortran_COMPILER="$(command -v nvfortran)"
  -DCMAKE_C_COMPILER="$(command -v gcc)" -DCMAKE_CXX_COMPILER="$(command -v g++)"
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$(command -v g++)"
  -DMPI_Fortran_COMPILER="$MPI_ROOT/bin/mpif90" -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc"
  -DMPI_CXX_COMPILER="$MPI_ROOT/bin/mpicxx" -DHDF5_ROOT="$HDF5_ROOT"
  -DHDF5_Fortran_COMPILER_EXECUTABLE="$HDF5_ROOT/bin/h5hlfc"
  -Dcatalyst_DIR="$INSITU_DEPENDENCY_ROOT/opt/catalyst/lib/cmake/catalyst-2.1"
  -DParaView_DIR="$PV/lib/cmake/paraview-6.1" -DVTK_DIR="$PV/lib/cmake/paraview-6.1/vtk"
  -DViskores_DIR="$PV/lib/cmake/paraview-6.1/vtk/viskores"
  -DASTR_INSITU_FLYING_EDGES_SOURCE="$INSITU_DEPENDENCY_ROOT/src/ParaView-v6.1.1/VTK/ThirdParty/viskores/vtkviskores/viskores/viskores/filter/contour/ContourFlyingEdges.cxx"
  -DASTR_NVML_INCLUDE_DIR="$CUDA_ROOT/include"
  -DASTR_WITH_CUDA=ON -DASTR_CUDA_ARCHITECTURES="${ASTR_PLATFORM_CUDA_ARCH:-80}"
  -DASTR_WITH_CATALYST=ON -DASTR_WITH_INSITU_DEVICE=ON -DASTR_WITH_INSITU_DEVICE_RENDERING=ON
  -DASTR_WITH_AIR5_CHEMISTRY=OFF -DCHEMISTRY=OFF
  -DASTR_USERDEFINE_SOURCE="$ROOT/inputs/M12/user_define_module/userdefine_hbl.F90"
)
cmake -S "$ROOT/src/astr" -B "$ROOT/build/astr-debug" "${common[@]}" \
  -DBUILD_TESTING=ON -DASTR_BUILD_INSITU_DEVICE_PROBES=ON -DASTR_BUILD_INSITU_RENDER_PROBES=ON
cmake --build "$ROOT/build/astr-debug" --target astr insitu_device_map_probe \
  insitu_standard_device_probe halo_cuda_aware_probe checkpoint_bundle_probe --parallel "$JOBS"
cmake -S "$ROOT/src/astr" -B "$ROOT/build/astr-native" "${common[@]}" -DBUILD_TESTING=OFF
cmake --build "$ROOT/build/astr-native" --target astr --parallel "$JOBS"
for binary in "$ROOT/build/astr-debug/bin/astr" "$ROOT/build/astr-native/bin/astr" \
              "$ROOT/build/astr-debug/bin/insitu_standard_device_probe" \
              "$PV/lib/catalyst/libcatalyst-paraview.so"; do
  ldd -r "$binary"
done > "$ROOT/logs/login-runtime-dependencies.txt" 2>&1
if grep -E 'not found|undefined symbol' "$ROOT/logs/login-runtime-dependencies.txt"; then
  printf 'Unresolved login-node dependency\n' >&2
  exit 2
fi
if [[ -n "${ASTR_PLATFORM_RENDERING_LIBRARY:-}" ]]; then
  python3 "$ROOT/src/astr/tests/gpu_validation/check_platform_rendering_runtime.py" \
    --expected "$ASTR_PLATFORM_RENDERING_LIBRARY" \
    "$ROOT/build/astr-debug/bin/astr" "$ROOT/build/astr-native/bin/astr" \
    "$ROOT/build/astr-debug/bin/insitu_standard_device_probe" \
    "$PV/lib/catalyst/libcatalyst-paraview.so"
fi
python3 "$ROOT/src/astr/tests/gpu_validation/run_checkpoint_publication_preflight.py" \
  --probe "$ROOT/build/astr-debug/bin/checkpoint_bundle_probe" --mpiexec "$MPI_ROOT/bin/mpiexec" \
  --directory "$ROOT/logs"
sha256sum "$ROOT/build/astr-debug/bin/astr" "$ROOT/build/astr-native/bin/astr" \
  "$ROOT/build/astr-debug/bin/checkpoint_bundle_probe" \
  "$ROOT/build/astr-debug/bin/insitu_standard_device_probe" \
  "$PV/lib/catalyst/libcatalyst-paraview.so" > "$ROOT/logs/solver-binaries.sha256"
python3 "$ROOT/src/astr/tests/gpu_validation/run_platform_insitu_acceptance.py" --help
printf 'ASTR_PLATFORM_SOLVER_BUILD=PASS\n'
