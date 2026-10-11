#!/usr/bin/env bash
set -euo pipefail

: "${DEPLOY_ROOT:?Set a new deployment directory below the permitted work root}"
: "${ASTR_PLATFORM_WORK_ROOT:?Set the permitted remote work root}"
: "${CUDA_ROOT:?Set the CUDA toolkit used by both compilers}"
: "${MPI_ROOT:?Set the matching MPI installation}"
: "${PV_DEVICE_PYTHON:?Set the Python interpreter with development headers}"
ROOT=$(realpath "$DEPLOY_ROOT")
WORK_ROOT=$(realpath "$ASTR_PLATFORM_WORK_ROOT")
case "$ROOT/" in
  "$WORK_ROOT"/*) ;;
  *) printf 'Deployment escapes permitted work root: %s\n' "$ROOT" >&2; exit 2 ;;
esac
export TMPDIR="$ROOT/tmp"
mkdir -p "$TMPDIR"
export OMPI_CC="${PV_DEVICE_CC:-gcc}"
export OMPI_CXX="${PV_DEVICE_CXX:-g++}"
JOBS="${BUILD_JOBS:-2}"
[[ "$JOBS" =~ ^[1-8]$ ]] || { printf 'BUILD_JOBS must be between 1 and 8\n' >&2; exit 2; }

CATALYST_SOURCE="$ROOT/src/catalyst-v2.1.0"
export PV_DEVICE_SOURCE="$ROOT/src/ParaView-v6.1.1"
export PV_DEVICE_BUILD="$ROOT/build/paraview-device"
export PV_DEVICE_PREFIX="$ROOT/opt/paraview-device"
ASTR_SOURCE="$ROOT/src/astr"
export CATALYST_DIR="$ROOT/opt/catalyst/lib/cmake/catalyst-2.1"
for archive in catalyst-v2.1.0.tar.gz ParaView-v6.1.1.tar.xz; do
  [[ -f "$ROOT/downloads/$archive" ]] || { printf 'Missing archive: %s\n' "$archive" >&2; exit 2; }
done
if [[ ! -d "$CATALYST_SOURCE" ]]; then
  tar -xf "$ROOT/downloads/catalyst-v2.1.0.tar.gz" -C "$ROOT/src"
fi
if [[ ! -d "$PV_DEVICE_SOURCE" ]]; then
  tar -xf "$ROOT/downloads/ParaView-v6.1.1.tar.xz" -C "$ROOT/src"
fi
if [[ -f "$ROOT/logs/paraview-patches.sha256" ]]; then
  sha256sum --check "$ROOT/logs/paraview-patches.sha256"
else
for name in streamline-fp64 viskores-fp64-mpi streamline-accepted-length \
            device-rendering device-core-dependencies pixel-audit \
            device-buffer-capacity allocation-preflight shared-allocation-lifecycle; do
  patch_file="$ASTR_SOURCE/scripts/insitu/patches/paraview-6.1.1-$name.patch"
  if patch --dry-run --reverse --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$patch_file" > "$ROOT/logs/patch-$name.log" 2>&1; then
    printf 'Patch already applied: %s\n' "$name"
  else
    patch --dry-run --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$patch_file"
    patch --fuzz=0 -p1 -d "$PV_DEVICE_SOURCE" -i "$patch_file"
  fi
done
sha256sum "$ASTR_SOURCE"/scripts/insitu/patches/paraview-6.1.1-*.patch > "$ROOT/logs/paraview-patches.sha256"
fi
cmake -S "$CATALYST_SOURCE" -B "$ROOT/build/catalyst" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$ROOT/opt/catalyst" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_C_COMPILER="$OMPI_CC" -DCMAKE_CXX_COMPILER="$OMPI_CXX" \
  -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc" \
  -DCATALYST_USE_MPI=ON -DCATALYST_BUILD_TESTING=OFF -DCATALYST_BUILD_TOOLS=OFF
cmake --build "$ROOT/build/catalyst" --parallel "$JOBS"
cmake --install "$ROOT/build/catalyst"
export PV_DEVICE_CC="$OMPI_CC" PV_DEVICE_CXX="$OMPI_CXX"
export PV_DEVICE_RENDERING=ON PV_DEVICE_INSTALL=0
export PV_DEVICE_CUDA_ARCHITECTURES="${PV_DEVICE_CUDA_ARCHITECTURES:-80}"
export PV_DEVICE_TARGETS='catalyst-paraview pvpython paraview_all_python_modules viskores_filter_core viskores_filter_vector_analysis'
export BUILD_JOBS="$JOBS"
export PV_DEVICE_PYTHON_INCLUDE
PV_DEVICE_PYTHON_INCLUDE=$($PV_DEVICE_PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')
export PV_DEVICE_PYTHON_LIBRARY
PV_DEVICE_PYTHON_LIBRARY=$($PV_DEVICE_PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR")+"/libpython"+sysconfig.get_python_version()+".so")')
bash "$ASTR_SOURCE/scripts/insitu/build_device_paraview.sh"
printf 'ASTR_PLATFORM_DEVICE_DEPENDENCIES_BUILD=PASS\n'
