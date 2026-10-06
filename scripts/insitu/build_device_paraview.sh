#!/usr/bin/env bash
set -euo pipefail

# Apply patches only to a separate ParaView source copy, never the IS7 baseline.
: "${PV_DEVICE_SOURCE:?Set a separate patched ParaView 6.1.1 source directory}"
: "${PV_DEVICE_BUILD:?Set the separate build directory}"
: "${PV_DEVICE_PREFIX:?Set the separate installation directory}"
: "${CATALYST_DIR:?Set the Catalyst package-config directory}"
: "${MPI_ROOT:?Set the MPI installation directory}"
: "${CUDA_ROOT:?Set the CUDA toolkit directory}"
export OMPI_CC="${PV_DEVICE_CC:-gcc-13}"
export OMPI_CXX="${PV_DEVICE_CXX:-g++-13}"
targets=()
if [[ -n "${PV_DEVICE_TARGETS:-}" ]]; then
  read -r -a targets <<< "$PV_DEVICE_TARGETS"
  if [[ "${PV_DEVICE_INSTALL:-1}" != 0 ]]; then
    printf '%s\n' 'Partial target builds require PV_DEVICE_INSTALL=0; do not install missing modules.' >&2
    exit 2
  fi
fi
python_options=()
device_modules=()
vtkm_plugin=ON
viskores_filters=ON
if [[ "${PV_DEVICE_RENDERING:-OFF}" == ON ]]; then
  # ASTR dispatches its own GPU filters; retain standard VTK device arrays.
  device_modules+=("-DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmCore=YES"
                   "-DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmDataModel=YES"
                   "-DVTK_MODULE_ENABLE_VTK_AcceleratorsVTKmFilters=NO")
  vtkm_plugin=OFF
  viskores_filters=OFF
fi
if [[ -n "${PV_DEVICE_PYTHON_INCLUDE:-}" ]]; then
  python_options+=("-DPython3_INCLUDE_DIR=$PV_DEVICE_PYTHON_INCLUDE")
fi
if [[ -n "${PV_DEVICE_PYTHON_LIBRARY:-}" ]]; then
  python_options+=("-DPython3_LIBRARY=$PV_DEVICE_PYTHON_LIBRARY")
fi

cmake -S "$PV_DEVICE_SOURCE" -B "$PV_DEVICE_BUILD" -G "${CMAKE_GENERATOR:-Unix Makefiles}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PV_DEVICE_PREFIX" \
  -DCMAKE_C_COMPILER="${PV_DEVICE_CC:-gcc-13}" \
  -DCMAKE_CXX_COMPILER="${PV_DEVICE_CXX:-g++-13}" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="${PV_DEVICE_CXX:-g++-13}" \
  -DCMAKE_CUDA_FLAGS="${PV_DEVICE_CUDA_FLAGS:-}" \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -Dcatalyst_DIR="$CATALYST_DIR" \
  -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc" \
  -DMPI_CXX_COMPILER="$MPI_ROOT/bin/mpicxx" \
  -DPython3_EXECUTABLE="${PV_DEVICE_PYTHON:-/usr/bin/python3}" \
  "${python_options[@]}" \
  "${device_modules[@]}" \
  -DPARAVIEW_BUILD_EDITION=CATALYST_RENDERING \
  -DPARAVIEW_BUILD_TESTING=OFF \
  -DPARAVIEW_USE_MPI=ON \
  -DPARAVIEW_USE_PYTHON=ON \
  -DPARAVIEW_ENABLE_CATALYST=ON \
  -DPARAVIEW_ENABLE_WEB=OFF \
  -DPARAVIEW_USE_CUDA=ON \
  -DPARAVIEW_USE_VISKORES="$viskores_filters" \
  -DPARAVIEW_PLUGIN_ENABLE_VTKmFilters="$vtkm_plugin" \
  -DASTR_VISKORES_FP64_MPI=ON \
  -DASTR_VTK_DEVICE_RENDERING="${PV_DEVICE_RENDERING:-OFF}" \
  -DVTK_USE_X=OFF \
  -DVTK_OPENGL_HAS_EGL=ON \
  -DVTK_DEFAULT_RENDER_WINDOW_OFFSCREEN=ON

if [[ "${CONFIGURE_ONLY:-0}" != 1 ]]; then
  if (( ${#targets[@]} )); then
    cmake --build "$PV_DEVICE_BUILD" --target "${targets[@]}" --parallel "${BUILD_JOBS:-2}"
  else
    cmake --build "$PV_DEVICE_BUILD" --parallel "${BUILD_JOBS:-2}"
  fi
  if [[ "${PV_DEVICE_INSTALL:-1}" == 1 ]]; then
    cmake --install "$PV_DEVICE_BUILD"
  fi
fi
