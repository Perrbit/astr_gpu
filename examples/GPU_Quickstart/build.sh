#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build_prod}"
AIR5="${AIR5:-OFF}"
CUDA="${CUDA:-ON}"
FC="${FC:-nvfortran}"
JOBS="${JOBS:-2}"
case "$AIR5" in ON|OFF) ;; *) echo "AIR5 must be ON or OFF" >&2; exit 2 ;; esac
case "$CUDA" in ON|OFF) ;; *) echo "CUDA must be ON or OFF" >&2; exit 2 ;; esac
args=(-S "$ROOT" -B "$BUILD_DIR" -DCMAKE_Fortran_COMPILER="$FC"
      -DCMAKE_BUILD_TYPE=Release -DASTR_WITH_CUDA="$CUDA"
      -DASTR_WITH_AIR5_CHEMISTRY="$AIR5" -DCHEMISTRY=OFF -DBUILD_TESTING=OFF)
if [[ -n "${HDF5_ROOT:-}" ]]; then args+=(-DHDF5_ROOT="$HDF5_ROOT"); fi
cmake "${args[@]}"
cmake --build "$BUILD_DIR" --target astr -j "$JOBS"
echo "Executable: $BUILD_DIR/bin/astr"
