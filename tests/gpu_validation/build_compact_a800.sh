#!/usr/bin/env bash
set -eo pipefail
export DEPLOY_DIR=${1:?supply the private campaign directory}
cd "$DEPLOY_DIR"
source "$DEPLOY_DIR/src/tests/gpu_validation/compact_a800_platform_env.sh"
mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/build"
common=(
  -DCMAKE_BUILD_TYPE=Release -DASTR_NUMERICAL_PRECISION=fp64
  -DCMAKE_Fortran_COMPILER="$(command -v nvfortran)"
  -DCMAKE_C_COMPILER="$(command -v gcc)"
  -DMPI_Fortran_COMPILER="$MPI_ROOT/bin/mpif90" -DMPI_C_COMPILER="$MPI_ROOT/bin/mpicc"
  -DHDF5_ROOT="$HDF5_ROOT" -DHDF5_Fortran_COMPILER_EXECUTABLE="$HDF5_ROOT/bin/h5hlfc"
  -DASTR_WITH_CATALYST=OFF -DASTR_WITH_INSITU_DEVICE=OFF -DASTR_WITH_INSITU_DEVICE_RENDERING=OFF
  -DASTR_WITH_AIR5_CHEMISTRY=OFF -DCHEMISTRY=OFF -DBUILD_TESTING=ON
  -DASTR_USERDEFINE_SOURCE="$DEPLOY_DIR/src/user_define_module/userdefine.F90"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
cmake -S "$DEPLOY_DIR/src" -B "$DEPLOY_DIR/build/cpu" "${common[@]}" -DASTR_WITH_CUDA=OFF
cmake --build "$DEPLOY_DIR/build/cpu" --target astr --parallel 8
cmake -S "$DEPLOY_DIR/src" -B "$DEPLOY_DIR/build/gpu" "${common[@]}" \
  -DASTR_WITH_CUDA=ON -DASTR_CUDA_ARCHITECTURES=80
cmake --build "$DEPLOY_DIR/build/gpu" --target astr --parallel 8
for backend in cpu gpu; do
  ldd -r "$DEPLOY_DIR/build/$backend/bin/astr" > "$DEPLOY_DIR/logs/ldd-$backend.txt" 2>&1
  if grep -E 'not found|undefined symbol' "$DEPLOY_DIR/logs/ldd-$backend.txt"; then
    printf 'Unresolved dependency for %s\n' "$backend" >&2
    exit 1
  fi
done
"$PYTHON_EXE" -c "import sys; sys.path.insert(0,'$DEPLOY_DIR/src/tests/gpu_validation'); import run_compact_a800_campaign; print('Actual campaign imports: PASS')"
"$PYTHON_EXE" "$DEPLOY_DIR/src/tests/gpu_validation/run_compact_a800_campaign.py" \
  --mode manifest --backend cpu --out "$DEPLOY_DIR/results/unused" > "$DEPLOY_DIR/logs/cpu-matrix.json"
"$PYTHON_EXE" "$DEPLOY_DIR/src/tests/gpu_validation/run_compact_a800_campaign.py" \
  --mode manifest --backend gpu --out "$DEPLOY_DIR/results/unused" > "$DEPLOY_DIR/logs/gpu-matrix.json"
printf 'LOGIN_BUILD_AND_DEPENDENCIES=PASS\n'
