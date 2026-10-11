#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
  printf 'Usage: bash run_platform_insitu.sh DEPLOYMENT_ENVIRONMENT\n' >&2
  exit 2
fi
source "$1"
: "${DEPLOY_ROOT:?Set the deployment directory}"
: "${ASTR_PLATFORM_WORK_ROOT:?Set the permitted remote work root}"
: "${INSITU_DEPENDENCY_ROOT:?Set the existing validated device dependency directory}"
ROOT=$(realpath "$DEPLOY_ROOT")
WORK_ROOT=$(realpath "$ASTR_PLATFORM_WORK_ROOT")
case "$ROOT/" in "$WORK_ROOT"/*) ;; *) exit 2 ;; esac
cd "$ROOT"
PV="$INSITU_DEPENDENCY_ROOT/build/paraview-device"
OUT="$ROOT/runs-${SLURM_JOB_ID:?Run inside an allocated batch job}"
mkdir "$OUT"
{
  date -Iseconds
  hostname
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-not-set}"
  nvidia-smi
  nvidia-smi topo -m
  "$MPI_ROOT/bin/mpiexec" --version
  for binary in "$ROOT/build/astr-debug/bin/astr" "$ROOT/build/astr-native/bin/astr" \
                "$ROOT/build/astr-debug/bin/insitu_standard_device_probe" \
                "$PV/lib/catalyst/libcatalyst-paraview.so"; do
    ldd -r "$binary"
  done
} > "$OUT/environment.txt" 2>&1
if grep -E 'not found|undefined symbol' "$OUT/environment.txt"; then
  printf 'Compute-node runtime dependencies are incomplete\n' >&2
  exit 2
fi
sha256sum --check "$ROOT/logs/solver-binaries.sha256"
python3 "$ROOT/src/astr/tests/gpu_validation/run_checkpoint_publication_preflight.py" \
  --probe "$ROOT/build/astr-debug/bin/checkpoint_bundle_probe" --mpiexec "$MPI_ROOT/bin/mpiexec" \
  --directory "$OUT"
python3 -c 'import numpy,h5py,PIL,psutil,pytest,pynvml; print("PYTHON_RUNTIME_IMPORTS=PASS")'
read -r -a topologies <<< "${M12_TOPOLOGIES:-2,1,1 4,1,1}"
python3 -u "$ROOT/src/astr/tests/gpu_validation/run_platform_insitu_acceptance.py" \
  --source "$ROOT/inputs/M12" --output "$OUT/acceptance" \
  --executable "$ROOT/build/astr-debug/bin/astr" --native-executable "$ROOT/build/astr-native/bin/astr" \
  --mpiexec "$MPI_ROOT/bin/mpiexec" --library "$PV/lib/catalyst" \
  --probe-directory "$ROOT/build/astr-debug/bin" --m12-topologies "${topologies[@]}" \
  --timeout-seconds "${CASE_TIMEOUT_SECONDS:-3600}" --probe-timeout-seconds "${PROBE_TIMEOUT_SECONDS:-600}" \
  --host-budget-gib "${POST_HOST_BUDGET_GIB:-32}" --device-budget-gib "${POST_DEVICE_BUDGET_GIB:-8}" \
  --device-reserve-gib "${DEVICE_RESERVE_GIB:-2}" --directory-budget-gib "${CASE_DIRECTORY_BUDGET_GIB:-32}"
printf 'ASTR_PLATFORM_INSITU_SHORT_GATE=PASS\n'
