#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  printf 'Usage: bash run_platform_fullsize.sh DEPLOYMENT_ENVIRONMENT\n' >&2
  exit 2
fi
source "$1"
: "${DEPLOY_ROOT:?Set the deployment directory}"
: "${ASTR_PLATFORM_WORK_ROOT:?Set the permitted remote work root}"
: "${INSITU_DEPENDENCY_ROOT:?Set the validated dependency directory}"
ROOT=$(realpath "$DEPLOY_ROOT")
WORK_ROOT=$(realpath "$ASTR_PLATFORM_WORK_ROOT")
case "$ROOT/" in "$WORK_ROOT"/*) ;; *) exit 2 ;; esac
cd "$ROOT"
PV="$INSITU_DEPENDENCY_ROOT/build/paraview-device"
OUT="$ROOT/runs-${SLURM_JOB_ID:?Run inside an allocated job}"
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
  printf 'Unresolved compute-node runtime dependency\n' >&2
  exit 2
fi
if [[ -n "${ASTR_PLATFORM_RENDERING_LIBRARY:-}" ]]; then
  python3 "$ROOT/src/astr/tests/gpu_validation/check_platform_rendering_runtime.py" \
    --expected "$ASTR_PLATFORM_RENDERING_LIBRARY" \
    "$ROOT/build/astr-debug/bin/astr" "$ROOT/build/astr-native/bin/astr" \
    "$ROOT/build/astr-debug/bin/insitu_standard_device_probe" \
    "$PV/lib/catalyst/libcatalyst-paraview.so"
fi
sha256sum --check "$ROOT/logs/solver-binaries.sha256"
python3 -c 'import numpy,h5py,PIL,psutil,pytest,pynvml; print("PYTHON_RUNTIME_IMPORTS=PASS")'
python3 "$ROOT/src/astr/tests/gpu_validation/run_checkpoint_publication_preflight.py" \
  --probe "$ROOT/build/astr-debug/bin/checkpoint_bundle_probe" --mpiexec "$MPI_ROOT/bin/mpiexec" \
  --directory "$OUT"
python3 -u "$ROOT/src/astr/tests/gpu_validation/run_platform_insitu_fullsize.py" \
  --source "$ROOT/inputs/M12" --output "$OUT/coverage" \
  --executable "$ROOT/build/astr-native/bin/astr" --diagnostic-executable "$ROOT/build/astr-debug/bin/astr" \
  --mpiexec "$MPI_ROOT/bin/mpiexec" --library "$PV/lib/catalyst" \
  --probe-directory "$ROOT/build/astr-debug/bin" \
  --device-budget "${ASTR_PLATFORM_INSITU_DEVICE_BUDGET_BYTES:-34359738368}" \
  --reserve "${ASTR_PLATFORM_INSITU_DEVICE_RESERVE_BYTES:-8589934592}"
printf 'ASTR_PLATFORM_FULLSIZE_RUN=COMPLETED; inspect coverage/report.json for image completeness\n'
