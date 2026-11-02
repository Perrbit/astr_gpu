#!/bin/bash -l
set -eo pipefail
backend=${1:?supply cpu or gpu}
export DEPLOY_DIR=${2:?supply the private deployment directory}
cd "$DEPLOY_DIR"
source "$DEPLOY_DIR/src/tests/gpu_validation/compact_a800_platform_env.sh"
case "$backend" in cpu|gpu) ;; *) printf 'Expected cpu or gpu\n' >&2; exit 2 ;; esac
tag="${backend}_${SLURM_JOB_ID}"
mkdir -p "$DEPLOY_DIR/logs" "$DEPLOY_DIR/results"
{
  date -Iseconds
  hostname
  lscpu
  module -t list
  mpirun --version
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  if [[ "$backend" == gpu ]]; then nvidia-smi; nvidia-smi topo -m; fi
  ldd "$DEPLOY_DIR/build/$backend/bin/astr"
} > "$DEPLOY_DIR/logs/environment-$tag.txt" 2>&1
executables=(--cpu "$DEPLOY_DIR/build/cpu/bin/astr")
if [[ "$backend" == gpu ]]; then executables+=(--gpu "$DEPLOY_DIR/build/gpu/bin/astr"); fi
"$PYTHON_EXE" "$DEPLOY_DIR/src/tests/gpu_validation/run_compact_a800_node_preflight.py" \
  --backend "$backend" "${executables[@]}" --out "$DEPLOY_DIR/results/preflight_$tag"
"$PYTHON_EXE" "$DEPLOY_DIR/src/tests/gpu_validation/run_compact_a800_campaign.py" \
  --mode timing --backend "$backend" "${executables[@]}" \
  --halo-transport device-aware --compact-transport device-aware \
  --ucx-tls self,sm,cuda_copy,cuda_ipc --out "$DEPLOY_DIR/results/$tag"
