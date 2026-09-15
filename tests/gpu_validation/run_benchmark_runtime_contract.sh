#!/usr/bin/env bash
set -euo pipefail

GPU_PROBE="${1:?usage: run_benchmark_runtime_contract.sh /absolute/path/to/gpu-probe /absolute/path/to/cpu-probe}"
CPU_PROBE="${2:?usage: run_benchmark_runtime_contract.sh /absolute/path/to/gpu-probe /absolute/path/to/cpu-probe}"

# Local HPC-X/Open MPI may discover HCOLL without a usable HCA. Keep the
# contract output deterministic without changing production MPI behavior.
export OMPI_MCA_coll_hcoll_enable=0

expect_fail() {
  set +e
  "$@" > /tmp/astr_p4_benchmark_expected_failure.log 2>&1
  local status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "expected command to fail: $*" >&2
    exit 1
  fi
}

env -u ASTR_GPU_BENCHMARK_NO_FIELD_IO -u ASTR_GPU_RK_TIMING \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic disabled
ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 2 "$GPU_PROBE" gpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" cpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$CPU_PROBE" gpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=0 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu channel periodic periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv physical physical enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic physical enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=invalid ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic disabled
expect_fail env -u ASTR_GPU_BENCHMARK_NO_FIELD_IO ASTR_GPU_RK_TIMING=invalid \
  mpirun -np 1 "$GPU_PROBE" gpu tgv periodic periodic disabled
expect_fail mpirun -np 2 bash -c '
  if [[ "$OMPI_COMM_WORLD_RANK" == 0 ]]; then
    export ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1
  else
    unset ASTR_GPU_BENCHMARK_NO_FIELD_IO ASTR_GPU_RK_TIMING
  fi
  exec "$1" gpu tgv periodic periodic enabled
' bash "$GPU_PROBE"
expect_fail mpirun -np 2 bash -c '
  unset ASTR_GPU_BENCHMARK_NO_FIELD_IO
  if [[ "$OMPI_COMM_WORLD_RANK" == 0 ]]; then
    export ASTR_GPU_RK_TIMING=1
  else
    export ASTR_GPU_RK_TIMING=0
  fi
  exec "$1" gpu tgv periodic periodic disabled
' bash "$GPU_PROBE"
echo "benchmark runtime contract passed"
