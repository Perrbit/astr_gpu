#!/usr/bin/env bash
set -euo pipefail

PROBE="${1:?usage: run_benchmark_runtime_contract.sh /absolute/path/to/probe}"

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
  mpirun -np 1 "$PROBE" gpu tgv periodic disabled
ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 2 "$PROBE" gpu tgv periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$PROBE" cpu tgv periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=0 \
  mpirun -np 1 "$PROBE" gpu tgv periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$PROBE" gpu channel periodic enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$PROBE" gpu tgv physical enabled
expect_fail env ASTR_GPU_BENCHMARK_NO_FIELD_IO=invalid ASTR_GPU_RK_TIMING=1 \
  mpirun -np 1 "$PROBE" gpu tgv periodic disabled
expect_fail mpirun -np 2 bash -c '
  if [[ "$OMPI_COMM_WORLD_RANK" == 0 ]]; then
    export ASTR_GPU_BENCHMARK_NO_FIELD_IO=1 ASTR_GPU_RK_TIMING=1
  else
    unset ASTR_GPU_BENCHMARK_NO_FIELD_IO ASTR_GPU_RK_TIMING
  fi
  exec "$1" gpu tgv periodic enabled
' bash "$PROBE"
echo "benchmark runtime contract passed"
