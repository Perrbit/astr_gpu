#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

MP2_CANDIDATE=viscous_flux \
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_viscous_flux_benchmark_$STAMP}" \
  "$ROOT_DIR/tests/gpu_validation/run_mp2_derivative_benchmark.sh"
