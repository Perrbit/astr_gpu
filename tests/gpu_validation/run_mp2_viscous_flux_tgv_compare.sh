#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

MP2_CANDIDATE=viscous_flux \
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_viscous_flux_tgv_$STAMP}" \
FIELD_ATOL="${FIELD_ATOL:-1e-9}" FIELD_RTOL="${FIELD_RTOL:-0}" \
STATS_ATOL="${STATS_ATOL:-1e-10}" STATS_RTOL="${STATS_RTOL:-0}" \
  "$ROOT_DIR/tests/gpu_validation/run_mp2_derivative_tgv_compare.sh"
