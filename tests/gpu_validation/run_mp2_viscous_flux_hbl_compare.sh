#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"

MP2_CANDIDATE=viscous_flux \
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_viscous_flux_hbl_$STAMP}" \
FIELD_ATOL="${FIELD_ATOL:-1e-8}" FIELD_RTOL="${FIELD_RTOL:-0}" \
STATS_ATOL="${STATS_ATOL:-1e-8}" STATS_RTOL="${STATS_RTOL:-0}" \
DIAGNOSTIC_ATOL="${DIAGNOSTIC_ATOL:-1e-8}" DIAGNOSTIC_RTOL="${DIAGNOSTIC_RTOL:-0}" \
  "$ROOT_DIR/tests/gpu_validation/run_mp2_derivative_hbl_compare.sh"
