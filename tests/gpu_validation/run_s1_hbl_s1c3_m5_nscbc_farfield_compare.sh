#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s1_hbl_s1c3_m5_nscbc_farfield}"

UPPER_BCTYPE=52 OUT_DIR="$OUT_DIR" \
  exec "$ROOT_DIR/tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh"
