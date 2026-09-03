#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_hbl_c8_filter}"

LFILTER=t DIFFTERM="${DIFFTERM:-t}" OUT_DIR="$OUT_DIR" \
  exec "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c7_compare.sh"
