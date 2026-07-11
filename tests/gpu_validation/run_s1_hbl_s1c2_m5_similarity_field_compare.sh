#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INITIALIZATION_MODE=field \
  exec "$ROOT_DIR/tests/gpu_validation/run_s1_hbl_s1c1_m5_sutherland_compare.sh"
