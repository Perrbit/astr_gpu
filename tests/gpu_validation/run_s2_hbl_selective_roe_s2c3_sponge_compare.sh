#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_selective_roe_s2c3_sponge}"
export SPONGE_IM="${SPONGE_IM:-16}"

exec "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh"
