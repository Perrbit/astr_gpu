#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_viscous_uniform_np1}"
DIFFTERM=t
SCHEME=643e

export OUT_DIR DIFFTERM SCHEME
export ASTR_NSCBC_FARFIELD_MODE=nonreflecting
exec "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_uniform_compare.sh"
