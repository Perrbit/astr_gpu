#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_selective_roe_s2c4}"
export XMIN_BCTYPE=12
export UPPER_BCTYPE=52
export DIFFTERM=f
export LCHARDECOMP=t
export SPONGE_IM="${SPONGE_IM:-0}"
export COMPARE_SENSOR="${COMPARE_SENSOR:-f}"
export MAXSTEP="${MAXSTEP:-1}"
export FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"

exec "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh"
