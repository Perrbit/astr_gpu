#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_nscbc52_incoming_only}"
export XMIN_BCTYPE=12
export UPPER_BCTYPE=52
export DIFFTERM=f
export LCHARDECOMP=t
export SPONGE_IM=0
export COMPARE_SENSOR="${COMPARE_SENSOR:-f}"
export MAXSTEP="${MAXSTEP:-1}"
export FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"

export ASTR_NSCBC_FARFIELD_MODE=incoming_only
export ASTR_NSCBC_FARFIELD_RHO="${ASTR_NSCBC_FARFIELD_RHO:-0.9}"
export ASTR_NSCBC_FARFIELD_U="${ASTR_NSCBC_FARFIELD_U:-0.8}"
export ASTR_NSCBC_FARFIELD_V="${ASTR_NSCBC_FARFIELD_V:--0.05}"
export ASTR_NSCBC_FARFIELD_W="${ASTR_NSCBC_FARFIELD_W:-0.02}"
export ASTR_NSCBC_FARFIELD_T="${ASTR_NSCBC_FARFIELD_T:-1.2}"

exec "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh"
