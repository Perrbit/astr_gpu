#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_viscous_hbl_np1}"
if [[ "$OUT_DIR" != /* ]]; then
  OUT_DIR="$ROOT_DIR/$OUT_DIR"
fi
UPPER_BCTYPE=52
DIFFTERM=t
LFILTER=f
CONSCHM=643e

export OUT_DIR UPPER_BCTYPE DIFFTERM LFILTER CONSCHM
export ASTR_NSCBC_FARFIELD_MODE=nonreflecting
exec "$ROOT_DIR/tests/gpu_validation/run_curvilinear_hbl_c7_compare.sh"
