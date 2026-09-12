#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_viscous_acoustic_np1}"
DIFFTERM=t
SCHEME=643e
FIELD_ATOL="${FIELD_ATOL:-1e-10}"

# The first viscous gate measures CPU/GPU agreement and refinement trend. It
# does not inherit CURVE-C22's inviscid absolute-reflection threshold.
REFLECTION_MAX=1e30
CONTROL_RATIO_MAX=1e30
RUN_CONTROL=f

export OUT_DIR DIFFTERM SCHEME FIELD_ATOL REFLECTION_MAX CONTROL_RATIO_MAX RUN_CONTROL
export ASTR_NSCBC_FARFIELD_MODE=nonreflecting
exec "$ROOT_DIR/tests/gpu_validation/run_curvilinear_nscbc52_acoustic_compare.sh"
