#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_selective_roe_s2c3}"
export PROFILE_OBLIQUE_SHOCK="${PROFILE_OBLIQUE_SHOCK:-t}"
export PROFILE_PRESSURE_MODE="${PROFILE_PRESSURE_MODE:-provided}"
export UPPER_BCTYPE="${UPPER_BCTYPE:-51}"
export DIFFTERM="${DIFFTERM:-f}"
export LCHARDECOMP="${LCHARDECOMP:-t}"
export COMPARE_SENSOR="${COMPARE_SENSOR:-f}"
export SPONGE_IM="${SPONGE_IM:-0}"
export SHOCK_X0="${SHOCK_X0:--1.0}"
export SHOCK_Y0="${SHOCK_Y0:-0.18}"
export PROFILE_SHOCK_Y_MIN="${PROFILE_SHOCK_Y_MIN:-0.18}"
export MAXSTEP="${MAXSTEP:-1}"
export FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"

exec "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh"
