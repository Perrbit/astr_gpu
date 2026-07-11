#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_hbl_inlet_sustained_shock}"
export PROFILE_OBLIQUE_SHOCK="${PROFILE_OBLIQUE_SHOCK:-t}"
export PROFILE_PRESSURE_MODE="${PROFILE_PRESSURE_MODE:-provided}"
export SHOCK_X0="${SHOCK_X0:--1.0}"
export SHOCK_Y0="${SHOCK_Y0:-0.18}"
export PROFILE_SHOCK_Y_MIN="${PROFILE_SHOCK_Y_MIN:-$SHOCK_Y0}"

exec "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh"
