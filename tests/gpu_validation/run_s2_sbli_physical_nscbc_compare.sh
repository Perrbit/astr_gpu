#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/s2_sbli_physical_nscbc}"
export XMIN_BCTYPE=11
export UPPER_BCTYPE=52
export DIFFTERM=t
export LCHARDECOMP=t
export PROFILE_OBLIQUE_SHOCK=f
export FIELD_OBLIQUE_SHOCK=f
export PROFILE_PRESSURE_MODE=reconstruct
export MACH="${MACH:-2.0}"
export REYNOLDS="${REYNOLDS:-296000.0}"
export REFERENCE_TEMPERATURE="${REFERENCE_TEMPERATURE:-148.9}"
export WALL_TEMPERATURE="${WALL_TEMPERATURE:-0.8}"
export DELTAT="${DELTAT:-2.5e-4}"
export SHOCK_ANGLE_DEG="${SHOCK_ANGLE_DEG:-32.6}"
export SHOCK_X0="${SHOCK_X0:-1.0}"
export SHOCK_Y0="${SHOCK_Y0:-1.0}"
export SHOCK_Y_MIN="${SHOCK_Y_MIN:-0.0}"
export SPONGE_IM="${SPONGE_IM:-0}"
export MAXSTEP="${MAXSTEP:-2}"
export FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"

export ASTR_NSCBC_FARFIELD_MODE=sbli_shock
export ASTR_NSCBC_FARFIELD_RHO="${ASTR_NSCBC_FARFIELD_RHO:-1.0}"
export ASTR_NSCBC_FARFIELD_U="${ASTR_NSCBC_FARFIELD_U:-1.0}"
export ASTR_NSCBC_FARFIELD_V="${ASTR_NSCBC_FARFIELD_V:-0.0}"
export ASTR_NSCBC_FARFIELD_W="${ASTR_NSCBC_FARFIELD_W:-0.0}"
export ASTR_NSCBC_FARFIELD_T="${ASTR_NSCBC_FARFIELD_T:-1.0}"
export ASTR_NSCBC_FARFIELD_SHOCK_X="$SHOCK_X0"
export ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG="$SHOCK_ANGLE_DEG"

exec "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_oblique_shock_compare.sh"
