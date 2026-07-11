#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/sod_phase_s0a3_accuracy}"
export PHASE_LABEL=S0-A3
export RECON_SCHEM=3

exec "$ROOT_DIR/tests/gpu_validation/run_sod_phase_s0a2_accuracy.sh"
