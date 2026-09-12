#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/mp2_viscous_flux_mpi_$STAMP}"

[[ ! -e "$OUT_DIR" ]] || { echo "refusing to overwrite MPI evidence directory: $OUT_DIR" >&2; exit 2; }
mkdir -p "$OUT_DIR"

NP=2 TOPOLOGY=2,1,1 OUT_DIR="$OUT_DIR/np2_2x1x1" \
  "$ROOT_DIR/tests/gpu_validation/run_mp2_viscous_flux_tgv_compare.sh"
NP=4 TOPOLOGY=2,2,1 OUT_DIR="$OUT_DIR/np4_2x2x1" \
  "$ROOT_DIR/tests/gpu_validation/run_mp2_viscous_flux_tgv_compare.sh"

printf 'case\tstatus\nnp2_2x1x1\tPASS\nnp4_2x2x1\tPASS\n' > "$OUT_DIR/status.tsv"
echo "MP2 viscous_flux MPI matrix passed: $OUT_DIR"
