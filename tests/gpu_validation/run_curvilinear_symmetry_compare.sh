#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NP="${NP:-1}"
if [[ -n "${TOPOLOGY:-}" ]]; then
  TOPOLOGY="$TOPOLOGY"
elif [[ "$NP" == "1" ]]; then
  TOPOLOGY="1,1,1"
elif [[ "$NP" == "2" ]]; then
  TOPOLOGY="2,1,1"
else
  printf 'TOPOLOGY is required when NP is neither 1 nor 2\n' >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_symmetry_np${NP}}"

OUT_DIR="$OUT_DIR" \
MAPPING=x-wavy \
HOMOGENEOUS=f,t,t \
BCTYPE='60;60;1;1;1;1' \
MAXSTEP="${MAXSTEP:-1}" \
FEQCHKPT="${FEQCHKPT:-1}" \
LFILTER=f \
DIFFTERM=f \
GRID="${GRID:-32,32,32}" \
NP="$NP" \
TOPOLOGY="$TOPOLOGY" \
FIELD_ATOL="${FIELD_ATOL:-1e-10}" \
FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"
