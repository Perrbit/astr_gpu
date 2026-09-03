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
  printf 'TOPOLOGY is required when NP is greater than 2\n' >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_wall41_np${NP}}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-273.15}"

OUT_DIR="$OUT_DIR" \
MAPPING=x-wavy \
HOMOGENEOUS=f,t,t \
BCTYPE="41,${WALL_TEMPERATURE}d0;41,${WALL_TEMPERATURE}d0;1;1;1;1" \
MAXSTEP="${MAXSTEP:-5}" \
FEQCHKPT="${FEQCHKPT:-${MAXSTEP:-5}}" \
LFILTER="${LFILTER:-t}" \
DIFFTERM="${DIFFTERM:-t}" \
GRID="${GRID:-32,32,32}" \
NP="$NP" \
TOPOLOGY="$TOPOLOGY" \
FIELD_ATOL="${FIELD_ATOL:-1e-10}" \
FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"

for mode in cpu gpu; do
  python3 "$ROOT_DIR/tests/gpu_validation/check_wall41_invariants.py" \
    --input "$OUT_DIR/$mode" \
    --report "$OUT_DIR/${mode}_wall41_invariants.txt" \
    --axis 2 \
    --wall-temperature "$WALL_TEMPERATURE" \
    --mach 0.1 \
    --atol "${WALL_ATOL:-1e-12}"
done
