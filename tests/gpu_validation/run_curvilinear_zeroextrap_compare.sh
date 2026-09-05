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

OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_zeroextrap_np${NP}}"
ZERO_AXIS="${ZERO_AXIS:-x}"

case "$ZERO_AXIS" in
  x)
    MAPPING="x-wavy"
    HOMOGENEOUS="f,t,t"
    BCTYPE="50;50;1;1;1;1"
    HDF_AXIS=2
    ;;
  y)
    MAPPING="y-wavy"
    HOMOGENEOUS="t,f,t"
    BCTYPE="1;1;50;50;1;1"
    HDF_AXIS=1
    ;;
  z)
    MAPPING="z-wavy"
    HOMOGENEOUS="t,t,f"
    BCTYPE="1;1;1;1;50;50"
    HDF_AXIS=0
    ;;
  *)
    printf 'ZERO_AXIS must be x, y, or z; got %s\n' "$ZERO_AXIS" >&2
    exit 2
    ;;
esac

OUT_DIR="$OUT_DIR" \
MAPPING="$MAPPING" \
HOMOGENEOUS="$HOMOGENEOUS" \
BCTYPE="$BCTYPE" \
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
  if [[ "$mode" == "cpu" ]]; then
    invariant_input="$OUT_DIR/cpu/outdat/rk_complete_snapshot.h5"
  else
    invariant_input="$OUT_DIR/gpu"
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/check_zeroextrap_invariants.py" \
    --input "$invariant_input" \
    --report "$OUT_DIR/${mode}_zeroextrap_invariants.txt" \
    --axis "$HDF_AXIS" \
    --gamma "${GAMMA:-1.4}" \
    --mach "${MACH:-0.1}" \
    --atol "${BOUNDARY_ATOL:-1e-10}"
done
