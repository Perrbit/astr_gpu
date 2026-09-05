#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
BC_KIND="${BC_KIND:-411}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_wall_c19_${BC_KIND}_np${NP}}"
WALL_TEMPERATURE="${WALL_TEMPERATURE:-273.15}"
XSLIP="${XSLIP:-3.141592653589793}"
WALL_BLOWING="${WALL_BLOWING:-f}"
WALL_AMPLITUDE="${WALL_AMPLITUDE:-0.01}"
WALL_BETA="${WALL_BETA:-1.0}"
WALL_XA="${WALL_XA:-0.5}"
WALL_XB="${WALL_XB:-3.0}"
WALL_XC="${WALL_XC:-5.8}"
WALL_NMOD_T="${WALL_NMOD_T:-0}"
WALL_NMOD_Z="${WALL_NMOD_Z:-2}"
CPU_GEOMETRY_DUMP=""

case "$BC_KIND" in
  41) BCTYPE="1;1;41,${WALL_TEMPERATURE}d0;41,${WALL_TEMPERATURE}d0;1;1" ;;
  42) BCTYPE="1;1;42;42;1;1" ;;
  411) BCTYPE="1;1;411,${XSLIP}d0,${WALL_TEMPERATURE}d0;411,${XSLIP}d0,${WALL_TEMPERATURE}d0;1;1" ;;
  421) BCTYPE="1;1;421,${XSLIP}d0;421,${XSLIP}d0;1;1" ;;
  *) printf 'BC_KIND must be 41, 42, 411, or 421\n' >&2; exit 2 ;;
esac
if [[ "$WALL_BLOWING" != "t" && "$WALL_BLOWING" != "f" ]]; then
  printf 'WALL_BLOWING must be t or f\n' >&2
  exit 2
fi
if [[ "$NP" == "1" ]]; then
  CPU_GEOMETRY_DUMP="$OUT_DIR/cpu_geometry.h5"
fi

OUT_DIR="$OUT_DIR" \
CPU_GEOMETRY_DUMP="$CPU_GEOMETRY_DUMP" \
MAPPING=y-wavy \
HOMOGENEOUS=t,f,t \
BCTYPE="$BCTYPE" \
WALL_BLOWING="$WALL_BLOWING" \
WALL_AMPLITUDE="$WALL_AMPLITUDE" \
WALL_BETA="$WALL_BETA" \
WALL_XA="$WALL_XA" \
WALL_XB="$WALL_XB" \
WALL_XC="$WALL_XC" \
WALL_NMOD_T="$WALL_NMOD_T" \
WALL_NMOD_Z="$WALL_NMOD_Z" \
MAXSTEP="${MAXSTEP:-1}" \
FEQCHKPT="${FEQCHKPT:-1}" \
LFILTER="${LFILTER:-f}" \
DIFFTERM="${DIFFTERM:-f}" \
GRID="${GRID:-32,32,32}" \
DELTAT="${DELTAT:-5.d-4}" \
NP="$NP" \
TOPOLOGY="$TOPOLOGY" \
FIELD_ATOL="${FIELD_ATOL:-1e-10}" \
FIELD_RTOL="${FIELD_RTOL:-1e-10}" \
  "$ROOT_DIR/tests/gpu_validation/run_curvilinear_tgv_compare.sh"

if [[ "$NP" == "1" ]]; then
  amplitude=0.0
  if [[ "$WALL_BLOWING" == "t" ]]; then amplitude="$WALL_AMPLITUDE"; fi
  for mode in cpu gpu; do
    input="$OUT_DIR/$mode"
    if [[ "$mode" == "cpu" ]]; then input="$OUT_DIR/cpu/outdat/rk_complete_snapshot.h5"; fi
    python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_wall_c19_invariants.py" \
      --input "$input" \
      --geometry "$CPU_GEOMETRY_DUMP" \
      --grid "$OUT_DIR/$mode/datin/grid.curvilinear.h5" \
      --kind "$BC_KIND" \
      --xslip "$XSLIP" \
      --wall-amplitude "$amplitude" \
      --xa "$WALL_XA" --xb "$WALL_XB" --xc "$WALL_XC" \
      --nmod-z "$WALL_NMOD_Z" \
      --atol "${WALL_ATOL:-1e-12}" \
      --report "$OUT_DIR/${mode}_c19_invariants.txt"
  done
fi
