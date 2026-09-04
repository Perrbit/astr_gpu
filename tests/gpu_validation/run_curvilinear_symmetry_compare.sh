#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NP="${NP:-1}"
AXIS="${AXIS:-x}"
case "$AXIS" in
  x)
    MAPPING=x-wavy
    HOMOGENEOUS=f,t,t
    BCTYPE='60;60;1;1;1;1'
    NP2_TOPOLOGY=2,1,1
    ;;
  y)
    MAPPING=y-wavy
    HOMOGENEOUS=t,f,t
    BCTYPE='1;1;60;60;1;1'
    NP2_TOPOLOGY=1,2,1
    ;;
  z)
    MAPPING=z-wavy
    HOMOGENEOUS=t,t,f
    BCTYPE='1;1;1;1;60;60'
    NP2_TOPOLOGY=1,1,2
    ;;
  *)
    printf 'AXIS must be x, y, or z\n' >&2
    exit 2
    ;;
esac
if [[ -n "${TOPOLOGY:-}" ]]; then
  TOPOLOGY="$TOPOLOGY"
elif [[ "$NP" == "1" ]]; then
  TOPOLOGY="1,1,1"
elif [[ "$NP" == "2" ]]; then
  TOPOLOGY="$NP2_TOPOLOGY"
else
  printf 'TOPOLOGY is required when NP is neither 1 nor 2\n' >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_symmetry_${AXIS}_np${NP}}"
CPU_GEOMETRY_DUMP=""
if [[ "$NP" == "1" ]]; then
  CPU_GEOMETRY_DUMP="$OUT_DIR/cpu_geometry.h5"
fi

OUT_DIR="$OUT_DIR" \
CPU_GEOMETRY_DUMP="$CPU_GEOMETRY_DUMP" \
MAPPING="$MAPPING" \
HOMOGENEOUS="$HOMOGENEOUS" \
BCTYPE="$BCTYPE" \
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

for mode in cpu gpu; do
  invariant_args=(
    --input "$OUT_DIR/$mode"
    --axis "$AXIS"
    --amplitude "${AMPLITUDE:-0.15}"
    --analytic-atol "${ANALYTIC_NORMAL_ATOL:-1e-6}"
    --projection-atol "${PROJECTION_ATOL:-1e-12}"
    --report "$OUT_DIR/${mode}_symmetry_invariants.txt"
  )
  if [[ "$mode" == "cpu" ]]; then
    invariant_args[1]="$OUT_DIR/cpu/outdat/rk_complete_snapshot.h5"
  fi
  if [[ -n "$CPU_GEOMETRY_DUMP" ]]; then
    invariant_args+=(--geometry "$CPU_GEOMETRY_DUMP")
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_symmetry_invariants.py" \
    "${invariant_args[@]}"
done
