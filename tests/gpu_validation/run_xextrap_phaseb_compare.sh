#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/xextrap_phaseb_compare}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-99}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-64,64,64}"
DELTAT="${DELTAT:-5.d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-9}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-9}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
COMPARE_STATS="${COMPARE_STATS:-t}"
COMPARE_FIELD="${COMPARE_FIELD:-f}"
ZERO_AXIS="${ZERO_AXIS:-x}"
BC_KIND="${BC_KIND:-zeroextrap}"
WALL_TEMP="${WALL_TEMP:-273.15d0}"
XSLIP="${XSLIP:-3.141592653589793d0}"

case "$BC_KIND" in
  zeroextrap)
    case "$ZERO_AXIS" in
      x)
        HOMOGENEOUS="f,t,t"
        BCTYPE="50,50,1,1,1,1"
        ;;
      y)
        HOMOGENEOUS="t,f,t"
        BCTYPE="1,1,50,50,1,1"
        ;;
      z)
        HOMOGENEOUS="t,t,f"
        BCTYPE="1,1,1,1,50,50"
        ;;
      *)
        printf 'unsupported ZERO_AXIS=%s; expected x, y, or z\n' "$ZERO_AXIS" >&2
        exit 2
        ;;
    esac
    ;;
  symmetry)
    case "$ZERO_AXIS" in
      x)
        HOMOGENEOUS="f,t,t"
        BCTYPE="60,60,1,1,1,1"
        ;;
      y)
        HOMOGENEOUS="t,f,t"
        BCTYPE="1,1,60,60,1,1"
        ;;
      z)
        HOMOGENEOUS="t,t,f"
        BCTYPE="1,1,1,1,60,60"
        ;;
      *)
        printf 'unsupported ZERO_AXIS=%s; expected x, y, or z\n' "$ZERO_AXIS" >&2
        exit 2
        ;;
    esac
    ;;
  adiabaticwall)
    case "$ZERO_AXIS" in
      x)
        HOMOGENEOUS="f,t,t"
        BCTYPE="42,42,1,1,1,1"
        ;;
      y)
        HOMOGENEOUS="t,f,t"
        BCTYPE="1,1,42,42,1,1"
        ;;
      *)
        printf 'unsupported ZERO_AXIS=%s for adiabaticwall; expected x or y because CPU noslip_adibatic implements ndir=1..4 only\n' "$ZERO_AXIS" >&2
        exit 2
        ;;
    esac
    ;;
  slipisotwall)
    case "$ZERO_AXIS" in
      y)
        HOMOGENEOUS="t,f,t"
        BCTYPE="1;1;411, ${XSLIP}, ${WALL_TEMP};411, ${XSLIP}, ${WALL_TEMP};1;1"
        ;;
      *)
        printf 'unsupported ZERO_AXIS=%s for slipisotwall; expected y because CPU slipisotwall implements ndir=3/4 only\n' "$ZERO_AXIS" >&2
        exit 2
        ;;
    esac
    ;;
  slipadibwall)
    case "$ZERO_AXIS" in
      y)
        HOMOGENEOUS="t,f,t"
        BCTYPE="1;1;421, ${XSLIP};421, ${XSLIP};1;1"
        ;;
      *)
        printf 'unsupported ZERO_AXIS=%s for slipadibwall; expected y because CPU slipadibwall implements ndir=3/4 only\n' "$ZERO_AXIS" >&2
        exit 2
        ;;
    esac
    ;;
  *)
    printf 'unsupported BC_KIND=%s; expected zeroextrap, symmetry, adiabaticwall, slipisotwall, or slipadibwall\n' "$BC_KIND" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" \
  --dst-case "$OUT_DIR/cpu" \
  --input-name input.tgv \
  --homogeneous "$HOMOGENEOUS" \
  --bctype "$BCTYPE" \
  --use-gpu f \
  --maxstep "$MAXSTEP" \
  --feqchkpt "$FEQCHKPT" \
  --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" \
  --scheme "$SCHEME" \
  --grid "$GRID" \
  --deltat "$DELTAT"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$CASE_DIR" \
  --dst-case "$OUT_DIR/gpu" \
  --input-name input.tgv \
  --homogeneous "$HOMOGENEOUS" \
  --bctype "$BCTYPE" \
  --use-gpu t \
  --maxstep "$MAXSTEP" \
  --feqchkpt "$FEQCHKPT" \
  --lfilter "$LFILTER" \
  --diffterm "$DIFFTERM" \
  --scheme "$SCHEME" \
  --grid "$GRID" \
  --deltat "$DELTAT"

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv > gpu.log 2>&1
)

if [[ "$COMPARE_STATS" == "t" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
    --cpu "$OUT_DIR/cpu" \
    --gpu "$OUT_DIR/gpu" \
    --report "$OUT_DIR/flowstate_compare.txt" \
    --atol "$STATS_ATOL" \
    --rtol "$STATS_RTOL"
fi

if [[ "$COMPARE_FIELD" == "t" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$OUT_DIR/cpu" \
    --gpu "$OUT_DIR/gpu" \
    --report "$OUT_DIR/flowfield_compare.txt" \
    --atol "$FIELD_ATOL" \
    --rtol "$FIELD_RTOL"
fi
