#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
WALL_AXIS="${WALL_AXIS:-${ZERO_AXIS:-x}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/wall41_cpu_probe/${WALL_AXIS}}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-99}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-64,64,64}"
DELTAT="${DELTAT:-5.d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
WALL_TEMP="${WALL_TEMP:-273.15d0}"

case "$WALL_AXIS" in
  x)
    HOMOGENEOUS="f,t,t"
    BCTYPE="41, ${WALL_TEMP};41, ${WALL_TEMP};1;1;1;1"
    ;;
  y)
    HOMOGENEOUS="t,f,t"
    BCTYPE="1;1;41, ${WALL_TEMP};41, ${WALL_TEMP};1;1"
    ;;
  z)
    HOMOGENEOUS="t,t,f"
    BCTYPE="1;1;1;1;41, ${WALL_TEMP};41, ${WALL_TEMP}"
    ;;
  *)
    printf 'unsupported WALL_AXIS=%s; expected x, y, or z\n' "$WALL_AXIS" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

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

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.tgv > cpu.log 2>&1
)

LOG_FILE="$OUT_DIR/cpu/cpu.log"
if rg -a -q 'COMPUTATION CRASHED|Warning: ieee_(invalid|divide_by_zero|overflow)|ERROR STOP|error @|Segmentation fault|forrtl:' "$LOG_FILE"; then
  printf 'wall41 CPU probe failed for axis=%s; see %s\n' "$WALL_AXIS" "$LOG_FILE" >&2
  tail -80 "$LOG_FILE" >&2
  exit 1
fi

FLOWSTATE="$OUT_DIR/cpu/flowstate.dat"
if [[ ! -s "$FLOWSTATE" ]]; then
  printf 'wall41 CPU probe produced no flowstate for axis=%s; see %s\n' "$WALL_AXIS" "$LOG_FILE" >&2
  exit 1
fi

printf 'wall41 CPU probe completed for axis=%s\n' "$WALL_AXIS"
printf 'case: %s\n' "$OUT_DIR/cpu"
tail -5 "$FLOWSTATE"
