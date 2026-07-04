#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Taylor_Green_Vortex"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
WALL_AXIS="${WALL_AXIS:-${ZERO_AXIS:-x}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/wall41_phased_compare/${WALL_AXIS}}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
LFILTER="${LFILTER:-f}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-64,64,64}"
DELTAT="${DELTAT:-5.d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
WALL_TEMP="${WALL_TEMP:-273.15d0}"
STATS_ATOL="${STATS_ATOL:-1e-9}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
COMPARE_STATS="${COMPARE_STATS:-t}"
COMPARE_FIELD="${COMPARE_FIELD:-t}"

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

for target in cpu gpu; do
  use_gpu=f
  if [[ "$target" == "gpu" ]]; then
    use_gpu=t
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$OUT_DIR/$target" \
    --input-name input.tgv \
    --homogeneous "$HOMOGENEOUS" \
    --bctype "$BCTYPE" \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" \
    --lfilter "$LFILTER" \
    --diffterm "$DIFFTERM" \
    --scheme "$SCHEME" \
    --grid "$GRID" \
    --deltat "$DELTAT"
done

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
