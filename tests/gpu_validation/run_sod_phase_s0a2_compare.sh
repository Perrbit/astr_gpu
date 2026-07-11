#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="${CASE_DIR:-$ROOT_DIR/examples/sod}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/sod_phase_s0a2_compare}"
PHASE_LABEL="${PHASE_LABEL:-S0-A2}"
RECON_SCHEM="${RECON_SCHEM:-1}"
MAXSTEP="${MAXSTEP:-20}"
FEQCHKPT="${FEQCHKPT:-$MAXSTEP}"
GRID="${GRID:-200,8,8}"
DELTAT="${DELTAT:-5.d-4}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
STATS_ATOL="${STATS_ATOL:-1e-10}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
COMPARE_STATS="${COMPARE_STATS:-t}"
COMPARE_FIELD="${COMPARE_FIELD:-t}"
COMPARE_EXACT="${COMPARE_EXACT:-f}"
EXACT_ANALYSIS_HALF_WIDTH="${EXACT_ANALYSIS_HALF_WIDTH:-2.5}"
EXACT_EXCLUDE_CELLS="${EXACT_EXCLUDE_CELLS:-3.0}"
EXACT_MAX_SMOOTH_L1="${EXACT_MAX_SMOOTH_L1:-inf}"
EXACT_MAX_SMOOTH_L2="${EXACT_MAX_SMOOTH_L2:-inf}"
EXACT_MAX_SMOOTH_LINF="${EXACT_MAX_SMOOTH_LINF:-inf}"
EXACT_MAX_BOUND_VIOLATION="${EXACT_MAX_BOUND_VIOLATION:-inf}"
EXACT_MAX_CONTACT_THICKNESS_CELLS="${EXACT_MAX_CONTACT_THICKNESS_CELLS:-inf}"
EXACT_MAX_SHOCK_THICKNESS_CELLS="${EXACT_MAX_SHOCK_THICKNESS_CELLS:-inf}"
EXACT_MAX_POSITION_ERROR_CELLS="${EXACT_MAX_POSITION_ERROR_CELLS:-inf}"

if [[ "$NP" != "1" || "$TOPOLOGY" != "1,1,1" ]]; then
  echo "Phase $PHASE_LABEL is a single-rank gate: require NP=1 and TOPOLOGY=1,1,1" >&2
  exit 2
fi

IFS=',' read -r GRID_I GRID_J GRID_K <<< "$GRID"
if [[ -z "${GRID_I:-}" || -z "${GRID_J:-}" || -z "${GRID_K:-}" ]] \
  || ! [[ "$GRID_I" =~ ^[0-9]+$ && "$GRID_J" =~ ^[0-9]+$ && "$GRID_K" =~ ^[0-9]+$ ]]; then
  echo "Phase $PHASE_LABEL GRID must contain three comma-separated non-negative integers" >&2
  exit 2
fi
if (( GRID_I < 5 || GRID_J < 5 || GRID_K < 5 )); then
  echo "Phase $PHASE_LABEL requires every 3-D grid dimension >= hm=5; got GRID=$GRID" >&2
  exit 2
fi

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$CASE_DIR" \
    --dst-case "$OUT_DIR/$target" \
    --input-name input.sod \
    --flowtype sod \
    --homogeneous t,t,t \
    --bctype 1,1,1,1,1,1 \
    --use-gpu "$use_gpu" \
    --maxstep "$MAXSTEP" \
    --feqchkpt "$FEQCHKPT" \
    --lfilter f \
    --diffterm f \
    --scheme 643e \
    --conschm 543e \
    --difschm 643e \
    --recon-schem "$RECON_SCHEM" \
    --lchardecomp f \
    --grid "$GRID" \
    --deltat "$DELTAT"
}

mkdir -p "$OUT_DIR"
prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.sod > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.sod > gpu.log 2>&1
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

if [[ "$COMPARE_EXACT" == "t" ]]; then
  for target in cpu gpu; do
    python3 "$ROOT_DIR/tests/gpu_validation/sod_exact_profile.py" \
      --case "$OUT_DIR/$target" \
      --report "$OUT_DIR/${target}_exact_profile.txt" \
      --analysis-half-width "$EXACT_ANALYSIS_HALF_WIDTH" \
      --exclude-cells "$EXACT_EXCLUDE_CELLS" \
      --max-smooth-l1 "$EXACT_MAX_SMOOTH_L1" \
      --max-smooth-l2 "$EXACT_MAX_SMOOTH_L2" \
      --max-smooth-linf "$EXACT_MAX_SMOOTH_LINF" \
      --max-bound-violation "$EXACT_MAX_BOUND_VIOLATION" \
      --max-contact-thickness-cells "$EXACT_MAX_CONTACT_THICKNESS_CELLS" \
      --max-shock-thickness-cells "$EXACT_MAX_SHOCK_THICKNESS_CELLS" \
      --max-position-error-cells "$EXACT_MAX_POSITION_ERROR_CELLS"
  done
fi
