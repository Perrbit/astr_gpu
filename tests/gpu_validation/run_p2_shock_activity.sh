#!/usr/bin/env bash
set -euo pipefail

P2_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$P2_ROOT_DIR/tests/gpu_validation/p2_shock_case_common.sh"

CASE="${CASE:-shuosher}"
p2_validate_case "$CASE"
GPU_EXE="${GPU_EXE:-$P2_ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$P2_ROOT_DIR/tests/gpu_validation/out/p2_${CASE}_activity}"
GRID="${GRID:-$(p2_default_grid "$CASE")}"
GPU_ID="${GPU_ID:-0}"
MAXSTEP=1
FEQCHKPT=9999
FEQLIST=9999

if [[ "$GPU_EXE" != /* ]]; then GPU_EXE="$P2_ROOT_DIR/$GPU_EXE"; fi
if [[ "$OUT_DIR" != /* ]]; then OUT_DIR="$P2_ROOT_DIR/$OUT_DIR"; fi
[[ -x "$GPU_EXE" ]] || { printf 'GPU executable not found: %s\n' "$GPU_EXE" >&2; exit 2; }

CASE_DIR="$OUT_DIR/case"
SENSOR="$OUT_DIR/shock_sensor.dat"
LOG="$OUT_DIR/run.log"
mkdir -p "$OUT_DIR" "$OUT_DIR/tmp"
p2_prepare_case "$CASE" "$CASE_DIR" "$GRID" "$MAXSTEP" "$FEQCHKPT" "$FEQLIST"

(
  cd "$CASE_DIR"
  env "${P2_RUNTIME_ENV[@]}" TMPDIR="$OUT_DIR/tmp" \
    OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}" CUDA_VISIBLE_DEVICES="$GPU_ID" \
    ASTR_FORCE_MPI_TOPOLOGY=1,1,1 ASTR_GPU_SYNC_MODE=explicit \
    ASTR_SHOCK_SENSOR_DUMP="$SENSOR" \
    mpirun -np 1 "$GPU_EXE" run "$P2_INPUT" > "$LOG" 2>&1
)
grep -q 'The job is done!' "$LOG"
python3 "$P2_ROOT_DIR/tests/gpu_validation/summarize_shock_activity.py" \
  --sensor "$SENSOR" --report "$OUT_DIR/shock_activity.md"
