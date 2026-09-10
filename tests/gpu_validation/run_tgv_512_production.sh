#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_EXE="${PYTHON_EXE:-python3}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:?OUT_DIR must name the production evidence directory}"
CFL_DT_ONE_FILE="${CFL_DT_ONE_FILE:?CFL_DT_ONE_FILE must contain the measured CFL=1 time step}"
DLR_REFERENCE="${DLR_REFERENCE:-$ROOT_DIR/documents/reference_data/tgv/spectral_Re1600_512.gdiag}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NP="${NP:-4}"
TOPOLOGY="${TOPOLOGY:-1,1,4}"
TARGET_TIME="${TARGET_TIME:-20.0}"
TARGET_CFL="${TARGET_CFL:-0.50}"
SEGMENT_STEPS="${SEGMENT_STEPS:-2000}"
MAX_RETRIES="${MAX_RETRIES:-1}"
SYNC_MODE="${SYNC_MODE:-explicit}"
HALO_TRANSPORT="${HALO_TRANSPORT:-pinned-overlap}"
FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"
CASE_DIR="$OUT_DIR/case"
SEGMENT_STATUS="$OUT_DIR/segment_status.tsv"

for required in "$GPU_EXE" "$CFL_DT_ONE_FILE" "$DLR_REFERENCE"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing required production input: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ ! -x "$GPU_EXE" ]]; then
  printf 'GPU executable is not executable: %s\n' "$GPU_EXE" >&2
  exit 2
fi

read -r cfl_dt_one < "$CFL_DT_ONE_FILE"
read -r deltat total_steps actual_target_cfl final_time < <(
  "$PYTHON_EXE" - "$cfl_dt_one" "$TARGET_CFL" "$TARGET_TIME" <<'PY'
import math
import sys

dt_one, target_cfl, target_time = map(float, sys.argv[1:])
if not math.isfinite(dt_one) or dt_one <= 0.0:
    raise SystemExit("measured CFL=1 time step must be positive and finite")
if not 0.0 < target_cfl < 1.0:
    raise SystemExit("TARGET_CFL must be between zero and one")
if not math.isfinite(target_time) or target_time <= 0.0:
    raise SystemExit("TARGET_TIME must be positive and finite")
raw_dt = dt_one * target_cfl
dt = math.floor(raw_dt * 1.0e12) / 1.0e12
if dt <= 0.0:
    raise SystemExit("rounded production time step is not positive")
steps = math.ceil(target_time / dt)
print(f"{dt:.12g} {steps} {dt / dt_one:.12g} {steps * dt:.12g}")
PY
) || exit 2

mkdir -p "$OUT_DIR"
if [[ ! -e "$SEGMENT_STATUS" ]]; then
  printf 'segment_start\tsegment_end\tattempt\tstatus\tlog\n' > "$SEGMENT_STATUS"
fi
cat > "$OUT_DIR/production_contract.txt" <<EOF
grid=512,512,512
reynolds=1600
mach=0.1
scheme=643e
filter=explicit_tenth_order
diffusion=enabled
np=$NP
topology=$TOPOLOGY
gpu_ids=$GPU_IDS
sync_mode=$SYNC_MODE
halo_transport=$HALO_TRANSPORT
filter_workspace=$FILTER_WORKSPACE
cfl_dt_one=$cfl_dt_one
target_cfl=$TARGET_CFL
actual_initial_cfl=$actual_target_cfl
deltat=$deltat
target_time=$TARGET_TIME
total_steps=$total_steps
checkpoint_time=$final_time
segment_steps=$SEGMENT_STEPS
EOF

checkpoint_step() {
  local auxiliary="$1"
  [[ -s "$auxiliary" ]] || return 1
  awk -F= '/^[[:space:]]*nstep=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' "$auxiliary"
}

prepare_fresh_case() {
  local end_step="$1"
  "$PYTHON_EXE" "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
    --dst-case "$CASE_DIR" --use-gpu t --restart f \
    --grid 512,512,512 --maxstep "$end_step" --feqchkpt "$end_step" \
    --feqlist 1 --deltat "$deltat" --lfilter t --diffterm t --scheme 643e
}

configure_restart_segment() {
  local end_step="$1"
  PYTHONPATH="$ROOT_DIR/tests/gpu_validation" "$PYTHON_EXE" - "$CASE_DIR" "$end_step" "$deltat" <<'PY'
import sys
from pathlib import Path
from prepare_tgv_case import set_controller_deltat, set_controller_steps, set_restart

case = Path(sys.argv[1])
end_step = int(sys.argv[2])
set_restart(case / "datin/input.tgv", "t")
set_controller_steps(case / "datin/controller", end_step, end_step, 1)
set_controller_deltat(case / "datin/controller", sys.argv[3])
PY
}

restore_checkpoint() {
  local expected_step="$1" current backup_step
  current="$(checkpoint_step "$CASE_DIR/outdat/auxiliary.txt" 2>/dev/null || true)"
  if [[ "$current" == "$expected_step" ]]; then
    return 0
  fi
  backup_step="$(checkpoint_step "$CASE_DIR/bakup/auxiliary.txt" 2>/dev/null || true)"
  if [[ "$backup_step" != "$expected_step" || ! -s "$CASE_DIR/bakup/flowfield.h5" ]]; then
    printf 'cannot recover checkpoint step %s; current=%s backup=%s\n' \
      "$expected_step" "${current:-missing}" "${backup_step:-missing}" >&2
    return 1
  fi
  cp -f "$CASE_DIR/bakup/auxiliary.txt" "$CASE_DIR/outdat/auxiliary.txt"
  cp -f "$CASE_DIR/bakup/flowfield.h5" "$CASE_DIR/outdat/flowfield.h5"
}

validate_segment() {
  local log="$1" expected_step="$2" stored_step
  grep -aq 'The job is done!' "$log" || return 1
  if grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
    return 1
  fi
  awk '/current CFL:/ {seen=1; if ($NF+0 >= 1.0 || $NF+0 <= 0.0) bad=1} END {exit !(seen && !bad)}' "$log" || return 1
  stored_step="$(checkpoint_step "$CASE_DIR/outdat/auxiliary.txt" 2>/dev/null || true)"
  [[ "$stored_step" == "$expected_step" && -s "$CASE_DIR/outdat/flowfield.h5" ]]
}

current_step="$(checkpoint_step "$CASE_DIR/outdat/auxiliary.txt" 2>/dev/null || true)"
current_step="${current_step:-0}"
if (( current_step > total_steps )); then
  printf 'existing checkpoint step %s exceeds target %s\n' "$current_step" "$total_steps" >&2
  exit 2
fi

while (( current_step < total_steps )); do
  end_step=$((current_step + SEGMENT_STEPS))
  if (( end_step > total_steps )); then
    end_step="$total_steps"
  fi
  if (( current_step == 0 )); then
    prepare_fresh_case "$end_step" || exit 2
  else
    restore_checkpoint "$current_step" || exit 2
    configure_restart_segment "$end_step" || exit 2
  fi

  segment_ok=f
  for ((attempt=0; attempt<=MAX_RETRIES; attempt++)); do
    log="$OUT_DIR/segment_${current_step}_${end_step}_attempt_${attempt}.log"
    (
      cd "$CASE_DIR" || exit 2
      OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}" \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      ASTR_GPU_SYNC_MODE="$SYNC_MODE" ASTR_GPU_HALO_TRANSPORT="$HALO_TRANSPORT" \
      ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
        mpirun -np "$NP" "$GPU_EXE" run datin/input.tgv > "$log" 2>&1
    )
    run_status=$?
    if (( run_status == 0 )) && validate_segment "$log" "$end_step"; then
      printf '%s\t%s\t%s\tPASS\t%s\n' "$current_step" "$end_step" "$attempt" "$log" >> "$SEGMENT_STATUS"
      segment_ok=t
      break
    fi
    printf '%s\t%s\t%s\tFAIL\t%s\n' "$current_step" "$end_step" "$attempt" "$log" >> "$SEGMENT_STATUS"
    if (( current_step == 0 )); then
      prepare_fresh_case "$end_step" || exit 2
    else
      restore_checkpoint "$current_step" || exit 2
      configure_restart_segment "$end_step" || exit 2
    fi
  done
  if [[ "$segment_ok" != t ]]; then
    printf 'production segment %s to %s failed after %s retries\n' \
      "$current_step" "$end_step" "$MAX_RETRIES" >&2
    exit 1
  fi
  current_step="$end_step"
done

"$PYTHON_EXE" "$ROOT_DIR/tests/gpu_validation/compare_tgv_dlr_reference.py" \
  --reference "$DLR_REFERENCE" --astr-flowstate "$CASE_DIR/flowstate.dat" \
  --output-dir "$OUT_DIR/dlr_comparison"
