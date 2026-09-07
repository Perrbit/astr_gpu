#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=/home/dell/workspace/astr_gpu
CASE_DIR="${ASTR_OPENSBLI_CASE_DIR:-${ASTR_OPENSLBI_CASE_DIR:-}}"
: "${CASE_DIR:?ASTR_OPENSBLI_CASE_DIR must name a prepared case}"

RESTART_MODE="${ASTR_OPENSBLI_RESTART:-0}"
GPU_EXE="$ROOT_DIR/build_gpu_probe/bin/astr"
REFERENCE_ZIP="$ROOT_DIR/documents/reference_data/opensbli_katzer/opensbli_katzer_data.zip"
BOUNDARY_FILE="$CASE_DIR/datin/conservative_boundary.nml"

for required in "$GPU_EXE" "$REFERENCE_ZIP" "$BOUNDARY_FILE" \
  "$CASE_DIR/datin/input.opensbli" "$CASE_DIR/datin/controller"; do
  if [[ ! -f "$required" ]]; then
    printf 'missing required file: %s\n' "$required" >&2
    exit 2
  fi
done

if [[ -e "$CASE_DIR/run.exit" ]]; then
  printf 'refusing to reuse a completed or failed service run in %s\n' "$CASE_DIR" >&2
  exit 2
fi
if [[ "$RESTART_MODE" == "1" ]]; then
  for required in "$CASE_DIR/outdat/flowfield.h5" "$CASE_DIR/outdat/auxiliary.txt"; do
    if [[ ! -f "$required" ]]; then
      printf 'missing restart checkpoint file: %s\n' "$required" >&2
      exit 2
    fi
  done
  restart_flag="$(awk '/^# lrestar$/{getline; gsub(/[[:space:]]/, ""); print; exit}' \
    "$CASE_DIR/datin/input.opensbli")"
  if [[ "$restart_flag" != "t" ]]; then
    printf 'restart service requires lrestar=t in input.opensbli\n' >&2
    exit 2
  fi
elif [[ -s "$CASE_DIR/flowstate.dat" || -e "$CASE_DIR/outdat/flowfield.h5" ]]; then
  printf 'refusing to overwrite an existing fresh run in %s\n' "$CASE_DIR" >&2
  exit 2
fi

on_signal() {
  printf '%s signal=%s\n' "$(date --iso-8601=seconds)" "$1" > "$CASE_DIR/service_terminated.txt"
  exit 143
}
trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP

cd "$CASE_DIR" || exit 2
rm -f service_terminated.txt
printf '%s service run starting\n' "$(date --iso-8601=seconds)"
set +e
mpirun -np 2 "$GPU_EXE" run datin/input.opensbli >> gpu.log 2>&1
run_status=$?
set -e
printf '%s\n' "$run_status" > run.exit

if (( run_status != 0 )); then
  printf '%s solver exited with status %d\n' "$(date --iso-8601=seconds)" "$run_status" >&2
  exit "$run_status"
fi
if ! tr -d '\000' < gpu.log | grep -q 'The job is done!'; then
  printf '%s solver returned zero without completion marker\n' "$(date --iso-8601=seconds)" >&2
  exit 3
fi

set +e
python3 "$ROOT_DIR/tests/gpu_validation/compare_opensbli_katzer_statistics.py" \
  --astr-flow "$CASE_DIR/outdat/flowfield.h5" \
  --astr-grid "$CASE_DIR/datin/grid.flatplate.h5" \
  --reference-zip "$REFERENCE_ZIP" \
  --output-dir "$CASE_DIR/database_comparison" \
  --expected-time 13000 > database_compare.log 2>&1
compare_status=$?
set -e
printf '%s\n' "$compare_status" > database_compare.exit
printf '%s service run completed, comparison status=%d\n' \
  "$(date --iso-8601=seconds)" "$compare_status"
exit "$compare_status"
