#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new evidence directory}"
NX="${NX:-33}"
NY="${NY:-33}"
NZ="${NZ:-9}"
DELTAT="${DELTAT:-1.0e-6}"
FIELD_ATOL="${FIELD_ATOL:-1.0e-10}"

if [[ -e "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUT_DIR"

set_restart_input() {
  local case_dir="$1"
  local maxstep="$2"
  local feqchkpt="$3"
  sed -i '/^# lrestar$/{n;s/^[[:space:]]*f[[:space:]]*$/t/;}' \
    "$case_dir/datin/input.opensbli"
  sed -i "/^# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg$/{n;s/.*/${maxstep},${feqchkpt},9999,9999,1,9999/;}" \
    "$case_dir/datin/controller"
}

run_solver() {
  local kind="$1"
  local np="$2"
  local case_dir="$3"
  local log_name="$4"
  local exe="$ROOT_DIR/build_${kind}_probe/bin/astr"
  local topology="1,1,1"
  if [[ "$np" == "2" ]]; then
    topology="2,1,1"
  fi
  (
    cd "$case_dir"
    CUDA_VISIBLE_DEVICES=0,1 \
    OMPI_MCA_sharedfp=individual \
    ASTR_FORCE_MPI_TOPOLOGY="$topology" \
    ASTR_PERFECT_GAS_PRANDTL=0.72 \
    ASTR_SUTHERLAND_TEMPERATURE_K=110.4 \
    ASTR_CONSERVATIVE_BOUNDARY_FILE="$case_dir/datin/conservative_boundary.nml" \
      timeout --kill-after=10s 180s mpirun --oversubscribe -np "$np" \
      "$exe" run datin/input.opensbli > "$log_name" 2>&1
  )
}

prepare_case() {
  local kind="$1"
  local destination="$2"
  local maxstep="$3"
  local use_gpu=f
  if [[ "$kind" == "gpu" ]]; then
    use_gpu=t
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_opensbli_astr_case.py" \
    --destination "$destination" --use-gpu "$use_gpu" \
    --nx "$NX" --ny "$NY" --nz "$NZ" --maxstep "$maxstep" \
    --feqchkpt "$maxstep" --deltat "$DELTAT" --prandtl 0.72 \
    > "$destination.prepare.json"
}

for spec in cpu:1 gpu:2; do
  kind="${spec%%:*}"
  np="${spec##*:}"
  continuous="$OUT_DIR/${kind}_np${np}_continuous"
  split="$OUT_DIR/${kind}_np${np}_restart"
  mismatch="$OUT_DIR/${kind}_np${np}_mismatch"

  prepare_case "$kind" "$continuous" 4
  run_solver "$kind" "$np" "$continuous" continuous.log
  grep -q 'The job is done!' "$continuous/continuous.log"

  prepare_case "$kind" "$split" 2
  run_solver "$kind" "$np" "$split" fresh.log
  grep -q 'The job is done!' "$split/fresh.log"

  cp -a "$split" "$mismatch"
  set_restart_input "$mismatch" 3 3
  sed -i 's/nstep= *2/nstep=               1/' "$mismatch/outdat/auxiliary.txt"
  set +e
  run_solver "$kind" "$np" "$mismatch" mismatch.log
  mismatch_status=$?
  set -e
  if [[ "$mismatch_status" == "0" ]] || \
     ! grep -q 'Checkpoint step mismatch' "$mismatch/mismatch.log"; then
    printf '%s NP=%s accepted inconsistent checkpoint metadata\n' "$kind" "$np" >&2
    exit 1
  fi

  set_restart_input "$split" 4 4
  run_solver "$kind" "$np" "$split" restart.log
  grep -q 'The job is done!' "$split/restart.log"
  grep -q 'checkpoint file read' "$split/restart.log"
  flowstate_steps="$(awk 'NR>1 {print $1}' "$split/flowstate.dat" | paste -sd, -)"
  if [[ "$flowstate_steps" != "1,2,3,4" ]]; then
    printf '%s NP=%s restart flowstate steps are %s, expected 1,2,3,4\n' \
      "$kind" "$np" "$flowstate_steps" >&2
    exit 1
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$continuous/outdat/flowfield.h5" \
    --gpu "$split/outdat/flowfield.h5" \
    --report "$OUT_DIR/${kind}_np${np}_field_compare.txt" \
    --atol "$FIELD_ATOL" --rtol 0
done

printf 'OPENSBLI_RESTART_EQUIVALENCE_PASS\n'
