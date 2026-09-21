#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new evidence directory}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
GRID="${GRID:-15,15,7}"
DELTAT="${DELTAT:-1.d-10}"
CONTINUOUS_STEP=4
SPLIT_STEP=2
HBL_INITIAL_FIELD=matched

if [[ -e "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
for exe in "$CPU_EXE" "$GPU_EXE"; do
  if [[ ! -x "$exe" ]]; then
    printf 'ASTR executable is missing or not executable: %s\n' "$exe" >&2
    exit 2
  fi
done
mkdir -p "$OUT_DIR"

set_controller_steps() {
  local controller="$1" maxstep="$2" feqchkpt="$3"
  python3 - "$controller" "$maxstep" "$feqchkpt" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
maxstep, checkpoint = sys.argv[2:4]
lines = path.read_text(encoding="utf-8").splitlines()
marker = "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg"
for index, line in enumerate(lines):
    if marker in line:
        values = [item.strip() for item in lines[index + 1].split(",")]
        if len(values) != 6:
            raise ValueError("unexpected air5 controller step line")
        values[0], values[1] = maxstep, checkpoint
        lines[index + 1] = ",".join(values)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        break
else:
    raise ValueError("air5 controller step marker is missing")
PY
}

set_restart_case() {
  local input_file="$1"
  python3 - "$input_file" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
for index, line in enumerate(lines):
    if line.strip().startswith("# lrestar"):
        lines[index + 1] = "t"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        break
else:
    raise ValueError("air5 restart marker is missing")
PY
}

prepare_case() {
  local destination="$1" use_gpu="$2" maxstep="$3"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
    --destination "$destination" --grid "$GRID" --maxstep "$maxstep" \
    --deltat "$DELTAT" --list-frequency 1 --diffterm t \
    --use-gpu "$use_gpu" --initial-condition high-enthalpy-boundary-layer \
    --hbl-initial-field "$HBL_INITIAL_FIELD" >/dev/null
}

run_case() {
  local exe="$1" case_dir="$2" validation_step="$3" log_name="$4"
  mkdir -p "$case_dir/validation"
  (
    cd "$case_dir"
    ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
    ASTR_VALIDATION_RHS_STEP="$validation_step" \
    ASTR_AIR5_C4_CONSERVATION=f \
    ASTR_AIR5_SOURCE_MODE=coupled \
    OMPI_MCA_coll='^hcoll,ucc' \
    OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self,vader,tcp \
    OMPI_MCA_osc=pt2pt \
    OMPI_MCA_opal_cuda_support=0 \
    UCX_MEMTYPE_CACHE=n \
      timeout --kill-after=10s 300s mpirun --oversubscribe -np 1 \
      "$exe" run datin/input.air5_c4 >"$log_name" 2>&1
  )
  grep -q 'The job is done!' "$case_dir/$log_name"
}

compare_restart() {
  local continuous="$1" restarted="$2" report="$3"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
    --cpu-prefix "$continuous/validation/air5" \
    --gpu-prefix "$restarted/validation/air5" \
    --report "$report" \
    --labels post_chemistry,pre_rhs,post_update,post_transport \
    --atol 1e-9 --rtol 1e-10 --scaled-tol 5e-9 --active-only
}

for mode in cpu gpu; do
  exe="$CPU_EXE"
  use_gpu=f
  if [[ "$mode" == gpu ]]; then
    exe="$GPU_EXE"
    use_gpu=t
  fi
  continuous="$OUT_DIR/${mode}_continuous"
  restarted="$OUT_DIR/${mode}_restart"

  prepare_case "$continuous" "$use_gpu" "$CONTINUOUS_STEP"
  run_case "$exe" "$continuous" "$CONTINUOUS_STEP" continuous.log

  prepare_case "$restarted" "$use_gpu" "$SPLIT_STEP"
  set_controller_steps "$restarted/datin/controller" "$SPLIT_STEP" "$SPLIT_STEP"
  run_case "$exe" "$restarted" 999999 fresh.log
  python3 - "$restarted/outdat/flowfield.h5" <<'PY'
import h5py
from pathlib import Path
import sys

with h5py.File(Path(sys.argv[1]), "r") as checkpoint:
    if "tv" not in checkpoint:
        raise SystemExit("fixed air5 checkpoint is missing tv")
PY
  set_restart_case "$restarted/datin/input.air5_c4"
  set_controller_steps "$restarted/datin/controller" "$CONTINUOUS_STEP" "$CONTINUOUS_STEP"
  run_case "$exe" "$restarted" "$CONTINUOUS_STEP" restart.log
  grep -q 'checkpoint file read' "$restarted/restart.log"
  compare_restart "$continuous" "$restarted" \
    "$OUT_DIR/${mode}_restart_compare.txt"
done

compare_restart "$OUT_DIR/cpu_continuous" "$OUT_DIR/gpu_continuous" \
  "$OUT_DIR/cpu_gpu_continuous_compare.txt"
printf 'AIR5_HBL_RESTART_EQUIVALENCE_PASS\n'
