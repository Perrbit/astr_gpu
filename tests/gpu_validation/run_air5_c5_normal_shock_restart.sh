#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTLET_REFERENCE_LENGTH="${OUTLET_REFERENCE_LENGTH:-}"
outlet_args=()
if [[ -n "$OUTLET_REFERENCE_LENGTH" ]]; then
  outlet_args=(--outlet-reference-length "$OUTLET_REFERENCE_LENGTH")
fi
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new evidence directory}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-8,6,6}"
DELTAT="${DELTAT:-1.d-8}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
CONTINUOUS_STEP=2
SPLIT_STEP=1

if [[ -e "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
if [[ ! -x "$EXE" ]]; then
  printf 'ASTR executable is missing or not executable: %s\n' "$EXE" >&2
  exit 2
fi
mkdir -p "$OUT_DIR" "$TMP_DIR"

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
  local destination="$1" maxstep="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
    --destination "$destination" --grid "$GRID" --maxstep "$maxstep" \
    --deltat "$DELTAT" --list-frequency 1 --diffterm t --lfilter f \
    --use-gpu t --initial-condition normal-shock >/dev/null
  python3 "$ROOT_DIR/tests/gpu_validation/generate_air5_normal_shock_states.py" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --output "$destination/datin/air5_normal_shock_states.dat" "${outlet_args[@]}"
}

run_case() {
  local case_dir="$1" validation_step="$2" log_name="$3"
  mkdir -p "$case_dir/validation"
  (
    cd "$case_dir"
    TMPDIR="$TMP_DIR" \
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
      timeout --kill-after=10s "${TIMEOUT_SECONDS}s" mpirun -np 1 \
      "$EXE" run datin/input.air5_c4 >"$log_name" 2>&1
  )
  grep -Fq 'The job is done!' "$case_dir/$log_name"
}

continuous="$OUT_DIR/gpu_continuous"
restarted="$OUT_DIR/gpu_restart"

prepare_case "$continuous" "$CONTINUOUS_STEP"
run_case "$continuous" "$CONTINUOUS_STEP" continuous.log

prepare_case "$restarted" "$SPLIT_STEP"
set_controller_steps "$restarted/datin/controller" "$SPLIT_STEP" "$SPLIT_STEP"
run_case "$restarted" 999999 fresh.log
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
run_case "$restarted" "$CONTINUOUS_STEP" restart.log
grep -Fq 'checkpoint file read' "$restarted/restart.log"

python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
  --cpu-prefix "$continuous/validation/air5" \
  --gpu-prefix "$restarted/validation/air5" \
  --report "$OUT_DIR/gpu_restart_compare.txt" \
  --labels post_chemistry,pre_rhs,post_update,post_transport \
  --atol 1e-9 --rtol 1e-10 --scaled-tol 5e-9 --active-only

printf 'AIR5_C5_NORMAL_SHOCK_RESTART_PASS\n'
