#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR_INPUT="${OUT_DIR:?OUT_DIR must name a new evidence directory}"
OUT_DIR="$(realpath -m "$OUT_DIR_INPUT")"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-32,6,6}"
MAXSTEP="${MAXSTEP:-600}"
DELTAT="${DELTAT:-8.d-8}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
CHECKPOINT_FREQUENCY="${CHECKPOINT_FREQUENCY:-20}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
MAX_CFL="${MAX_CFL:-1.0}"
SHOCK_SKIP_CELLS="${SHOCK_SKIP_CELLS:-3}"
RESUME="${RESUME:-0}"
OUTLET_REFERENCE_LENGTH="${OUTLET_REFERENCE_LENGTH:-0.01}"
outlet_args=()

ELAPSED_TIME="$(python3 - "$MAXSTEP" "$DELTAT" <<'PY'
import sys

step = int(sys.argv[1])
dt = float(sys.argv[2].lower().replace("d", "e"))
print(f"{(step + 1) * dt:.17e}")
PY
)"

for value in "$MAXSTEP" "$NP" "$CHECKPOINT_FREQUENCY" "$TIMEOUT_SECONDS" \
             "$SHOCK_SKIP_CELLS"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf 'Integer runtime controls must be non-negative: %s\n' "$value" >&2
    exit 2
  fi
done
if (( MAXSTEP < 1 || NP < 1 || CHECKPOINT_FREQUENCY < 1 || TIMEOUT_SECONDS < 1 )); then
  printf 'MAXSTEP, NP, CHECKPOINT_FREQUENCY, and TIMEOUT_SECONDS must be positive\n' >&2
  exit 2
fi
if [[ "$RESUME" != 0 && "$RESUME" != 1 ]]; then
  printf 'RESUME must be 0 or 1\n' >&2
  exit 2
fi
OUTLET_REFERENCE_LENGTH="$(python3 - "$OUTLET_REFERENCE_LENGTH" <<'PY'
import sys
length = float(sys.argv[1])
if length not in (0.0, 0.01):
    raise SystemExit("this case has a fixed 0.01 m reference relaxation length; use 0 for legacy extrapolation")
print("0" if length == 0.0 else "0.01")
PY
)"
if [[ "$OUTLET_REFERENCE_LENGTH" != 0 ]]; then
  outlet_args=(--outlet-reference-length "$OUTLET_REFERENCE_LENGTH")
fi
IFS=',' read -r GRID_X GRID_Y GRID_Z <<< "$GRID"
IFS=',' read -r TOPOLOGY_X TOPOLOGY_Y TOPOLOGY_Z <<< "$TOPOLOGY"
if (( TOPOLOGY_X * TOPOLOGY_Y * TOPOLOGY_Z != NP )); then
  printf 'TOPOLOGY product must equal NP\n' >&2
  exit 2
fi
CASE_DIR="$OUT_DIR/gpu"
if [[ "$RESUME" == 0 && -e "$OUT_DIR" ]]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
if [[ "$RESUME" == 1 && ! -f "$CASE_DIR/outdat/flowfield.h5" ]]; then
  printf 'Resume checkpoint file is missing: %s\n' "$CASE_DIR/outdat/flowfield.h5" >&2
  exit 2
fi
if [[ ! -x "$EXE" ]]; then
  printf 'ASTR executable is missing or not executable: %s\n' "$EXE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR" "$TMP_DIR"
if [[ "$RESUME" == 0 ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
    --destination "$CASE_DIR" --grid "$GRID" --maxstep "$MAXSTEP" \
    --deltat "$DELTAT" --list-frequency "$CHECKPOINT_FREQUENCY" \
    --diffterm f --lfilter f --use-gpu t --initial-condition normal-shock >/dev/null
  python3 "$ROOT_DIR/tests/gpu_validation/generate_air5_normal_shock_states.py" \
    --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
    --output "$CASE_DIR/datin/air5_normal_shock_states.dat" "${outlet_args[@]}"
fi

python3 - "$OUT_DIR/run_contract.json" "$RESUME" "$GRID" "$DELTAT" \
  "$NP" "$TOPOLOGY" "$OUTLET_REFERENCE_LENGTH" "$CASE_DIR/datin/air5_normal_shock_states.dat" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
resume = bool(int(sys.argv[2]))
requested = {
    "grid": sys.argv[3],
    "deltat": float(sys.argv[4].lower().replace("d", "e")),
    "np": int(sys.argv[5]),
    "topology": sys.argv[6],
    "outlet_reference_length": float(sys.argv[7]),
    "boundary_states_sha256": hashlib.sha256(Path(sys.argv[8]).read_bytes()).hexdigest(),
}
if resume:
    if not path.is_file():
        raise SystemExit(f"resume run contract is missing: {path}")
    recorded = json.loads(path.read_text(encoding="utf-8"))
    if recorded != requested:
        raise SystemExit(
            f"resume configuration differs from the original run: "
            f"recorded={recorded}, requested={requested}"
        )
else:
    path.write_text(json.dumps(requested, indent=2) + "\n", encoding="utf-8")
PY

set_restart_case() {
  python3 - "$CASE_DIR/datin/input.air5_c4" <<'PY'
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
if [[ "$RESUME" == 1 ]]; then
  set_restart_case
fi

python3 - "$CASE_DIR/datin/controller" "$MAXSTEP" "$CHECKPOINT_FREQUENCY" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
maxstep, frequency = sys.argv[2:4]
lines = path.read_text(encoding="utf-8").splitlines()
marker = "maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg"
for index, line in enumerate(lines):
    if marker in line:
        values = [item.strip() for item in lines[index + 1].split(",")]
        if len(values) != 6:
            raise ValueError("unexpected air5 controller step line")
        values[0], values[1], values[4] = maxstep, frequency, frequency
        lines[index + 1] = ",".join(values)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        break
else:
    raise ValueError("air5 controller step marker is missing")
PY

mkdir -p "$CASE_DIR/validation"
LOG_FILE="$CASE_DIR/run_to_step${MAXSTEP}.log"
(
  cd "$CASE_DIR"
  TMPDIR="$TMP_DIR" \
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
    ASTR_VALIDATION_RHS_STEP=0 \
    ASTR_VALIDATION_RHS_STEP_SECONDARY="$MAXSTEP" \
    ASTR_AIR5_C4_CONSERVATION=f \
    ASTR_AIR5_SOURCE_MODE=coupled \
    OMPI_MCA_coll='^hcoll,ucc' \
    OMPI_MCA_pml=ob1 \
    OMPI_MCA_btl=self,vader,tcp \
    OMPI_MCA_osc=pt2pt \
    OMPI_MCA_opal_cuda_support=0 \
    UCX_MEMTYPE_CACHE=n \
    timeout --kill-after=30s "${TIMEOUT_SECONDS}s" mpirun -np "$NP" \
    "$EXE" run datin/input.air5_c4 >"$LOG_FILE" 2>&1
)
grep -Fq 'The job is done!' "$LOG_FILE"
grep -Fq 'ASTR_AIR5_SOURCE_MODE=coupled' "$LOG_FILE"

python3 - "$LOG_FILE" "$MAX_CFL" <<'PY'
from pathlib import Path
import math
import re
import sys

values = [
    float(match.group(1))
    for match in re.finditer(r"current CFL:\s+([0-9.Ee+-]+)", Path(sys.argv[1]).read_text())
]
limit = float(sys.argv[2])
if not values or any(not math.isfinite(value) or value >= limit for value in values):
    raise SystemExit(f"invalid CFL history: {values}, limit={limit}")
PY

python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_normal_shock.py" \
  --prefix "$CASE_DIR/validation/air5" \
  --boundary-states "$CASE_DIR/datin/air5_normal_shock_states.dat" \
  --topology "$TOPOLOGY" \
  --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
  --report "$OUT_DIR/short_contract.txt"
python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c5_captured_normal_shock.py" \
  --prefix "$CASE_DIR/validation/air5" \
  --mechanism "$ROOT_DIR/chemMech/air5_kimjo12.json" \
  --topology "$TOPOLOGY" --point-count "$((GRID_X + 1))" \
  --step "$MAXSTEP" --domain-length 0.02 \
  --elapsed-time "$ELAPSED_TIME" --minimum-flowthroughs 1.0 \
  --previous-checkpoint "$CASE_DIR/bakup/flowfield.h5" \
  --current-checkpoint "$CASE_DIR/outdat/flowfield.h5" \
  --shock-skip-cells "$SHOCK_SKIP_CELLS" \
  --report "$OUT_DIR/physical_contract.txt"

printf 'AIR5_C5_CAPTURED_NORMAL_SHOCK_PASS\n'
