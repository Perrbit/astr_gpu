#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/curvilinear_nscbc52_acoustic_np1}"
INPUT_NAME="${INPUT_NAME:-input.tgv}"
GRID_LEVELS="${GRID_LEVELS:-64,48,64;80,60,80;96,72,96}"
AMPLITUDE="${AMPLITUDE:-0.15}"
GRID_MAPPING="${GRID_MAPPING:-y-upper-ramp}"
RAMP_START_FRACTION="${RAMP_START_FRACTION:-0.72}"
PROBE_FRACTION="${PROBE_FRACTION:-2,3}"
MACH="${MACH:-0.3}"
GAMMA="${GAMMA:-1.4}"
PULSE_AMPLITUDE="${PULSE_AMPLITUDE:-1e-4}"
PULSE_CENTER="${PULSE_CENTER:-3.141592653589793,3.0,3.141592653589793}"
PULSE_WIDTH="${PULSE_WIDTH:-1.0}"
MIN_SUPPORT_INTERVALS="${MIN_SUPPORT_INTERVALS:-15.0}"
PULSE_DIRECTION="${PULSE_DIRECTION:-0.0,1.0,0.0}"
ACOUSTIC_PROFILE="${ACOUSTIC_PROFILE:-plane-y-compact}"
DELTAT="${DELTAT:-0.005}"
MAXSTEP="${MAXSTEP:-440}"
FEQCHKPT="${FEQCHKPT:-4}"
INCIDENT_WINDOW="${INCIDENT_WINDOW:-0.04,0.72}"
REFLECTED_WINDOW="${REFLECTED_WINDOW:-1.15,2.10}"
REFLECTION_MAX="${REFLECTION_MAX:-0.05}"
CONTROL_RATIO_MAX="${CONTROL_RATIO_MAX:-0.25}"
REFLECTION_ATOL="${REFLECTION_ATOL:-1e-3}"
FIELD_ATOL="${FIELD_ATOL:-1e-10}"
DIFFTERM="${DIFFTERM:-f}"
SCHEME="${SCHEME:-643e}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
RUN_CONTROL="${RUN_CONTROL:-t}"

if [[ ! -x "$CPU_EXE" || ! -x "$GPU_EXE" ]]; then
  echo "CPU_EXE and GPU_EXE must point to built ASTR executables" >&2
  exit 2
fi

P0="$(python3 -c "print(1.0 / (float('$GAMMA') * float('$MACH')**2))")"
C0="$(python3 -c "print((float('$GAMMA') * float('$P0'))**0.5)")"
IFS=, read -r probe_numerator probe_denominator <<< "$PROBE_FRACTION"
if (( probe_numerator <= 0 || probe_denominator <= probe_numerator )); then
  echo "PROBE_FRACTION must satisfy 0 < numerator < denominator" >&2
  exit 4
fi
python3 - "$RAMP_START_FRACTION" "$probe_numerator" "$probe_denominator" <<'PY'
import sys

ramp_start = float(sys.argv[1])
probe_fraction = int(sys.argv[2]) / int(sys.argv[3])
if not 0.0 < probe_fraction < ramp_start < 1.0:
    raise SystemExit("probe plane must precede the upper-y geometry ramp")
PY

prepare_case() {
  local case_dir="$1"
  local use_gpu="$2"
  local grid="$3"
  local im="$4"
  local jm="$5"
  local km="$6"

  python3 "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py" \
    --src-case "$ROOT_DIR/examples/Taylor_Green_Vortex" \
    --dst-case "$case_dir" --input-name "$INPUT_NAME" \
    --use-gpu "$use_gpu" --mach "$MACH" --grid "$grid" --scheme "$SCHEME" \
    --diffterm "$DIFFTERM" --lfilter f --lreadgrid t --ninit 3 --restart f \
    --homogeneous t,f,t --bctype "1;1;41,1.0;52;1;1" \
    --gridfile ./datin/grid.acoustic.h5 \
    --maxstep "$MAXSTEP" --feqchkpt "$FEQCHKPT" --feqlist 9999 \
    --lwsequ t --feqwsequ "$FEQCHKPT" \
    --deltat "$DELTAT"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_tgv_grid.py" \
    --output "$case_dir/datin/grid.acoustic.h5" \
    --report "$case_dir/grid_generation.txt" --grid "$grid" \
    --mapping "$GRID_MAPPING" --amplitude "$AMPLITUDE" \
    --ramp-start-fraction "$RAMP_START_FRACTION"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_curvilinear_acoustic_pulse.py" \
    --grid "$case_dir/datin/grid.acoustic.h5" \
    --output "$case_dir/datin/flowini3d.h5" --rho0 1.0 --p0 "$P0" \
    --gamma "$GAMMA" --mach "$MACH" --amplitude "$PULSE_AMPLITUDE" \
    --center "$PULSE_CENTER" --width "$PULSE_WIDTH" \
    --direction "$PULSE_DIRECTION" --clearance-widths 2.5 \
    --profile "$ACOUSTIC_PROFILE" \
    --minimum-support-intervals "$MIN_SUPPORT_INTERVALS"
}

run_case() {
  local case_dir="$1"
  local executable="$2"
  local mode="$3"
  (
    cd "$case_dir"
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_NSCBC_FARFIELD_MODE="$mode" \
      mpirun -np "$NP" "$executable" run "datin/$INPUT_NAME" > run.log 2>&1
  )
}

analyze_case() {
  local case_dir="$1"
  local probe_index="$2"
  local output_dir="$case_dir/reflection"
  local snapshots=()
  mapfile -t snapshots < <(find "$case_dir/outdat" -maxdepth 1 -type f \
    -name 'flowfield[0-9][0-9][0-9][0-9].h5' | sort)
  if (( ${#snapshots[@]} < 2 )); then
    echo "insufficient acoustic snapshots in $case_dir" >&2
    exit 3
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/analyze_curvilinear_acoustic_reflection.py" \
    --grid "$case_dir/datin/grid.acoustic.h5" \
    --snapshots "${snapshots[@]}" --output-dir "$output_dir" \
    --probe-index "$probe_index" --rho0 1.0 --p0 "$P0" --c0 "$C0" \
    --incident-window "$INCIDENT_WINDOW" --reflected-window "$REFLECTED_WINDOW"
}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
IFS=';' read -r -a levels <<< "$GRID_LEVELS"
summary_files=()

for grid in "${levels[@]}"; do
  IFS=, read -r im jm km <<< "$grid"
  if (( jm % probe_denominator != 0 )); then
    echo "acoustic probe requires jm divisible by $probe_denominator: $grid" >&2
    exit 4
  fi
  level_dir="$OUT_DIR/${im}x${jm}x${km}"
  cpu_nr="$level_dir/cpu_nonreflecting"
  gpu_nr="$level_dir/gpu_nonreflecting"
  cpu_control="$level_dir/cpu_compatibility"
  prepare_case "$cpu_nr" f "$grid" "$im" "$jm" "$km"
  prepare_case "$gpu_nr" t "$grid" "$im" "$jm" "$km"
  if [[ "$RUN_CONTROL" == "t" ]]; then
    prepare_case "$cpu_control" f "$grid" "$im" "$jm" "$km"
  fi
  python3 "$ROOT_DIR/tests/gpu_validation/check_curvilinear_nscbc_geometry.py" \
    --grid "$cpu_nr/datin/grid.acoustic.h5" \
    --report "$level_dir/geometry.txt" --amplitude "$AMPLITUDE" \
    --mapping "$GRID_MAPPING" --ramp-start-fraction "$RAMP_START_FRACTION"

  run_case "$cpu_nr" "$CPU_EXE" nonreflecting
  run_case "$gpu_nr" "$GPU_EXE" nonreflecting
  if [[ "$RUN_CONTROL" == "t" ]]; then
    run_case "$cpu_control" "$CPU_EXE" compatibility
  fi

  probe_index=$((probe_numerator * jm / probe_denominator))
  analyze_case "$cpu_nr" "$probe_index"
  analyze_case "$gpu_nr" "$probe_index"
  if [[ "$RUN_CONTROL" == "t" ]]; then
    analyze_case "$cpu_control" "$probe_index"
  fi

  mapfile -t cpu_snapshots < <(find "$cpu_nr/outdat" -maxdepth 1 -type f \
    -name 'flowfield[0-9][0-9][0-9][0-9].h5' | sort)
  mapfile -t gpu_snapshots < <(find "$gpu_nr/outdat" -maxdepth 1 -type f \
    -name 'flowfield[0-9][0-9][0-9][0-9].h5' | sort)
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "${cpu_snapshots[-1]}" --gpu "${gpu_snapshots[-1]}" \
    --report "$level_dir/final_field_compare.txt" \
    --atol "$FIELD_ATOL" --rtol "$FIELD_ATOL"

  level_summary="$level_dir/reflection_summary.json"
  if [[ "$RUN_CONTROL" == "t" ]]; then
    python3 - "$grid" "$cpu_nr/reflection/reflection_metrics.json" \
      "$gpu_nr/reflection/reflection_metrics.json" \
      "$cpu_control/reflection/reflection_metrics.json" "$level_summary" \
      "$REFLECTION_MAX" "$CONTROL_RATIO_MAX" "$REFLECTION_ATOL" <<'PY'
import json
from pathlib import Path
import sys

grid, cpu_path, gpu_path, control_path, output_path = sys.argv[1:6]
absolute_max, ratio_max, cpu_gpu_atol = map(float, sys.argv[6:9])
cpu = json.loads(Path(cpu_path).read_text())
gpu = json.loads(Path(gpu_path).read_text())
control = json.loads(Path(control_path).read_text())
r_cpu = float(cpu["reflection"])
r_gpu = float(gpu["reflection"])
r_control = float(control["reflection"])
if r_cpu > absolute_max or r_gpu > absolute_max:
    raise SystemExit(f"absolute reflection gate failed for {grid}: {r_cpu=}, {r_gpu=}")
if r_cpu > ratio_max*r_control or r_gpu > ratio_max*r_control:
    raise SystemExit(
        f"compatibility-control ratio gate failed for {grid}: "
        f"{r_cpu=}, {r_gpu=}, {r_control=}"
    )
if abs(r_cpu-r_gpu) > cpu_gpu_atol:
    raise SystemExit(f"CPU/GPU reflection gate failed for {grid}: {r_cpu=}, {r_gpu=}")
summary = {
    "grid": grid,
    "cpu_nonreflecting": r_cpu,
    "gpu_nonreflecting": r_gpu,
    "cpu_compatibility": r_control,
    "cpu_gpu_abs_difference": abs(r_cpu-r_gpu),
}
Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
  else
    python3 - "$grid" "$cpu_nr/reflection/reflection_metrics.json" \
      "$gpu_nr/reflection/reflection_metrics.json" "$level_summary" \
      "$REFLECTION_ATOL" <<'PY'
import json
from pathlib import Path
import sys

grid, cpu_path, gpu_path, output_path, atol = sys.argv[1:]
cpu = json.loads(Path(cpu_path).read_text())
gpu = json.loads(Path(gpu_path).read_text())
r_cpu = float(cpu["reflection"])
r_gpu = float(gpu["reflection"])
difference = abs(r_cpu-r_gpu)
if difference > float(atol):
    raise SystemExit(
        f"CPU/GPU reflection gate failed for {grid}: {difference} > {atol}"
    )
summary = {
    "grid": grid,
    "cpu_nonreflecting": r_cpu,
    "gpu_nonreflecting": r_gpu,
    "cpu_gpu_abs_difference": difference,
}
Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY
  fi
  summary_files+=("$level_summary")
done

python3 - "$OUT_DIR/refinement_summary.json" "${summary_files[@]}" <<'PY'
import json
from pathlib import Path
import sys

output = Path(sys.argv[1])
rows = [json.loads(Path(path).read_text()) for path in sys.argv[2:]]
for key in ("cpu_nonreflecting", "gpu_nonreflecting"):
    values = [float(row[key]) for row in rows]
    if any(current > previous + 1.0e-12 for previous, current in zip(values, values[1:])):
        raise SystemExit(f"reflection worsens under refinement for {key}: {values}")
output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
print("acoustic_refinement: PASS")
PY
