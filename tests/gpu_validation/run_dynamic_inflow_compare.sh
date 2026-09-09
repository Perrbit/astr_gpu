#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/dynamic_inflow}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
IM="${IM:-32}"
JM="${JM:-32}"
KM="${KM:-8}"
MAXSTEP="${MAXSTEP:-3}"
DELTAT="${DELTAT:-1.0e-5}"
SLICE_COUNT="${SLICE_COUNT:-8}"
SLICE_DT="${SLICE_DT:-1.0e-5}"
CONS_SCHM="${CONS_SCHM:-}"
CPU_SNAPSHOT="outdat/rk_complete_snapshot.h5"

if [[ -z "$CONS_SCHM" ]]; then
  if [[ "$NP" -eq 1 ]]; then
    CONS_SCHM=643e
  else
    CONS_SCHM=543e
  fi
fi
IFS=',' read -r TOPO_I TOPO_J TOPO_K <<< "$TOPOLOGY"
if [[ $((TOPO_I * TOPO_J * TOPO_K)) -ne "$NP" ]]; then
  printf 'topology %s does not match NP=%s\n' "$TOPOLOGY" "$NP" >&2
  exit 2
fi
if (( IM % TOPO_I != 0 || JM % TOPO_J != 0 || KM % TOPO_K != 0 )); then
  printf 'grid %s,%s,%s is not divisible by topology %s\n' "$IM" "$JM" "$KM" "$TOPOLOGY" >&2
  exit 2
fi
if (( IM / TOPO_I < 5 || JM / TOPO_J < 5 || KM / TOPO_K < 5 )); then
  printf 'local active extent must be at least hm=5\n' >&2
  exit 2
fi

if [[ -e "$OUT_DIR" ]]; then
  printf 'output directory already exists: %s\n' "$OUT_DIR" >&2
  exit 2
fi

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$OUT_DIR/$target" --use-gpu "$use_gpu" \
    --im "$IM" --jm "$JM" --km "$KM" --mach 0.3 \
    --conschm "$CONS_SCHM" --diffterm t --lfilter f --wall-temperature 1.4 \
    --isobaric-profile --ninit 3 --maxstep "$MAXSTEP" --feqchkpt "$MAXSTEP" \
    --deltat "$DELTAT" --turbinf intp
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$OUT_DIR/$target/datin/flowini3d.h5" --grid "$IM,$JM,$KM" \
    --density 0.7 --u1 0.2 --u2 -0.1 --u3 0.05 --temperature 0.8
  python3 "$ROOT_DIR/tests/gpu_validation/generate_dynamic_inflow_slices.py" \
    --output "$OUT_DIR/$target/inflow" --jm "$JM" --km "$KM" \
    --count "$SLICE_COUNT" --delta-time "$SLICE_DT"
}

prepare_case cpu f
prepare_case gpu t

(
  cd "$OUT_DIR/cpu"
  OMPI_MCA_sharedfp=individual ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_VALIDATION_RK_SNAPSHOT="$CPU_SNAPSHOT" \
    mpirun -np "$NP" "$CPU_EXE" run datin/input.flatplate > cpu.log 2>&1
)
(
  cd "$OUT_DIR/gpu"
  OMPI_MCA_sharedfp=individual ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.flatplate > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_flowstate.py" \
  --cpu "$OUT_DIR/cpu" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowstate_compare.txt" --atol 1e-10 --rtol 1e-10
python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
  --cpu "$OUT_DIR/cpu/$CPU_SNAPSHOT" --gpu "$OUT_DIR/gpu" \
  --report "$OUT_DIR/flowfield_compare.txt" \
  --atol 1e-10 --rtol 1e-10

for slice in 00000 00001 00002 00003; do
  grep -q "islice${slice}.h5" "$OUT_DIR/cpu/cpu.log"
  grep -q "islice${slice}.h5" "$OUT_DIR/gpu/gpu.log"
done
if awk -v steps="$MAXSTEP" -v dt="$DELTAT" -v slice_dt="$SLICE_DT" \
  'BEGIN { exit ! (steps * dt > 2.0 * slice_dt) }'; then
  grep -q 'islice00004.h5' "$OUT_DIR/cpu/cpu.log"
  grep -q 'islice00004.h5' "$OUT_DIR/gpu/gpu.log"
fi
printf 'dynamic inflow NP=%s topology=%s comparison passed: %s\n' \
  "$NP" "$TOPOLOGY" "$OUT_DIR"
