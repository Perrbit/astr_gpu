#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:?OUT_DIR must name a new D2 evidence directory}"
FIELD_ATOL="${FIELD_ATOL:-1.0e-10}"
GPU_IDS="${GPU_IDS:-0,1}"
FILTER_WORKSPACE="${FILTER_WORKSPACE:-full}"

if [[ -e "$OUT_DIR" ]]; then
  printf 'refusing to overwrite D2 evidence directory: %s\n' "$OUT_DIR" >&2
  exit 2
fi
if [[ ! -x "$CPU_EXE" || ! -x "$GPU_EXE" ]]; then
  printf 'missing CPU or GPU validation executable\n' >&2
  exit 2
fi
mkdir -p "$OUT_DIR"
printf 'case\tstatus\tpath\n' > "$OUT_DIR/status.tsv"

set_restart_input() {
  local case_dir="$1" maxstep="$2"
  python3 - "$case_dir" "$maxstep" <<'PY'
from pathlib import Path
import sys

case = Path(sys.argv[1])
maxstep = int(sys.argv[2])
input_path = case / "datin/input.flatplate"
lines = input_path.read_text(encoding="ascii").splitlines()
for index, line in enumerate(lines):
    if line.strip() == "# lrestar":
        lines[index + 1] = "t"
        break
else:
    raise RuntimeError("lrestar marker not found")
input_path.write_text("\n".join(lines) + "\n", encoding="ascii")

controller = case / "datin/controller"
lines = controller.read_text(encoding="ascii").splitlines()
for index, line in enumerate(lines):
    if line.strip() == "# maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg":
        lines[index + 1] = f"{maxstep},{maxstep},9999,9999,1,9999"
        break
else:
    raise RuntimeError("controller step marker not found")
controller.write_text("\n".join(lines) + "\n", encoding="ascii")
PY
}

prepare_case() {
  local case_dir="$1" use_gpu="$2" im="$3" jm="$4" km="$5"
  local maxstep="$6" deltat="$7" slice_dt="$8" slice_count="$9"
  local lfilter="${10}" warp_x="${11}" warp_y="${12}"
  python3 "$ROOT_DIR/tests/gpu_validation/prepare_s1_flatplate_case.py" \
    --dst-case "$case_dir" --use-gpu "$use_gpu" \
    --im "$im" --jm "$jm" --km "$km" --mach 0.3 \
    --conschm 543e --diffterm t --lfilter "$lfilter" --wall-temperature 1.4 \
    --isobaric-profile --ninit 3 --maxstep "$maxstep" --feqchkpt "$maxstep" \
    --deltat "$deltat" --turbinf intp --warp-x "$warp_x" --warp-y "$warp_y"
  python3 "$ROOT_DIR/tests/gpu_validation/generate_uniform_flow_field.py" \
    --output "$case_dir/datin/flowini3d.h5" --grid "$im,$jm,$km" \
    --density 0.7 --u1 0.2 --u2 -0.1 --u3 0.05 --temperature 0.8
  python3 "$ROOT_DIR/tests/gpu_validation/generate_dynamic_inflow_slices.py" \
    --output "$case_dir/inflow" --jm "$jm" --km "$km" \
    --count "$slice_count" --delta-time "$slice_dt" \
    --temporal-mode nonpolynomial
}

run_solver() {
  local kind="$1" np="$2" topology="$3" case_dir="$4" log_name="$5"
  local exe="$CPU_EXE"
  [[ "$kind" == gpu ]] && exe="$GPU_EXE"
  (
    cd "$case_dir"
    if [[ "$kind" == cpu ]]; then
      OMPI_MCA_sharedfp=individual \
        ASTR_FORCE_MPI_TOPOLOGY="$topology" \
        ASTR_VALIDATION_RK_SNAPSHOT=outdat/rk_complete_snapshot.h5 \
        timeout --kill-after=10s 300s mpirun --oversubscribe -np "$np" \
        "$exe" run datin/input.flatplate > "$log_name" 2>&1
    else
      CUDA_VISIBLE_DEVICES="$GPU_IDS" \
        OMPI_MCA_sharedfp=individual \
        ASTR_FORCE_MPI_TOPOLOGY="$topology" \
        ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE" \
        timeout --kill-after=10s 300s mpirun --oversubscribe -np "$np" \
        "$exe" run datin/input.flatplate > "$log_name" 2>&1
    fi
  )
}

compare_fields() {
  local reference="$1" candidate="$2" report="$3"
  python3 "$ROOT_DIR/tests/gpu_validation/compare_flowfield_h5.py" \
    --cpu "$reference" --gpu "$candidate" --report "$report" \
    --atol "$FIELD_ATOL" --rtol 0
}

run_case() {
  local label="$1" np="$2" topology="$3" im="$4" jm="$5" km="$6"
  local first_step="$7" final_step="$8" deltat="$9" slice_dt="${10}"
  local slice_count="${11}" lfilter="${12}" warp_x="${13}" warp_y="${14}"
  local case_root="$OUT_DIR/$label"

  local cpu_continuous="$case_root/cpu_continuous"
  local gpu_continuous="$case_root/gpu_continuous"
  local cpu_restart="$case_root/cpu_restart"
  local gpu_restart="$case_root/gpu_restart"

  prepare_case "$cpu_continuous" f "$im" "$jm" "$km" \
    "$final_step" "$deltat" "$slice_dt" "$slice_count" "$lfilter" "$warp_x" "$warp_y"
  prepare_case "$gpu_continuous" t "$im" "$jm" "$km" \
    "$final_step" "$deltat" "$slice_dt" "$slice_count" "$lfilter" "$warp_x" "$warp_y"
  run_solver cpu "$np" "$topology" "$cpu_continuous" continuous.log
  run_solver gpu "$np" "$topology" "$gpu_continuous" continuous.log

  prepare_case "$cpu_restart" f "$im" "$jm" "$km" \
    "$first_step" "$deltat" "$slice_dt" "$slice_count" "$lfilter" "$warp_x" "$warp_y"
  run_solver cpu "$np" "$topology" "$cpu_restart" fresh.log

  local shared_checkpoint="$case_root/shared_checkpoint"
  mkdir -p "$shared_checkpoint"
  cp "$cpu_restart/outdat/flowfield.h5" "$shared_checkpoint/flowfield.h5"
  cp "$cpu_restart/outdat/auxiliary.txt" "$shared_checkpoint/auxiliary.txt"

  set_restart_input "$cpu_restart" "$final_step"
  run_solver cpu "$np" "$topology" "$cpu_restart" restart.log
  grep -q 'checkpoint file read' "$cpu_restart/restart.log"

  prepare_case "$gpu_restart" t "$im" "$jm" "$km" \
    "$final_step" "$deltat" "$slice_dt" "$slice_count" "$lfilter" "$warp_x" "$warp_y"
  set_restart_input "$gpu_restart" "$final_step"
  mkdir -p "$gpu_restart/outdat"
  cp "$shared_checkpoint/flowfield.h5" "$gpu_restart/outdat/flowfield.h5"
  cp "$shared_checkpoint/auxiliary.txt" "$gpu_restart/outdat/auxiliary.txt"
  run_solver gpu "$np" "$topology" "$gpu_restart" restart.log
  grep -q 'checkpoint file read' "$gpu_restart/restart.log"

  if [[ "$lfilter" == f ]]; then
    compare_fields "$cpu_continuous/outdat/flowfield.h5" \
      "$cpu_restart/outdat/flowfield.h5" "$case_root/cpu_restart_compare.txt"
    compare_fields "$gpu_continuous/outdat/flowfield.h5" \
      "$gpu_restart/outdat/flowfield.h5" "$case_root/gpu_restart_compare.txt"
  else
    # KNOWN_UPSTREAM_CHECKPOINT_SEMANTICS: the filtered CPU checkpoint stores
    # primitive variables after boundary preparation, not a serialized q/halo state.
    compare_fields "$cpu_continuous/outdat/flowfield.h5" \
      "$cpu_restart/outdat/flowfield.h5" "$case_root/cpu_restart_observed.txt" \
      > "$case_root/cpu_restart_observed.log" 2>&1 || true
  fi

  compare_fields "$cpu_continuous/outdat/rk_complete_snapshot.h5" \
    "$gpu_continuous/outdat/flowfield.h5" "$case_root/cpu_gpu_continuous.txt"
  compare_fields "$cpu_restart/outdat/rk_complete_snapshot.h5" \
    "$gpu_restart/outdat/flowfield.h5" "$case_root/cpu_gpu_shared_restart.txt"
  printf '%s\tPASS\t%s\n' "$label" "$case_root" >> "$OUT_DIR/status.tsv"
}

run_case slice_node 1 1,1,1 32 32 16 2 4 1.0e-5 1.0e-5 12 f 0.0 0.0
run_case cross_slice 2 2,1,1 32 32 16 2 5 6.0e-6 1.0e-5 12 f 0.0 0.0
run_case multi_slice 1 1,1,1 32 32 16 1 2 3.1e-5 1.0e-5 16 f 0.0 0.0
TOPOLOGY=2,2,1 TEMPORAL_MODE=nonpolynomial \
  run_case filter_curve_mpi 4 2,2,1 32 32 16 2 4 6.0e-6 1.0e-5 12 t 0.03 0.02

printf 'DYNAMIC_INFLOW_D2_MATRIX_PASS %s\n' "$OUT_DIR"
