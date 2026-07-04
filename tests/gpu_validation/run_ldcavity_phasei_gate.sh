#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CASE_DIR="$ROOT_DIR/examples/Lid-Driven-Cavity"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/ldcavity_phasei_gate}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-1}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-}"
DELTAT="${DELTAT:-}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
RUN_CPU="${RUN_CPU:-f}"
EXPECT_PATTERN="${EXPECT_PATTERN:-GPU first-stage supports 3D cases only|GPU LDC pending: filter path for multi-axis physical boundary not implemented|GPU first-stage supports periodic or one zeroextrap/symmetry/isothermal/adiabatic/slip-isothermal/slip-adiabatic wall direction only}"
SUMMARY="$OUT_DIR/gate_summary.txt"

prepare_case() {
  local target="$1"
  local use_gpu="$2"
  local args=(
    "$ROOT_DIR/tests/gpu_validation/prepare_tgv_case.py"
    --src-case "$CASE_DIR"
    --dst-case "$OUT_DIR/$target"
    --input-name input.ldcav2d
    --use-gpu "$use_gpu"
    --maxstep "$MAXSTEP"
    --feqchkpt "$FEQCHKPT"
    --lfilter "$LFILTER"
    --diffterm "$DIFFTERM"
    --scheme "$SCHEME"
  )
  if [[ -n "$GRID" ]]; then
    args+=(--grid "$GRID")
  fi
  if [[ -n "$DELTAT" ]]; then
    args+=(--deltat "$DELTAT")
  fi
  python3 "${args[@]}"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

if [[ "$RUN_CPU" == "t" ]]; then
  prepare_case cpu f
  (
    cd "$OUT_DIR/cpu"
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
      mpirun -np "$NP" "$CPU_EXE" run datin/input.ldcav2d > cpu.log 2>&1
  )
  printf 'pass cpu np=%s topology=%s out=%s\n' "$NP" "$TOPOLOGY" "$OUT_DIR/cpu" >> "$SUMMARY"
fi

prepare_case gpu t

set +e
(
  cd "$OUT_DIR/gpu"
  ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    mpirun -np "$NP" "$GPU_EXE" run datin/input.ldcav2d > gpu.log 2>&1
)
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  printf 'unexpected LDC Phase I GPU pass; expected capability-gate rejection. See %s\n' \
    "$OUT_DIR/gpu/gpu.log" >&2
  exit 1
fi

if ! grep -Eq "$EXPECT_PATTERN" "$OUT_DIR/gpu/gpu.log"; then
  printf 'LDC Phase I GPU rejected with unexpected reason; expected pattern: %s. See %s\n' \
    "$EXPECT_PATTERN" "$OUT_DIR/gpu/gpu.log" >&2
  exit 1
fi

printf 'pass gpu-reject status=%s pattern=%s out=%s\n' \
  "$status" "$EXPECT_PATTERN" "$OUT_DIR/gpu" >> "$SUMMARY"
