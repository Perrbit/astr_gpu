#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/tests/gpu_validation/out/c4_cuda_build_4}"
EXE="${EXE:-$BUILD_DIR/bin/astr}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/air5_c4_transport_compare}"
TMP_DIR="${TMPDIR:-$ROOT_DIR/tests/gpu_validation/out/tmp_nvfortran}"
GRID="${GRID:-15,15,15}"
MAXSTEP="${MAXSTEP:-0}"
DELTAT="${DELTAT:-1.d-7}"
DIFFTERM="${DIFFTERM:-t}"
MPI_NP="${MPI_NP:-${NP:-1}}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
ATOL="${ATOL:-1e-10}"
RTOL="${RTOL:-1e-10}"
CONSERVATION="${CONSERVATION:-t}"
CONSERVATION_ATOL="${CONSERVATION_ATOL:-1e-12}"
CONSERVATION_RTOL="${CONSERVATION_RTOL:-1e-11}"
INITIAL_CONDITION="${INITIAL_CONDITION:-tgv}"
SPECIES_MINIMUM_RELATIVE_DROP="${SPECIES_MINIMUM_RELATIVE_DROP:-1e-10}"

mkdir -p "$OUT_DIR" "$TMP_DIR"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
  --destination "$OUT_DIR/cpu" \
  --grid "$GRID" \
  --maxstep "$MAXSTEP" \
  --deltat "$DELTAT" \
  --diffterm "$DIFFTERM" \
  --use-gpu f \
  --initial-condition "$INITIAL_CONDITION"

python3 "$ROOT_DIR/tests/gpu_validation/prepare_air5_c4_case.py" \
  --destination "$OUT_DIR/gpu" \
  --grid "$GRID" \
  --maxstep "$MAXSTEP" \
  --deltat "$DELTAT" \
  --diffterm "$DIFFTERM" \
  --use-gpu t \
  --initial-condition "$INITIAL_CONDITION"

mkdir -p "$OUT_DIR/cpu/validation" "$OUT_DIR/gpu/validation"

(
  cd "$OUT_DIR/cpu"
  TMPDIR="$TMP_DIR" \
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
    mpirun -np "$MPI_NP" "$EXE" run datin/input.air5_c4 > cpu.log 2>&1
)

(
  cd "$OUT_DIR/gpu"
  TMPDIR="$TMP_DIR" \
    ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" \
    ASTR_VALIDATION_RHS_PREFIX=validation/air5 \
    ASTR_AIR5_C4_CONSERVATION="$CONSERVATION" \
    mpirun -np "$MPI_NP" "$EXE" run datin/input.air5_c4 > gpu.log 2>&1
)

python3 "$ROOT_DIR/tests/gpu_validation/compare_q_validation_snapshots.py" \
  --cpu-prefix "$OUT_DIR/cpu/validation/air5" \
  --gpu-prefix "$OUT_DIR/gpu/validation/air5" \
  --report "$OUT_DIR/q_compare.txt" \
  --labels pre_rhs,post_update \
  --atol "$ATOL" \
  --rtol "$RTOL"

if [[ "$CONSERVATION" == "t" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c4_conservation.py" \
    --input "$OUT_DIR/gpu/air5_c4_conservation.dat" \
    --report "$OUT_DIR/conservation_compare.txt" \
    --atol "$CONSERVATION_ATOL" \
    --rtol "$CONSERVATION_RTOL"
fi

if [[ "$INITIAL_CONDITION" == "species-wave" ]]; then
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c4_species_variance.py" \
    --prefix "$OUT_DIR/cpu/validation/air5" \
    --report "$OUT_DIR/cpu_species_variance.txt" \
    --minimum-relative-drop "$SPECIES_MINIMUM_RELATIVE_DROP"
  python3 "$ROOT_DIR/tests/gpu_validation/check_air5_c4_species_variance.py" \
    --prefix "$OUT_DIR/gpu/validation/air5" \
    --report "$OUT_DIR/gpu_species_variance.txt" \
    --minimum-relative-drop "$SPECIES_MINIMUM_RELATIVE_DROP"
fi
