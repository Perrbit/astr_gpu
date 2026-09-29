#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CASE="${1:?usage: bash run_case.sh CASE}"
if (( $# != 1 )); then
  echo "Use environment variables for options; see README.md." >&2
  exit 2
fi
case "$CASE" in
  tgv|channel|flatplate|air5_tgv|air5_flatplate) ;;
  *) echo "Unsupported case: $CASE (AIR5 channel is not implemented)." >&2; exit 2 ;;
esac
MODE="${MODE:-gpu}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
FILTER_WORKSPACE="${FILTER_WORKSPACE:-default}"
HALO_TRANSPORT="${HALO_TRANSPORT:-pinned}"
EXE="${EXE:-$ROOT/build_prod/bin/astr}"
MPIEXEC="${MPIEXEC:-mpirun}"
PYTHON="${PYTHON:-python3}"
RUN_DIR="${RUN_DIR:-$HERE/runs/${CASE}_${MODE}_$(date +%Y%m%d_%H%M%S)_$$}"

case "$MODE" in cpu|gpu) ;; *) echo "MODE must be cpu or gpu" >&2; exit 2 ;; esac
case "$FILTER_WORKSPACE" in default|scalar|full) ;; *) echo "Invalid FILTER_WORKSPACE" >&2; exit 2 ;; esac
case "$HALO_TRANSPORT" in
  pageable|pinned|device-aware) ;;
  *) echo "Quickstart supports pageable, pinned or device-aware transport." >&2; exit 2 ;;
esac
if [[ ! -x "$EXE" ]]; then echo "Executable not found: $EXE" >&2; exit 2; fi
EXE="$(realpath "$EXE")"
command -v "$MPIEXEC" >/dev/null
command -v "$PYTHON" >/dev/null
libraries="$(ldd "$EXE" 2>&1)"
if [[ "$libraries" == *"not found"* ]]; then
  printf '%s\n' "$libraries" >&2
  exit 2
fi

# Each example defines its own physics. Do not inherit another case's ASTR options.
while IFS= read -r name; do
  if [[ "$name" == ASTR_* ]]; then unset "$name"; fi
done < <(compgen -e)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY"
if [[ "$MODE" == gpu ]]; then
  export ASTR_GPU_PRECISION_MODE=fp64
  export ASTR_GPU_SYNC_MODE=explicit
  export ASTR_GPU_HALO_TRANSPORT="$HALO_TRANSPORT"
  if [[ "$FILTER_WORKSPACE" != default ]]; then
    export ASTR_GPU_FILTER_WORKSPACE="$FILTER_WORKSPACE"
  fi
fi
case "$CASE" in
  channel)
    export ASTR_CHANNEL_FORCE_MODE=fixed
    export ASTR_CHANNEL_FORCE_FIXED=1.d-4
    ;;
  flatplate)
    export ASTR_PROFILE_INFLOW_MODE=complete_state
    ;;
  air5_tgv|air5_flatplate)
    export ASTR_AIR5_SOURCE_MODE=coupled
    export ASTR_AIR5_CONVECTION_LIMITER=symmetric_species
    export ASTR_AIR5_DIFFUSION_LIMITER=layered
    export ASTR_AIR5_PRIMITIVE_REUSE=off
    export ASTR_AIR5_CHEMISTRY_REDUCTIONS=baseline
    if [[ "$CASE" == air5_flatplate ]]; then
      export ASTR_AIR5_COMPENSATION=on
      export ASTR_AIR5_FILTER_VALIDATION=on
      export ASTR_AIR5_TOP_MODE=prescribed
      export ASTR_AIR5_FLOW_MONITOR_STRIDE=1
    fi
    ;;
esac

args=("$CASE" --destination "$RUN_DIR" --mode "$MODE" --np "$NP" --topology "$TOPOLOGY")
if [[ -n "${STEPS:-}" ]]; then args+=(--steps "$STEPS"); fi
if [[ -n "${DELTAT:-}" ]]; then args+=(--deltat "$DELTAT"); fi
"$PYTHON" "$HERE/prepare.py" "${args[@]}"
cd "$RUN_DIR"
printf '%s\n' "$libraries" > libraries.txt
env | LC_ALL=C sort | grep -E '^(ASTR_|OMP_NUM_THREADS=|CUDA_VISIBLE_DEVICES=)' > runtime.env
printf 'EXE=%q\nMPIEXEC=%q\nNP=%q\n' "$EXE" "$MPIEXEC" "$NP" > launch.txt
echo "Running $CASE ($MODE, NP=$NP, topology=$TOPOLOGY) in $PWD"
set +e
"$MPIEXEC" -np "$NP" "$EXE" run datin/input.dat > run.log 2>&1
status=$?
set -e
if (( status != 0 )); then
  tail -60 run.log >&2
  echo "ASTR failed with exit status $status; files retained at $PWD" >&2
  exit "$status"
fi
if grep -Eiq 'COMPUTATION CRASHED|error stop|unsupported configuration|(^|[[:space:]])nan([[:space:]]|$)' run.log; then
  tail -60 run.log >&2
  echo "ASTR reported a failure despite a zero exit status." >&2
  exit 1
fi
if ! grep -Fq 'The job is done!' run.log; then
  tail -60 run.log >&2
  echo "No normal completion marker; do not treat this run as successful." >&2
  exit 1
fi
echo "Run finished. Inspect $PWD/run.log and output fields before drawing physical conclusions."
