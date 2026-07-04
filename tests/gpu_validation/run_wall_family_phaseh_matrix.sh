#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/wall_family_phaseh_matrix}"
MATRIX="${MATRIX:-wall41:x:2:2,1,1 wall41:y:2:1,2,1 wall41:z:2:1,1,2 adiabaticwall:x:2:2,1,1 adiabaticwall:y:2:1,2,1 slipisotwall:y:2:1,2,1 slipisotwall:y:2:2,1,1 slipisotwall:y:2:1,1,2 slipadibwall:y:2:1,2,1 slipadibwall:y:2:2,1,1 slipadibwall:y:2:1,1,2}"
REJECT_MATRIX="${REJECT_MATRIX:-adiabaticwall:z slipisotwall:x slipisotwall:z slipadibwall:x slipadibwall:z}"
RUN_SUPPORTED="${RUN_SUPPORTED:-t}"
RUN_REJECTS="${RUN_REJECTS:-t}"
DRY_RUN="${DRY_RUN:-f}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT="${FEQCHKPT:-1}"
LFILTER="${LFILTER:-t}"
DIFFTERM="${DIFFTERM:-t}"
SCHEME="${SCHEME:-643e}"
GRID="${GRID:-32,32,32}"
DELTAT="${DELTAT:-5.d-4}"
STATS_ATOL="${STATS_ATOL:-1e-9}"
STATS_RTOL="${STATS_RTOL:-1e-10}"
FIELD_ATOL="${FIELD_ATOL:-1e-8}"
FIELD_RTOL="${FIELD_RTOL:-1e-10}"
RUN_FIELD="${RUN_FIELD:-t}"
WALL_TEMP="${WALL_TEMP:-273.15d0}"
XSLIP="${XSLIP:-3.141592653589793d0}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

topology_tag() {
  printf '%s' "$1" | tr ',' 'x'
}

bc_tag() {
  case "$1" in
    wall41) printf 'wall41' ;;
    adiabaticwall) printf 'wall42' ;;
    slipisotwall) printf 'wall411' ;;
    slipadibwall) printf 'wall421' ;;
    *)
      printf 'unsupported Phase H BC_KIND=%s; expected wall41, adiabaticwall, slipisotwall, or slipadibwall\n' "$1" >&2
      exit 2
      ;;
  esac
}

run_supported_case() {
  local entry="$1"
  local bc axis np topology tag label case_out

  IFS=':' read -r bc axis np topology <<< "$entry"
  if [[ -z "${bc:-}" || -z "${axis:-}" || -z "${np:-}" || -z "${topology:-}" ]]; then
    printf 'invalid Phase H matrix entry=%s; expected bc:axis:np:topology\n' "$entry" >&2
    exit 2
  fi

  tag="$(topology_tag "$topology")"
  label="$(bc_tag "$bc")"
  case_out="$OUT_DIR/${label}_${axis}_np${np}_${tag}"

  if [[ "$DRY_RUN" == "t" ]]; then
    printf 'dry-run supported bc=%s axis=%s np=%s topology=%s out=%s\n' \
      "$bc" "$axis" "$np" "$topology" "$case_out" >> "$SUMMARY"
    return
  fi

  if [[ "$bc" == "wall41" ]]; then
    WALL_AXIS="$axis" NP="$np" TOPOLOGY="$topology" \
      MAXSTEP="$MAXSTEP" FEQCHKPT="$FEQCHKPT" \
      LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" GRID="$GRID" DELTAT="$DELTAT" \
      WALL_TEMP="$WALL_TEMP" COMPARE_STATS=t COMPARE_FIELD="$RUN_FIELD" \
      STATS_ATOL="$STATS_ATOL" STATS_RTOL="$STATS_RTOL" FIELD_ATOL="$FIELD_ATOL" FIELD_RTOL="$FIELD_RTOL" \
      OUT_DIR="$case_out" "$ROOT_DIR/tests/gpu_validation/run_wall41_phased_compare.sh"
  else
    BC_KIND="$bc" ZERO_AXIS="$axis" NP="$np" TOPOLOGY="$topology" \
      MAXSTEP="$MAXSTEP" FEQCHKPT="$FEQCHKPT" \
      LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" SCHEME="$SCHEME" GRID="$GRID" DELTAT="$DELTAT" \
      WALL_TEMP="$WALL_TEMP" XSLIP="$XSLIP" COMPARE_STATS=t COMPARE_FIELD="$RUN_FIELD" \
      STATS_ATOL="$STATS_ATOL" STATS_RTOL="$STATS_RTOL" FIELD_ATOL="$FIELD_ATOL" FIELD_RTOL="$FIELD_RTOL" \
      OUT_DIR="$case_out" "$ROOT_DIR/tests/gpu_validation/run_xextrap_phaseb_compare.sh"
  fi

  printf 'pass supported bc=%s axis=%s np=%s topology=%s lfilter=%s diffterm=%s field=%s out=%s\n' \
    "$bc" "$axis" "$np" "$topology" "$LFILTER" "$DIFFTERM" "$RUN_FIELD" "$case_out" >> "$SUMMARY"
}

run_reject_case() {
  local entry="$1"
  local bc axis label case_out log status

  IFS=':' read -r bc axis <<< "$entry"
  if [[ -z "${bc:-}" || -z "${axis:-}" ]]; then
    printf 'invalid Phase H reject entry=%s; expected bc:axis\n' "$entry" >&2
    exit 2
  fi

  label="$(bc_tag "$bc")"
  case_out="$OUT_DIR/reject_${label}_${axis}"
  log="$case_out/reject.log"
  mkdir -p "$case_out"

  if [[ "$DRY_RUN" == "t" ]]; then
    printf 'dry-run reject bc=%s axis=%s out=%s\n' "$bc" "$axis" "$case_out" >> "$SUMMARY"
    return
  fi

  set +e
  BC_KIND="$bc" ZERO_AXIS="$axis" NP=1 TOPOLOGY=1,1,1 \
    MAXSTEP=1 FEQCHKPT=1 GRID="$GRID" LFILTER="$LFILTER" DIFFTERM="$DIFFTERM" \
    WALL_TEMP="$WALL_TEMP" XSLIP="$XSLIP" COMPARE_STATS=f COMPARE_FIELD=f \
    OUT_DIR="$case_out/run" "$ROOT_DIR/tests/gpu_validation/run_xextrap_phaseb_compare.sh" > "$log" 2>&1
  status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    printf 'unexpected pass for rejected Phase H case bc=%s axis=%s; see %s\n' "$bc" "$axis" "$log" >&2
    exit 1
  fi
  if [[ "$status" -ne 2 ]]; then
    printf 'unexpected exit status=%s for rejected Phase H case bc=%s axis=%s; expected 2; see %s\n' \
      "$status" "$bc" "$axis" "$log" >&2
    exit 1
  fi

  printf 'pass reject bc=%s axis=%s status=%s out=%s\n' "$bc" "$axis" "$status" "$case_out" >> "$SUMMARY"
}

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

if [[ "$RUN_SUPPORTED" == "t" ]]; then
  for entry in $MATRIX; do
    run_supported_case "$entry"
  done
fi

if [[ "$RUN_REJECTS" == "t" ]]; then
  for entry in $REJECT_MATRIX; do
    run_reject_case "$entry"
  done
fi
