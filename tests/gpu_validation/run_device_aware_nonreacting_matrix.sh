#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/device_aware_nonreacting_matrix}"
CPU_EXE="${CPU_EXE:-$ROOT_DIR/build_cpu_probe/bin/astr}"
GPU_EXE="${GPU_EXE:-$ROOT_DIR/build_gpu_probe/bin/astr}"
CASE_GROUPS="${CASE_GROUPS:-cartesian-boundary curve shock-sensor wall-family}"
GPU_HALO_TRANSPORT="${GPU_HALO_TRANSPORT:-device-aware}"
GPU_UCX_PROTO_INFO="${GPU_UCX_PROTO_INFO:-y}"
GPU_SYNC_MODE="${GPU_SYNC_MODE:-explicit}"
REQUIRE_CUDA_IPC="${REQUIRE_CUDA_IPC:-t}"
DRY_RUN="${DRY_RUN:-f}"
SUMMARY="$OUT_DIR/matrix_summary.txt"

[[ "$GPU_HALO_TRANSPORT" == "device-aware" ]] || {
  printf 'this admission matrix requires GPU_HALO_TRANSPORT=device-aware\n' >&2
  exit 2
}
[[ "$REQUIRE_CUDA_IPC" == "t" || "$REQUIRE_CUDA_IPC" == "f" ]] || {
  printf 'REQUIRE_CUDA_IPC must be t or f\n' >&2
  exit 2
}
[[ "$DRY_RUN" == "t" || "$DRY_RUN" == "f" ]] || {
  printf 'DRY_RUN must be t or f\n' >&2
  exit 2
}
[[ ! -e "$OUT_DIR" ]] || {
  printf 'refusing to overwrite: %s\n' "$OUT_DIR" >&2
  exit 2
}

for group in $CASE_GROUPS; do
  case "$group" in
    cartesian-boundary|curve|shock-sensor|wall-family) ;;
    *)
      printf 'unknown CASE_GROUPS entry: %s\n' "$group" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR"
: > "$SUMMARY"

if [[ "$DRY_RUN" != "t" ]]; then
  [[ -x "$CPU_EXE" ]] || { printf 'CPU executable is missing: %s\n' "$CPU_EXE" >&2; exit 2; }
  [[ -x "$GPU_EXE" ]] || { printf 'GPU executable is missing: %s\n' "$GPU_EXE" >&2; exit 2; }
fi

export CPU_EXE GPU_EXE GPU_HALO_TRANSPORT GPU_UCX_PROTO_INFO GPU_SYNC_MODE

record_dry_run() {
  local group="$1"
  local cases="$2"
  printf 'dry-run group=%s cases=%s backend=%s sync=%s\n' \
    "$group" "$cases" "$GPU_HALO_TRANSPORT" "$GPU_SYNC_MODE" | tee -a "$SUMMARY"
}

verify_group() {
  local group="$1"
  local expected_cases="$2"
  local expected_sensor_reports="$3"
  local group_dir="$OUT_DIR/$group"
  local logs=()
  local reports=()
  local sensor_reports=()
  local log report

  mapfile -t logs < <(find "$group_dir" -type f -path '*/gpu/gpu.log' -print | sort)
  [[ "${#logs[@]}" -eq "$expected_cases" ]] || {
    printf '%s expected %s GPU logs, found %s\n' "$group" "$expected_cases" "${#logs[@]}" >&2
    exit 1
  }
  for log in "${logs[@]}"; do
    grep -aq 'The job is done!' "$log"
    grep -aq 'ASTR_GPU_HALO_TRANSPORT_SELECTED=device-aware' "$log"
    if [[ "$REQUIRE_CUDA_IPC" == "t" ]]; then
      grep -aq 'cuda_ipc/cuda' "$log"
    fi
    if grep -aEq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "$log"; then
      printf 'invalid numerical state in %s\n' "$log" >&2
      exit 1
    fi
  done

  mapfile -t reports < <(find "$group_dir" -type f \
    \( -name flowstate_compare.txt -o -name flowfield_compare.txt \) -print | sort)
  [[ "${#reports[@]}" -eq $((2 * expected_cases)) ]] || {
    printf '%s expected %s field/statistics reports, found %s\n' \
      "$group" "$((2 * expected_cases))" "${#reports[@]}" >&2
    exit 1
  }
  for report in "${reports[@]}"; do
    grep -qx 'status: pass' "$report"
  done

  mapfile -t sensor_reports < <(find "$group_dir" -type f -name shock_sensor_compare.txt -print | sort)
  [[ "${#sensor_reports[@]}" -eq "$expected_sensor_reports" ]] || {
    printf '%s expected %s shock-sensor reports, found %s\n' \
      "$group" "$expected_sensor_reports" "${#sensor_reports[@]}" >&2
    exit 1
  }
  for report in "${sensor_reports[@]}"; do
    grep -qx 'status: pass' "$report"
  done

  printf 'pass group=%s cases=%s backend=device-aware cuda_ipc_required=%s\n' \
    "$group" "$expected_cases" "$REQUIRE_CUDA_IPC" | tee -a "$SUMMARY"
}

run_cartesian_boundary() {
  local group=cartesian-boundary
  if [[ "$DRY_RUN" == "t" ]]; then
    record_dry_run "$group" 3
    return
  fi
  OUT_DIR="$OUT_DIR/$group" \
  MATRIX='x:2:2,1,1 y:2:1,2,1 z:2:1,1,2' \
  MAXSTEP=1 FEQCHKPT=1 GRID=64,64,64 LFILTER=t DIFFTERM=t RUN_FIELD=t \
    "$ROOT_DIR/tests/gpu_validation/run_zeroextrap_phaseb_mpirank_matrix.sh"
  verify_group "$group" 3 0
}

run_curve() {
  local group=curve
  local axis topology
  if [[ "$DRY_RUN" == "t" ]]; then
    record_dry_run "$group" 3
    return
  fi
  for axis in x y z; do
    case "$axis" in
      x) topology=2,1,1 ;;
      y) topology=1,2,1 ;;
      z) topology=1,1,2 ;;
    esac
    OUT_DIR="$OUT_DIR/$group/${axis}_wall41_np2" \
    WALL_AXIS="$axis" NP=2 TOPOLOGY="$topology" MAXSTEP=2 FEQCHKPT=2 \
    GRID=32,32,32 LFILTER=t DIFFTERM=t \
      "$ROOT_DIR/tests/gpu_validation/run_curvilinear_wall41_compare.sh"
  done
  verify_group "$group" 3 0
}

run_shock_sensor() {
  local group=shock-sensor
  local axis topology km
  if [[ "$DRY_RUN" == "t" ]]; then
    record_dry_run "$group" 3
    return
  fi
  for axis in x y z; do
    case "$axis" in
      x) topology=2,1,1; km=8 ;;
      y) topology=1,2,1; km=8 ;;
      z) topology=1,1,2; km=16 ;;
    esac
    OUT_DIR="$OUT_DIR/$group/${axis}_slab_np2" \
    NP=2 TOPOLOGY="$topology" IM=192 JM=192 KM="$km" \
    MAXSTEP=1 FEQCHKPT=1 COMPARE_SENSOR=t SAME_PHASE_FIELD=t \
      "$ROOT_DIR/tests/gpu_validation/run_s2_hbl_selective_roe_s2c3_compare.sh"
  done
  verify_group "$group" 3 3
}

run_wall_family() {
  local group=wall-family
  if [[ "$DRY_RUN" == "t" ]]; then
    record_dry_run "$group" 11
    return
  fi
  OUT_DIR="$OUT_DIR/$group" RUN_SUPPORTED=t RUN_REJECTS=f \
  MAXSTEP=1 FEQCHKPT=1 GRID=32,32,32 LFILTER=t DIFFTERM=t RUN_FIELD=t \
    "$ROOT_DIR/tests/gpu_validation/run_wall_family_phaseh_matrix.sh"
  verify_group "$group" 11 0
}

for group in $CASE_GROUPS; do
  case "$group" in
    cartesian-boundary) run_cartesian_boundary ;;
    curve) run_curve ;;
    shock-sensor) run_shock_sensor ;;
    wall-family) run_wall_family ;;
  esac
done

if [[ "$DRY_RUN" == "t" ]]; then
  printf 'NONREACTING_DEVICE_AWARE_MATRIX=DRY_RUN\n' | tee -a "$SUMMARY"
else
  printf 'NONREACTING_DEVICE_AWARE_MATRIX=PASS\n' | tee -a "$SUMMARY"
fi
