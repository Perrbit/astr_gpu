#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CANDIDATE_ID="${CANDIDATE_ID:-}"
BASELINE_REF="${BASELINE_REF:-}"
BASELINE_TIMINGS="${BASELINE_TIMINGS:-}"
TARGET_KERNELS="${TARGET_KERNELS:-}"
ALLOWED_PATHS="${ALLOWED_PATHS:-}"
HYPOTHESIS="${HYPOTHESIS:-}"
GATE_SET="${GATE_SET:-correctness}"
DRY_RUN="${DRY_RUN:-f}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/tests/gpu_validation/out/gpu_candidate_${CANDIDATE_ID:-unset}}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-$ROOT_DIR/build_cpu_probe}"
GPU_BUILD_DIR="${GPU_BUILD_DIR:-$ROOT_DIR/build_gpu_probe}"
CPU_EXE="${CPU_EXE:-$CPU_BUILD_DIR/bin/astr}"
GPU_EXE="${GPU_EXE:-$GPU_BUILD_DIR/bin/astr}"
BUILD_JOBS="${BUILD_JOBS:-8}"
GPU_ID="${GPU_ID:-0}"
SYNC_MODE="${SYNC_MODE:-explicit}"
CORRECTNESS_MAXSTEP="${CORRECTNESS_MAXSTEP:-10}"
PERFORMANCE_GRID="${PERFORMANCE_GRID:-256,256,256}"
PERFORMANCE_MAXSTEP="${PERFORMANCE_MAXSTEP:-10}"
PERFORMANCE_REPEATS="${PERFORMANCE_REPEATS:-5}"
MIN_TIME_REDUCTION_PERCENT="${MIN_TIME_REDUCTION_PERCENT:-3.0}"
MAX_SPREAD_PERCENT="${MAX_SPREAD_PERCENT:-5.0}"
MAX_MEMORY_GROWTH_PERCENT="${MAX_MEMORY_GROWTH_PERCENT:-5.0}"

STATUS_FILE="$OUT_DIR/gate_status.tsv"
MANIFEST_FILE="$OUT_DIR/candidate_manifest.tsv"
SUMMARY_FILE="$OUT_DIR/candidate_gate_summary.md"
CURRENT_STAGE="preflight"

die() {
  echo "error: $*" >&2
  exit 2
}

is_true() {
  [[ "$1" == "t" || "$1" == "true" || "$1" == "1" ]]
}

sanitize_tsv() {
  printf '%s' "$1" | tr '\t\r\n' '   '
}

format_command() {
  printf '%q ' "$@"
  printf '\n'
}

write_summary() {
  local overall="pass"
  if awk -F '\t' 'NR > 1 && $2 == "fail" {found=1} END {exit !found}' "$STATUS_FILE"; then
    overall="fail"
  elif is_true "$DRY_RUN"; then
    overall="planned"
  fi
  {
    printf '# ASTR GPU Optimization Candidate Gate\n\n'
    printf -- '- candidate: `%s`\n' "$CANDIDATE_ID"
    printf -- '- gate set: `%s`\n' "$GATE_SET"
    printf -- '- synchronization mode: `%s`\n' "$SYNC_MODE"
    printf -- '- overall status: `%s`\n\n' "$overall"
    printf '| Stage | Status | Exit code | Duration (s) | Log |\n'
    printf '|---|---|---:|---:|---|\n'
    awk -F '\t' 'NR > 1 {printf "| %s | %s | %s | %s | `%s` |\n", $1, $2, $3, $4, $5}' "$STATUS_FILE"
  } > "$SUMMARY_FILE"
}

on_exit() {
  local rc=$?
  if [[ -f "$STATUS_FILE" ]]; then
    write_summary
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "Candidate gate stopped at stage '$CURRENT_STAGE'; see $OUT_DIR" >&2
  fi
}
trap on_exit EXIT

run_stage() {
  local name="$1"
  shift
  local log="$OUT_DIR/${name}.log"
  local start end duration rc
  CURRENT_STAGE="$name"
  echo "==> $name"
  if is_true "$DRY_RUN"; then
    format_command "$@" | tee "$log"
    printf '%s\tplanned\t0\t0\t%s\n' "$name" "${log#$ROOT_DIR/}" >> "$STATUS_FILE"
    return 0
  fi
  start="$(date +%s)"
  set +e
  "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  end="$(date +%s)"
  duration=$((end - start))
  if [[ "$rc" -eq 0 ]]; then
    printf '%s\tpass\t0\t%s\t%s\n' "$name" "$duration" "${log#$ROOT_DIR/}" >> "$STATUS_FILE"
  else
    printf '%s\tfail\t%s\t%s\t%s\n' "$name" "$rc" "$duration" "${log#$ROOT_DIR/}" >> "$STATUS_FILE"
    return "$rc"
  fi
}

check_changed_paths() {
  local path allowed match
  local -a changed=()
  local -a allow=()
  IFS=',' read -r -a allow <<< "$ALLOWED_PATHS"
  mapfile -t changed < <(
    {
      git -C "$ROOT_DIR" diff --name-only "$BASELINE_REF" --
      git -C "$ROOT_DIR" ls-files --others --exclude-standard
    } | sort -u
  )
  for path in "${changed[@]}"; do
    match=f
    for allowed in "${allow[@]}"; do
      allowed="${allowed#./}"
      if [[ "$path" == "$allowed" || "$path" == "$allowed/"* ]]; then
        match=t
        break
      fi
    done
    [[ "$match" == t ]] || {
      echo "changed path is outside ALLOWED_PATHS: $path" >&2
      return 1
    }
  done
}

check_compile_flags() {
  local commands="$GPU_BUILD_DIR/compile_commands.json"
  [[ -f "$commands" ]] || {
    echo "compile_commands.json not found: $commands" >&2
    return 1
  }
  if rg -n -- '(--use_fast_math|-Mfprelaxed|(^|[[:space:]])-fast([[:space:]]|$))' "$commands"; then
    echo "prohibited relaxed-math compiler option found" >&2
    return 1
  fi
}

capture_environment() {
  local executable
  for executable in cmake mpif90 mpirun python3 rg nvidia-smi; do
    command -v "$executable" >/dev/null 2>&1 || {
      echo "required executable not found: $executable" >&2
      return 127
    }
  done
  {
    printf 'git_head: %s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
    printf 'hostname: %s\n' "$(hostname)"
    printf 'gpu_id: %s\n' "$GPU_ID"
    nvidia-smi -i "$GPU_ID" \
      --query-gpu=name,uuid,driver_version,compute_cap,pstate,power.limit \
      --format=csv,noheader
    printf '\ncmake:\n'
    cmake --version | head -1
    printf '\nmpif90:\n'
    mpif90 --version | head -2
    printf '\nmpirun:\n'
    mpirun --version | head -2
    if command -v nvfortran >/dev/null 2>&1; then
      printf '\nnvfortran:\n'
      nvfortran --version | head -2
    fi
    if command -v nsys >/dev/null 2>&1; then
      printf '\nnsys:\n'
      nsys --version | head -1
    fi
    if command -v ncu >/dev/null 2>&1; then
      printf '\nncu:\n'
      ncu --version | head -3
    fi
  } | tee "$OUT_DIR/environment.txt"
}

[[ -n "$CANDIDATE_ID" ]] || die "CANDIDATE_ID is required"
[[ "$CANDIDATE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "invalid CANDIDATE_ID"
[[ -n "$TARGET_KERNELS" ]] || die "TARGET_KERNELS is required"
[[ -n "$ALLOWED_PATHS" ]] || die "ALLOWED_PATHS is required"
[[ -n "$HYPOTHESIS" ]] || die "HYPOTHESIS is required"
[[ "$GATE_SET" == correctness || "$GATE_SET" == performance || "$GATE_SET" == full ]] || \
  die "GATE_SET must be correctness, performance, or full"
[[ "$SYNC_MODE" == explicit ]] || die "optimization acceptance requires SYNC_MODE=explicit"
[[ "$PERFORMANCE_REPEATS" -ge 5 ]] || die "PERFORMANCE_REPEATS must be at least 5"
if [[ "$GATE_SET" != correctness ]]; then
  [[ -n "$BASELINE_REF" ]] || die "BASELINE_REF is required for $GATE_SET"
  [[ -n "$BASELINE_TIMINGS" ]] || die "BASELINE_TIMINGS is required for $GATE_SET"
  [[ -f "$BASELINE_TIMINGS" ]] || die "baseline timings not found: $BASELINE_TIMINGS"
fi
if [[ -n "$BASELINE_REF" ]]; then
  git -C "$ROOT_DIR" rev-parse --verify "$BASELINE_REF^{commit}" >/dev/null 2>&1 || \
    die "BASELINE_REF is not a commit: $BASELINE_REF"
fi

mkdir -p "$OUT_DIR"
printf 'stage\tstatus\texit_code\tduration_seconds\tlog\n' > "$STATUS_FILE"
{
  printf 'key\tvalue\n'
  printf 'candidate_id\t%s\n' "$(sanitize_tsv "$CANDIDATE_ID")"
  printf 'baseline_ref\t%s\n' "$(sanitize_tsv "$BASELINE_REF")"
  printf 'baseline_timings\t%s\n' "$(sanitize_tsv "$BASELINE_TIMINGS")"
  printf 'target_kernels\t%s\n' "$(sanitize_tsv "$TARGET_KERNELS")"
  printf 'allowed_paths\t%s\n' "$(sanitize_tsv "$ALLOWED_PATHS")"
  printf 'hypothesis\t%s\n' "$(sanitize_tsv "$HYPOTHESIS")"
  printf 'gate_set\t%s\n' "$GATE_SET"
  printf 'sync_mode\t%s\n' "$SYNC_MODE"
  printf 'git_head\t%s\n' "$(git -C "$ROOT_DIR" rev-parse HEAD)"
  printf 'git_branch\t%s\n' "$(git -C "$ROOT_DIR" branch --show-current)"
  printf 'created_at\t%s\n' "$(date -Iseconds)"
  printf 'hostname\t%s\n' "$(hostname)"
} > "$MANIFEST_FILE"
git -C "$ROOT_DIR" status --short > "$OUT_DIR/git_status.txt"

run_stage environment capture_environment
if [[ -n "$BASELINE_REF" ]]; then
  run_stage scope_check check_changed_paths
fi
run_stage configure_cpu cmake -S "$ROOT_DIR" -B "$CPU_BUILD_DIR" \
  -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
run_stage build_cpu cmake --build "$CPU_BUILD_DIR" -j "$BUILD_JOBS"
run_stage configure_gpu cmake -S "$ROOT_DIR" -B "$GPU_BUILD_DIR" \
  -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DASTR_WITH_CUDA=ON
run_stage build_gpu cmake --build "$GPU_BUILD_DIR" -j "$BUILD_JOBS"
run_stage compile_flag_check check_compile_flags

run_stage np1_stats env \
  CPU_EXE="$CPU_EXE" GPU_EXE="$GPU_EXE" \
  OUT_DIR="$OUT_DIR/np1_stats" MAXSTEP="$CORRECTNESS_MAXSTEP" \
  FEQCHKPT=9999 LFILTER=t DIFFTERM=t SCHEME=643e \
  ATOL=1e-10 RTOL=1e-10 \
  "$ROOT_DIR/tests/gpu_validation/run_tgv_stats_compare.sh"
run_stage np1_field env \
  CPU_EXE="$CPU_EXE" GPU_EXE="$GPU_EXE" \
  OUT_DIR="$OUT_DIR/np1_field" MAXSTEP="$CORRECTNESS_MAXSTEP" \
  FEQCHKPT="$CORRECTNESS_MAXSTEP" LFILTER=t DIFFTERM=t SCHEME=643e \
  ATOL=1e-10 RTOL=1e-10 \
  "$ROOT_DIR/tests/gpu_validation/run_tgv_field_compare.sh"
run_stage np2_stats env \
  CPU_EXE="$CPU_EXE" GPU_EXE="$GPU_EXE" \
  OUT_DIR="$OUT_DIR/np2_stats" MAXSTEP="$CORRECTNESS_MAXSTEP" \
  FEQCHKPT=9999 LFILTER=t DIFFTERM=t SCHEME=643e \
  ATOL=1e-10 RTOL=1e-10 MPI_NP=2 TOPOLOGY=2,1,1 \
  "$ROOT_DIR/tests/gpu_validation/run_tgv_mpirank2_stats_compare.sh"
run_stage np2_field env \
  CPU_EXE="$CPU_EXE" GPU_EXE="$GPU_EXE" \
  OUT_DIR="$OUT_DIR/np2_field" MAXSTEP="$CORRECTNESS_MAXSTEP" \
  FEQCHKPT="$CORRECTNESS_MAXSTEP" LFILTER=t DIFFTERM=t SCHEME=643e \
  ATOL=1e-10 RTOL=1e-10 MPI_NP=2 TOPOLOGY=2,1,1 \
  "$ROOT_DIR/tests/gpu_validation/run_tgv_mpirank2_field_compare.sh"

if [[ "$GATE_SET" == performance || "$GATE_SET" == full ]]; then
  run_stage benchmark env \
    GPU_EXE="$GPU_EXE" OUT_DIR="$OUT_DIR/benchmark" \
    LABEL="$CANDIDATE_ID" GRID="$PERFORMANCE_GRID" \
    MAXSTEP="$PERFORMANCE_MAXSTEP" REPEATS="$PERFORMANCE_REPEATS" \
    DISCARD_STEPS=1 GPU_ID="$GPU_ID" SYNC_MODE=explicit FEQCHKPT=9999 \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
  run_stage performance_compare python3 \
    "$ROOT_DIR/tests/gpu_validation/compare_tgv_candidate_performance.py" \
    --baseline "$BASELINE_TIMINGS" \
    --candidate "$OUT_DIR/benchmark/${CANDIDATE_ID}_timings.tsv" \
    --report "$OUT_DIR/performance_comparison.md" \
    --min-time-reduction-percent "$MIN_TIME_REDUCTION_PERCENT" \
    --max-spread-percent "$MAX_SPREAD_PERCENT" \
    --max-memory-growth-percent "$MAX_MEMORY_GROWTH_PERCENT"
  run_stage nsys_residency env \
    GPU_EXE="$GPU_EXE" OUT_DIR="$OUT_DIR/nsys" PROFILE_TOOL=nsys \
    GRID="$PERFORMANCE_GRID" MAXSTEP=2 FEQCHKPT=9999 GPU_ID="$GPU_ID" \
    SYNC_MODE=explicit \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_256_performance_profile.sh"
fi

if [[ "$GATE_SET" == full ]]; then
  run_stage ncu_hotspot_matrix env \
    GPU_EXE="$GPU_EXE" OUT_DIR="$OUT_DIR/ncu" \
    GRID="$PERFORMANCE_GRID" MAXSTEP=1 FEQCHKPT=9999 GPU_ID="$GPU_ID" \
    SYNC_MODE=explicit NCU_SET=full \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_256_ncu_hotspot_matrix.sh"
  run_stage compute_sanitizer env \
    GPU_EXE="$GPU_EXE" OUT_DIR="$OUT_DIR/memcheck" \
    GRID=32,32,32 MAXSTEP=1 FEQCHKPT=9999 GPU_ID="$GPU_ID" \
    SYNC_MODE=explicit \
    "$ROOT_DIR/tests/gpu_validation/run_tgv_gpu_memcheck.sh"
fi

CURRENT_STAGE="complete"
echo "Candidate gate complete: $SUMMARY_FILE"
