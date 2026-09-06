#!/usr/bin/env bash
set -euo pipefail

P2_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$P2_ROOT_DIR/tests/gpu_validation/p2_shock_case_common.sh"

CASE="${CASE:-shuosher}"
p2_validate_case "$CASE"
PROFILE_TOOL="${PROFILE_TOOL:-nsys}"
GPU_EXE="${GPU_EXE:-$P2_ROOT_DIR/build_gpu_probe/bin/astr}"
OUT_DIR="${OUT_DIR:-$P2_ROOT_DIR/tests/gpu_validation/out/p2_${CASE}_${PROFILE_TOOL}}"
GRID="${GRID:-$(p2_default_grid "$CASE")}"
GPU_ID="${GPU_ID:-0}"
GPU_IDS="${GPU_IDS:-$GPU_ID}"
NP="${NP:-1}"
TOPOLOGY="${TOPOLOGY:-1,1,1}"
MAXSTEP="${MAXSTEP:-1}"
FEQCHKPT=9999
FEQLIST=9999
NSYS_TRACE="${NSYS_TRACE:-cuda}"
NCU_SET="${NCU_SET:-full}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_PROCESS_FILTER="${NCU_PROCESS_FILTER:-regex:^astr$}"

if [[ "$GPU_EXE" != /* ]]; then GPU_EXE="$P2_ROOT_DIR/$GPU_EXE"; fi
if [[ "$OUT_DIR" != /* ]]; then OUT_DIR="$P2_ROOT_DIR/$OUT_DIR"; fi
[[ -x "$GPU_EXE" ]] || { printf 'GPU executable not found: %s\n' "$GPU_EXE" >&2; exit 2; }
case "$PROFILE_TOOL" in
  nsys|ncu) command -v "$PROFILE_TOOL" >/dev/null 2>&1 || { printf '%s not found\n' "$PROFILE_TOOL" >&2; exit 127; } ;;
  *) printf 'PROFILE_TOOL must be nsys or ncu\n' >&2; exit 2 ;;
esac

if [[ -z "${NCU_KERNEL:-}" ]]; then
  if [[ "$CASE" == shuosher ]]; then
    NCU_KERNEL=characteristic_upwind_flux_x_global_kernel
  else
    NCU_KERNEL=characteristic_upwind_flux_x_xyphysical_global_kernel
  fi
fi

CASE_DIR="$OUT_DIR/case"
mkdir -p "$OUT_DIR" "$OUT_DIR/tmp"
p2_prepare_case "$CASE" "$CASE_DIR" "$GRID" "$MAXSTEP" "$FEQCHKPT" "$FEQLIST"
printf 'case=%s\ngrid=%s\nmaxstep=%s\nprofile_tool=%s\nnp=%s\ntopology=%s\ngpu_ids=%s\nnsys_trace=%s\nncu_kernel=%s\n' \
  "$CASE" "$GRID" "$MAXSTEP" "$PROFILE_TOOL" "$NP" "$TOPOLOGY" \
  "$GPU_IDS" "$NSYS_TRACE" "$NCU_KERNEL" > "$OUT_DIR/profile_metadata.txt"

if [[ "$PROFILE_TOOL" == nsys ]]; then
  (
    cd "$CASE_DIR"
    env "${P2_RUNTIME_ENV[@]}" TMPDIR="$OUT_DIR/tmp" \
      OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_RK_TIMING=1 \
      nsys profile --trace="$NSYS_TRACE" --sample=none --cpuctxsw=none --stats=false \
        --force-overwrite=true \
        -o ../p2_shock_nsys mpirun -np "$NP" "$GPU_EXE" run "$P2_INPUT" \
        > ../nsys.log 2>&1
  )
  nsys stats --force-export=true --report cuda_gpu_kern_sum \
    "$OUT_DIR/p2_shock_nsys.nsys-rep" \
    > "$OUT_DIR/cuda_gpu_kern_sum.txt"
  if [[ ",$NSYS_TRACE," == *,mpi,* ]]; then
    nsys stats --force-export=true --report mpi_event_sum \
      "$OUT_DIR/p2_shock_nsys.nsys-rep" \
      > "$OUT_DIR/mpi_event_sum.txt"
    nsys stats --force-export=true --report mpi_msg_size_sum \
      "$OUT_DIR/p2_shock_nsys.nsys-rep" \
      > "$OUT_DIR/mpi_msg_size_sum.txt"
  fi
else
  (
    cd "$CASE_DIR"
    env "${P2_RUNTIME_ENV[@]}" TMPDIR="$OUT_DIR/tmp" \
      OMPI_MCA_sharedfp="${OMPI_MCA_sharedfp:-individual}" CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      ASTR_FORCE_MPI_TOPOLOGY="$TOPOLOGY" ASTR_GPU_SYNC_MODE=explicit \
      ncu --target-processes all --target-processes-filter "$NCU_PROCESS_FILTER" \
        --set "$NCU_SET" --kernel-name "regex:${NCU_KERNEL}" \
        --launch-skip "$NCU_LAUNCH_SKIP" --launch-count 1 \
        --import-source yes --source-folders "$P2_ROOT_DIR/src_gpu" \
        --force-overwrite --export ../p2_shock_kernel \
        mpirun -np "$NP" "$GPU_EXE" run "$P2_INPUT" > ../ncu.log 2>&1
  )
  ncu --import "$OUT_DIR/p2_shock_kernel.ncu-rep" \
    --page details --print-details all > "$OUT_DIR/ncu_details.txt"
  ncu --import "$OUT_DIR/p2_shock_kernel.ncu-rep" \
    --page source --print-source cuda,sass --csv --print-units base \
    > "$OUT_DIR/ncu_source.csv"
fi
