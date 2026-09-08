#!/bin/bash
#SBATCH --job-name=astr-a800-tgv
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-tgv-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-tgv-%j.err

set -euo pipefail

WORK_ROOT=/data/user/hd56000/weiph
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_gpu_a800_current}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_tgv_scaling_${SLURM_JOB_ID}}"

for path in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$RUN_ROOT"; do
  case "$path/" in
    "$WORK_ROOT"/*) ;;
    *) printf 'Path escapes approved work root: %s\n' "$path" >&2; exit 2 ;;
  esac
done

module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"

GPU_EXE="$BUILD_DIR/bin/astr"
[[ -x "$GPU_EXE" ]] || { printf 'GPU executable not found: %s\n' "$GPU_EXE" >&2; exit 2; }
[[ -d "$SRC_DIR/.git" ]] || { printf 'Source checkout not found: %s\n' "$SRC_DIR" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { printf 'Refusing to overwrite run root: %s\n' "$RUN_ROOT" >&2; exit 2; }
mkdir -p "$RUN_ROOT"

visible="${CUDA_VISIBLE_DEVICES:?SLURM did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 4 ]] || {
  printf 'Expected four visible GPUs, got: %s\n' "$visible" >&2
  exit 2
}
gpu1="${gpu[0]}"
gpu2="${gpu[0]},${gpu[1]}"
gpu4="${gpu[0]},${gpu[1]},${gpu[2]},${gpu[3]}"

{
  printf 'source_commit: %s\n' "$(git -C "$SRC_DIR" rev-parse HEAD)"
  printf 'source_status:\n'
  git -C "$SRC_DIR" status --short
  printf 'executable_sha256: '
  sha256sum "$GPU_EXE"
  printf 'date: %s\n' "$(date -Iseconds)"
  printf 'hostname: %s\n' "$(hostname)"
  printf 'slurm_job_id: %s\n' "$SLURM_JOB_ID"
  printf 'slurm_job_nodelist: %s\n' "$SLURM_JOB_NODELIST"
  printf 'cuda_visible_devices: %s\n' "$visible"
  printf '\nloaded_modules:\n'
  module -t list 2>&1
  printf '\nnvidia_smi:\n'
  nvidia-smi
  printf '\nnvidia_topology:\n'
  nvidia-smi topo -m
  printf '\nnvfortran:\n'
  nvfortran --version
  printf '\nmpirun:\n'
  mpirun --version
  printf '\nhdf5:\n'
  h5pfc -showconfig
  printf '\ncmake:\n'
  cmake --version
} > "$RUN_ROOT/environment.txt"

if grep -E -- '(--use_fast_math|-Mfprelaxed|(^|[[:space:]])-fast([[:space:]]|$))' \
  "$BUILD_DIR/compile_commands.json"; then
  printf 'Prohibited relaxed-math compiler option found\n' >&2
  exit 2
fi

preflight_case="$RUN_ROOT/t1_np1_preflight_case"
python3 "$SRC_DIR/tests/gpu_validation/prepare_tgv_case.py" \
  --src-case "$SRC_DIR/examples/Taylor_Green_Vortex" \
  --dst-case "$preflight_case" --use-gpu t --grid 512,512,512 \
  --maxstep 0 --feqchkpt 9999 --lfilter t --diffterm t --scheme 643e
(
  cd "$preflight_case"
  CUDA_VISIBLE_DEVICES="$gpu1" ASTR_FORCE_MPI_TOPOLOGY=1,1,1 \
    ASTR_GPU_SYNC_MODE=explicit ASTR_GPU_HALO_TRANSPORT=pinned-overlap \
    mpirun -np 1 "$GPU_EXE" run datin/input.tgv > "$RUN_ROOT/t1_np1_preflight.log" 2>&1
)
grep -q 'The job is done!' "$RUN_ROOT/t1_np1_preflight.log"
if grep -Eq 'COMPUTATION CRASHED|ieee_invalid|ieee_divide_by_zero|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' \
  "$RUN_ROOT/t1_np1_preflight.log"; then
  printf 'T1 preflight found a crash or non-finite marker\n' >&2
  exit 1
fi

run_timing() {
  local label="$1" np="$2" topology="$3" gpu_ids="$4" halo_transport="$5"
  GPU_EXE="$GPU_EXE" OUT_DIR="$RUN_ROOT/$label" LABEL="$label" \
    GRID=512,512,512 MAXSTEP=21 DISCARD_STEPS=2 REPEATS=5 \
    NP="$np" TOPOLOGY="$topology" GPU_IDS="$gpu_ids" \
    HALO_TRANSPORT="$halo_transport" SYNC_MODE=explicit FEQCHKPT=9999 \
    "$SRC_DIR/tests/gpu_validation/run_tgv_256_performance_benchmark.sh"
}

run_timing t2_np1 1 1,1,1 "$gpu1" pinned-overlap
run_timing t3_np2_z 2 1,1,2 "$gpu2" pinned-overlap
run_timing t4_np4_z 4 1,1,4 "$gpu4" pinned-overlap
run_timing t5_np4_pageable 4 1,1,4 "$gpu4" pageable

printf 'status: pass\n' > "$RUN_ROOT/job_status.txt"
