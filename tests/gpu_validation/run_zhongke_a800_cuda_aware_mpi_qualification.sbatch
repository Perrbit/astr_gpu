#!/bin/bash
#SBATCH --job-name=astr-a800-cuda-mpi
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-a800-cuda-mpi-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-a800-cuda-mpi-%j.err

set -uo pipefail

WORK_ROOT=/data/user/hd56000/weiph
BASE_COMMIT=c801eebc8f14a84d045eb60db92e9c022ccce7b7
SRC_DIR="${SRC_DIR:-$WORK_ROOT/astr_gpu_cuda_aware_qualification_c801eeb}"
BUILD_DIR="${BUILD_DIR:-$WORK_ROOT/build_cuda_aware_qualification_c801eeb}"
HDF5_ROOT="${HDF5_ROOT:-$WORK_ROOT/opt/hdf5-1.14.5-nvhpc25.5-noszip}"
PYTHON_EXE="${PYTHON_EXE:-$WORK_ROOT/venv_tgv_campaign_py311/bin/python}"
RUN_ROOT="${RUN_ROOT:-$WORK_ROOT/results/a800_cuda_aware_qualification_${SLURM_JOB_ID:-dryrun}}"
EXPECTED_PROBE_SOURCE_SHA256=4a018f19d657cf8176fea8c8bb531706341b34e2b1447bcafe630c968451ea9e

for path in "$SRC_DIR" "$BUILD_DIR" "$HDF5_ROOT" "$RUN_ROOT"; do
  case "$path/" in
    "$WORK_ROOT"/*) ;;
    *) printf 'path escapes approved work root: %s\n' "$path" >&2; exit 2 ;;
  esac
done

module purge
module load gcc/11.5
module load nvhpc/25.5

export HDF5_ROOT
export PATH="$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"

PROBE_EXE="$BUILD_DIR/bin/halo_cuda_aware_probe"
QUALIFICATION_DRIVER="$SRC_DIR/tests/gpu_validation/run_cuda_aware_mpi_qualification.sh"
for required in \
  "$PROBE_EXE" \
  "$QUALIFICATION_DRIVER" \
  "$SRC_DIR/tests/gpu_validation/summarize_cuda_aware_mpi_qualification.py" \
  "$SRC_DIR/tests/gpu_validation/compute_sanitizer_ucx_cuda_aware.supp.xml" \
  "$SRC_DIR/tests/gpu_validation/halo_cuda_aware_probe.cuf" \
  "$SRC_DIR/BASE_COMMIT"; do
  [[ -f "$required" ]] || { printf 'missing qualification input: %s\n' "$required" >&2; exit 2; }
done
[[ -x "$PROBE_EXE" ]] || { printf 'probe is not executable: %s\n' "$PROBE_EXE" >&2; exit 2; }
[[ -x "$PYTHON_EXE" ]] || { printf 'Python 3.11 is unavailable: %s\n' "$PYTHON_EXE" >&2; exit 2; }
[[ "$(cat "$SRC_DIR/BASE_COMMIT")" == "$BASE_COMMIT" ]] || {
  printf 'source base commit does not match qualification contract\n' >&2
  exit 2
}
[[ "$(sha256sum "$SRC_DIR/tests/gpu_validation/halo_cuda_aware_probe.cuf" | awk '{print $1}')" == \
   "$EXPECTED_PROBE_SOURCE_SHA256" ]] || {
  printf 'probe source hash does not match qualification contract\n' >&2
  exit 2
}
if ldd "$PROBE_EXE" | grep -q 'not found'; then
  printf 'unresolved compute-node probe dependency\n' >&2
  ldd "$PROBE_EXE" >&2
  exit 2
fi
[[ ! -e "$RUN_ROOT" ]] || {
  printf 'refusing to overwrite qualification output: %s\n' "$RUN_ROOT" >&2
  exit 2
}

visible="${CUDA_VISIBLE_DEVICES:?Slurm did not expose CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a gpu <<< "$visible"
[[ "${#gpu[@]}" -eq 2 ]] || {
  printf 'expected two Slurm-visible GPUs, got: %s\n' "$visible" >&2
  exit 2
}

export QUALIFICATION_CUDA_VISIBLE_DEVICES="$visible"
export PROBE_EXE
export OUT_DIR="$RUN_ROOT"
export PAYLOAD_REPEATS=5
export PYTHON_EXE

bash "$QUALIFICATION_DRIVER"
