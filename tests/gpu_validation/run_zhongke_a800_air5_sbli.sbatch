#!/bin/bash
#SBATCH --job-name=astr-air5-sbli-p03
#SBATCH --partition=A800-N
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user/hd56000/weiph/jobs/astr-air5-sbli-%j.out
#SBATCH --error=/data/user/hd56000/weiph/jobs/astr-air5-sbli-%j.err
set -euo pipefail
ROOT=/data/user/hd56000/weiph/uploads/air5_review_31c01ae_20260926
cd "$ROOT"
module purge
module load gcc/11.5 nvhpc/25.5
export TMPDIR="$ROOT/tmp/job_${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR"
export PMIX_MCA_gds=hash
export HDF5_ROOT=/data/user/hd56000/weiph/opt/hdf5-1.14.5-nvhpc25.5-noszip
export PATH="$HDF5_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
export OMPI_MCA_coll='^hcoll,ucc,cuda'
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_osc=pt2pt
export OMPI_MCA_opal_cuda_support=0
export OMPI_MCA_sharedfp=individual
export OMPI_MCA_io=romio321
export UCX_MEMTYPE_CACHE=n
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
PY="$ROOT/venv/bin/python"
EXE="$ROOT/build_cc80/bin/astr"
RUN=/data/user/hd56000/weiph/results/air5_sbli_31c01ae_${SLURM_JOB_ID:?}
test ! -e "$RUN"
sha256sum -c runtime_manifest.sha256
sha256sum -c executable.sha256
deps=$(ldd "$EXE")
printf '%s\n' "$deps"
if grep -q 'not found' <<< "$deps"; then exit 2; fi
grep -F "$HDF5_ROOT/lib/libhdf5_fortran" <<< "$deps"
test "$(printf '%s' "${CUDA_VISIBLE_DEVICES:?}" | awk -F, '{print NF}')" -eq 4
command -v compute-sanitizer
nvidia-smi
nvidia-smi topo -m
"$PY" -c 'import numpy,h5py,scipy; print(numpy.__version__,h5py.__version__,scipy.__version__)'
"$PY" "$ROOT/runtime/tests/gpu_validation/run_a800_air5_campaign.py" \
  --seed "$ROOT/seed" --output "$RUN" --executable "$EXE"
