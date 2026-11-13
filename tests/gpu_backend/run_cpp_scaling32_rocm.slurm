#!/bin/bash
#SBATCH --job-name=astr-dcu32-scaling
#SBATCH --partition=kshdnormal02
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:Hygon:4
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=scaling-%j.log
set -e -o pipefail
cd "$SLURM_SUBMIT_DIR"
work=$PWD
base=/public/home/ac389hy6i1/weiph
stack=$base/private_rocm_mpi_20261113/stack_dtk_payload
mpi=$stack/openmpi-4.1.5
ucx=$stack/ucx-1.18.1
gcc=/public/software/compiler/gcc-12.2.0
dtk=/public/software/compiler/rocm/dtk-26.04
hdf=$base/private_rocm_mpi_20261113/hdf5_gnu12/hdf5-1.14.6
mkdir -p "$work/home" "$work/cache/comgr" "$base/t/$SLURM_JOB_ID"
export HOME=$work/home TMPDIR=$base/t/$SLURM_JOB_ID XDG_CACHE_HOME=$work/cache
export AMD_COMGR_CACHE_DIR=$work/cache/comgr LC_ALL=C LANG=C OMP_NUM_THREADS=1
unset CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$gcc/lib64:$mpi/lib:$ucx/lib:$hdf/lib
source "$dtk/env.sh"
source "$dtk/cuda/env.sh"
export PATH=$gcc/bin:$mpi/bin:$PATH OPAL_PREFIX=$mpi
export LD_LIBRARY_PATH=$gcc/lib64:$mpi/lib:$ucx/lib:$hdf/lib:$LD_LIBRARY_PATH
export OMPI_MCA_orte_tmpdir_base=$TMPDIR OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt
export OMPI_MCA_coll=^cuda,hcoll,ucc OMPI_MCA_opal_cuda_support=0
export UCX_MODULE_DIR=$ucx/lib/ucx UCX_MEMTYPE_CACHE=n
export ASTR_GPU_FILTER_WORKSPACE=scalar ASTR_GPU_SYNC_MODE=dependency ASTR_GPU_STATISTICS_INTERVAL=1
binary=$work/bin/astr
date -Is
hostname
grep -E 'MemTotal:|MemAvailable:' /proc/meminfo
sha256sum "$binary"
ldd "$binary" > "$work/runtime_links.log"
if grep 'not found' "$work/runtime_links.log"; then exit 1; fi
"$ucx/bin/ucx_info" -v
"$ucx/bin/ucx_info" -d > "$work/ucx_devices.log"
export OMPI_MCA_pml=ob1 ASTR_GPU_HALO_TRANSPORT=pinned
"$mpi/bin/mpiexec" --bind-to none -n 4 "$work/bin/astr_backend_mb0" "$work/qualification.h5" </dev/null > qualification.log 2>&1
for mode in pinned device; do
  export ASTR_GPU_HALO_TRANSPORT=$mode
  if [ "$mode" = device ]; then export OMPI_MCA_pml=ucx; else export OMPI_MCA_pml=ob1; fi
  "$mpi/bin/mpiexec" --bind-to none -n 4 "$work/bin/astr_backend_halo" 0 "$mode" </dev/null >"halo_$mode.log" 2>&1
done
export ASTR_FORCE_MPI_TOPOLOGY=2,2,1 ASTR_GPU_HALO_TRANSPORT=device OMPI_MCA_pml=ucx
export ASTR_VALIDATION_RHS_PREFIX=outdat/rhs ASTR_VALIDATION_RHS_STEP=4
# Match the frozen four-step dt for this small compatibility check only.
(cd "$work/preflight"; "$mpi/bin/mpiexec" --bind-to none -n 4 "$binary" run </dev/null >run.log 2>&1)
for backend in cpu gpu; do
  "$work/bin/astr_backend_compare" field "$work/reference_$backend.h5" "$work/preflight"
done
unset ASTR_VALIDATION_RHS_PREFIX ASTR_VALIDATION_RHS_STEP ASTR_VALIDATION_RHS_STEP_SECONDARY ASTR_CFL_PROFILE_Y
export ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_BENCHMARK_NO_FIELD_IO=1
while read -r name ranks topology mode; do
  export ASTR_FORCE_MPI_TOPOLOGY=$topology ASTR_GPU_HALO_TRANSPORT=$mode
  if [ "$mode" = device ]; then export OMPI_MCA_pml=ucx; else export OMPI_MCA_pml=ob1; fi
  printf 'RUN %s ranks=%s topology=%s transport=%s\n' "$name" "$ranks" "$topology" "$mode"
  grep -E 'MemTotal:|MemAvailable:' /proc/meminfo >"$work/cases/$name/node_memory_before.log"
  (cd "$work/cases/$name"; "$mpi/bin/mpiexec" --bind-to none -n "$ranks" "$binary" run </dev/null >run.log 2>&1)
  grep -E 'MemTotal:|MemAvailable:' /proc/meminfo >"$work/cases/$name/node_memory_after.log"
  printf 'DONE %s\n' "$name"
done < "$work/runs.tsv"
date -Is
