#!/bin/bash
#SBATCH --job-name=astr-cpp-mb45-aware
#SBATCH --partition=kshdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --gres=dcu:Hygon:4
#SBATCH --time=00:10:00
#SBATCH --output=mb45-%j.log
set -e -o pipefail
cd "$SLURM_SUBMIT_DIR"
work=$PWD
base=/public/home/ac389hy6i1/weiph
stack="$base/private_rocm_mpi_20261113/stack_dtk_payload"
mpi="$stack/openmpi-4.1.5"
ucx="$stack/ucx-1.18.1"
dtk=/public/software/compiler/rocm/dtk-26.04
gcc=/public/software/compiler/gcc-12.2.0
hdf5="$base/private_rocm_mpi_20261113/hdf5_gnu12/hdf5-1.14.6"
mkdir -p "$work/home" "$work/cache/comgr" "$base/t/$SLURM_JOB_ID"
export HOME="$work/home" TMPDIR="$base/t/$SLURM_JOB_ID" XDG_CACHE_HOME="$work/cache"
export AMD_COMGR_CACHE_DIR="$work/cache/comgr" LC_ALL=C LANG=C OMP_NUM_THREADS=1
unset CPATH LIBRARY_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:$ucx/lib:$hdf5/lib"
source "$dtk/env.sh"
source "$dtk/cuda/env.sh"
export PATH="$gcc/bin:$mpi/bin:$PATH" OPAL_PREFIX="$mpi"
export LD_LIBRARY_PATH="$gcc/lib64:$mpi/lib:$ucx/lib:$hdf5/lib:$LD_LIBRARY_PATH"
export OMPI_MCA_orte_tmpdir_base="$TMPDIR" OMPI_MCA_btl=self,vader,tcp OMPI_MCA_osc=pt2pt
export OMPI_MCA_coll=^cuda,hcoll,ucc OMPI_MCA_opal_cuda_support=0
export UCX_MODULE_DIR="$ucx/lib/ucx" UCX_MEMTYPE_CACHE=n
export ASTR_CFL_PROFILE_Y=1
unset ASTR_GPU_RK_TIMING ASTR_GPU_RANK_RK_TIMING ASTR_GPU_BENCHMARK_NO_FIELD_IO
binary="$work/build/bin/astr"
compare="$work/build/bin/astr_backend_compare"
date -Is
hostname
"$ucx/bin/ucx_info" -v
"$ucx/bin/ucx_info" -d > "$work/ucx_devices.log"
ldd "$binary" | tee "$work/runtime_links.log"
if grep -q 'not found' "$work/runtime_links.log"; then exit 1; fi
transport() {
  export ASTR_GPU_HALO_TRANSPORT=$1
  if [ "$1" = device ]; then export OMPI_MCA_pml=ucx; else export OMPI_MCA_pml=ob1; fi
}
while read -r mode ranks axis; do
  transport "$mode"
  "$mpi/bin/mpiexec" --bind-to none -n "$ranks" "$work/build/bin/astr_backend_halo" "$axis" "$mode" \
    </dev/null > "$work/halo_${mode}_${ranks}_${axis}.log" 2>&1
done < "$work/components.tsv"
while read -r name ranks topology workspace sync mode trace steps reference check left; do
  transport "$mode"
  export ASTR_FORCE_MPI_TOPOLOGY="$topology" ASTR_GPU_FILTER_WORKSPACE="$workspace" ASTR_GPU_SYNC_MODE="$sync"
  export ASTR_VALIDATION_RHS_PREFIX=outdat/rhs ASTR_VALIDATION_RHS_STEP_SECONDARY="$steps"
  if [ "$trace" = 1 ]; then export ASTR_VALIDATION_RHS_STEP=0; else export ASTR_VALIDATION_RHS_STEP="$steps"; fi
  printf 'GATE %s ranks=%s topology=%s transport=%s\n' "$name" "$ranks" "$topology" "$mode"
  (cd "$work/cases/$name"; "$mpi/bin/mpiexec" --bind-to none -n "$ranks" "$binary" run </dev/null > run.log 2>&1)
  if [[ "$name" = io_archive* ]]; then
    root="$work/cases/$name/outdat/new"
    shopt -s nullglob
    frames=("$root"/slice_products/*/*/segment*/step*.h5)
    test "${#frames[@]}" -eq 27
    for step in 0 2 4; do
      printf -v filename 'step%012d.h5' "$step"
      frames=("$root"/slice_products/*/*/segment*/"$filename")
      test "${#frames[@]}" -eq 9
    done
    frames=("$root"/fields/segment*/step*/data.h5)
    test "${#frames[@]}" -eq 2
    shopt -u nullglob
    printf '27 numerical slices and two volume frames present\n'
  fi
  if [ "$reference" != none ]; then
    for backend in cpu gpu; do
      if [ "$reference" = fixed ]; then ref="$work/references/${backend}_exact/checkpoints/step000000000004/state.h5"
      else ref="$work/references/${backend}_${reference}.h5"; fi
      "$compare" field "$ref" "$work/cases/$name"
    done
  fi
  if [ "$check" != none ]; then
    "$compare" "$check" "$work/$left" "$work/cases/$name/outdat/new/checkpoints/step000000000004/state.h5"
  fi
done < "$work/gates.tsv"
unset ASTR_VALIDATION_RHS_PREFIX ASTR_VALIDATION_RHS_STEP ASTR_VALIDATION_RHS_STEP_SECONDARY ASTR_CFL_PROFILE_Y
export ASTR_GPU_RK_TIMING=1 ASTR_GPU_RANK_RK_TIMING=1 ASTR_GPU_BENCHMARK_NO_FIELD_IO=1
export ASTR_GPU_FILTER_WORKSPACE=scalar ASTR_GPU_SYNC_MODE=dependency ASTR_GPU_STATISTICS_INTERVAL=1
while read -r name ranks topology mode; do
  transport "$mode"
  export ASTR_FORCE_MPI_TOPOLOGY="$topology"
  printf 'TIMING %s ranks=%s topology=%s transport=%s\n' "$name" "$ranks" "$topology" "$mode"
  (cd "$work/cases/$name"; "$mpi/bin/mpiexec" --bind-to none -n "$ranks" "$binary" run </dev/null > run.log 2>&1)
done < "$work/timing.tsv"
date -Is
