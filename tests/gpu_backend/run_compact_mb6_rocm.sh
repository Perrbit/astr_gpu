#!/bin/bash
# Compute-node body only. Scheduler submission requires separate authorization.
set -e -o pipefail
work=${1:?batch directory is required}
base=${2:?weiph directory is required}
cd "$work"
stack="$base/private_rocm_mpi_20261113/stack_dtk_payload"
mpi="$stack/openmpi-4.1.5"
ucx="$stack/ucx-1.18.1"
gcc=/public/software/compiler/gcc-12.2.0
hdf5="$base/private_rocm_mpi_20261113/hdf5_gnu12/hdf5-1.14.6"
dtk=/public/software/compiler/rocm/dtk-26.04
mkdir -p "$work/home" "$work/cache/comgr" "$base/t/mb6_${SLURM_JOB_ID:?allocation is required}"
export HOME="$work/home" TMPDIR="$base/t/mb6_$SLURM_JOB_ID" XDG_CACHE_HOME="$work/cache"
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
unset ASTR_GPU_RK_TIMING ASTR_GPU_RANK_RK_TIMING ASTR_GPU_BENCHMARK_NO_FIELD_IO
unset ASTR_VALIDATION_RHS_PREFIX ASTR_VALIDATION_RHS_STEP ASTR_VALIDATION_RHS_STEP_SECONDARY
date -Is
hostname
ldd "$work/bin/astr" > "$work/compute_runtime_links.log"
if grep -q 'not found' "$work/compute_runtime_links.log"; then exit 1; fi
"$work/bin/astr_backend_compact_line" > "$work/compact_line.log" 2>&1
mkdir "$work/components"
transport() {
  if [ "$1" = device-aware ]; then
    export OMPI_MCA_pml=ucx ASTR_GPU_HALO_TRANSPORT=device
  else
    export OMPI_MCA_pml=ob1 ASTR_GPU_HALO_TRANSPORT=pinned
  fi
}
while read -r mode ranks x y z; do
  transport "$mode"
  output="$work/components/${mode}_np${ranks}_${x}x${y}x${z}"
  mkdir "$output"
  "$mpi/bin/mpiexec" --bind-to none -n "$ranks" "$work/bin/astr_backend_compact_distributed" \
    "$output" "$x" "$y" "$z" "$mode" </dev/null > "$output/run.log" 2>&1
done < "$work/components.tsv"
export ASTR_GPU_SYNC_MODE=dependency ASTR_VALIDATION_RHS_PREFIX=outdat/rhs ASTR_VALIDATION_RHS_STEP=4
while read -r name ranks topology mode workspace sync form check left; do
  transport "$mode"
  export ASTR_FORCE_MPI_TOPOLOGY="$topology" ASTR_GPU_COMPACT_TRANSPORT="$mode"
  export ASTR_GPU_FILTER_WORKSPACE="$workspace" ASTR_GPU_COMPACT_SYNC="$sync"
  printf 'COMPACT_MAIN %s ranks=%s topology=%s transport=%s\n' "$name" "$ranks" "$topology" "$mode"
  (cd "$work/cases/$name"; "$mpi/bin/mpiexec" --bind-to none -n "$ranks" "$work/bin/astr" run \
    </dev/null > run.log 2>&1)
  for backend in cpu old_gpu; do
    "$work/bin/astr_backend_compare" field "$work/references/$form/$backend/state.h5" "$work/cases/$name"
  done
  if [ "$check" != none ]; then
    "$work/bin/astr_backend_compare" "$check" "$work/$left" \
      "$work/cases/$name/outdat/new/checkpoints/step000000000004/state.h5"
  fi
  if [ "$name" = primitive_archives ]; then
    shopt -s nullglob
    frames=("$work/cases/$name"/outdat/new/fields/segment*/step*/data.h5)
    test "${#frames[@]}" -eq 2
    frames=("$work/cases/$name"/outdat/new/slice_products/*/*/segment*/step*.h5)
    test "${#frames[@]}" -eq 3
    shopt -u nullglob
  fi
done < "$work/gates.tsv"
date -Is
