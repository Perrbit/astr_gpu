#!/bin/bash
#SBATCH --job-name=astr-m4-pair
#SBATCH --partition=kshdnormal
#SBATCH --nodes=8
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:Hygon:4
#SBATCH --mem=110G
#SBATCH --time=00:45:00
#SBATCH --output=m4-%j.log

set -eo pipefail
work=$(cd "$SLURM_SUBMIT_DIR" && pwd -P)
cd "$work"
source "$work/stack.sh"
sha256sum -c baseline.sha256 regional.sha256
export OMP_NUM_THREADS=1 OMPI_MCA_orte_tmpdir_base="$TMPDIR" OMPI_MCA_osc=pt2pt
export OMPI_MCA_pml=ucx OMPI_MCA_coll=^cuda,hcoll,ucc OMPI_MCA_opal_cuda_support=0
export OMPI_MCA_io=romio321 OMPI_MCA_plm_slurm_args="--jobid=$SLURM_JOB_ID --overlap"
export UCX_MODULE_DIR="$ucx/lib/ucx" UCX_MEMTYPE_CACHE=n UCX_PROTO_ENABLE=y
export UCX_MAX_RNDV_RAILS=1 UCX_LOG_LEVEL=warn UCX_PROTO_INFO=n
export UCX_TLS=rc,posix,sysv,self,rocm UCX_ROCM_IPC_CACHE_IPC_HANDLES=y HSA_ENABLE_SDMA=1
unset UCX_LOG_FILE UCX_NET_DEVICES HSA_TOOLS_LIB AMD_SERIALIZE_KERNEL AMD_SERIALIZE_COPY
ulimit -c 0
exec "$python" -B "$work/run_dcu_m4_pair.py" run "$work/job_$SLURM_JOB_ID" \
    "$base/dcu_weak_scaling_a0e9217/run_nodes.py" "$base/dcu_tgv_device308_df8064f_romio/hca_affinity.py" \
    "$base/dcu_tgv_device308_independent/probe_pci" "$mpi/bin/mpiexec"
