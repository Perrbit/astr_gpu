"""Shared topology, MPI environment and timing definitions for compact runs."""
import math
import os

CPU_TOPOLOGIES = {1:(1,1,1), 2:(2,1,1), 4:(2,2,1), 8:(2,2,2), 16:(4,2,2), 32:(4,4,2)}
GPU_TOPOLOGIES = [(1,1,1),(2,1,1),(1,2,1),(1,1,2),
                  (4,1,1),(1,4,1),(1,1,4),(2,2,1),(2,1,2),(1,2,2)]
WEAK_TOPOLOGIES = [(1,1,1),(1,1,2),(1,2,2)]


def configuration_matrix(strong=32, local=16):
    rows=[]
    for scheme in ('643e','643c'):
        for backend in ('cpu','gpu'):
            topologies=list(GPU_TOPOLOGIES)
            if backend=='cpu':
                topologies += [CPU_TOPOLOGIES[n] for n in (8,16,32)]
            for topology in topologies:
                rows.append(dict(backend=backend,scheme=scheme,family='strong',
                    grid=(strong,)*3,topology=topology,ranks=math.prod(topology)))
            for topology in WEAK_TOPOLOGIES:
                rows.append(dict(backend=backend,scheme=scheme,family='weak',
                    grid=tuple(local*n for n in topology),topology=topology,ranks=math.prod(topology)))
    return rows


def formal_configuration_matrix(backend='all'):
    rows=[r for r in configuration_matrix(512,256) if r['backend']=='gpu' or
          (r['family']=='strong' and r['topology']==CPU_TOPOLOGIES[r['ranks']])]
    if backend!='all':
        rows=[row for row in rows if row['backend']==backend]
    return sorted(rows,key=lambda r:(r['backend'],r['family'],r['ranks'],r['topology'],r['scheme']))


def mpi_environment(backend, halo_transport='pinned', compact_transport='pinned', ucx_tls='self,sm,cuda_copy'):
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_opal_cuda_support='0')
    if backend=='gpu' and 'device-aware' in (halo_transport,compact_transport):
        env.update(OMPI_MCA_pml='ucx',UCX_TLS=ucx_tls,UCX_MEMTYPE_CACHE='n',
            UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',
            UCX_CUDA_IPC_ENABLE_MNNVL='no',OMPI_MCA_coll_hcoll_enable='0')
        env.pop('OMPI_MCA_opal_cuda_support')
    return env


def timing_samples(text, backend, ranks, steps, warmup):
    label=f'ASTR_{backend.upper()}_RANK_RK_TIMING'
    records={}
    for line in text.splitlines():
        if not line.startswith(label+' '):
            continue
        _,rank,step,prepare,integration,total=line.split()
        key=int(rank),int(step)
        if key in records:
            raise ValueError(f'duplicate rank timing: {key}')
        a,b,t=map(float,(prepare,integration,total))
        if not all(math.isfinite(v) and v>=0 for v in (a,b,t)) or not math.isclose(a+b,t,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError(f'invalid rank timing: {key}')
        records[key]=t
    if set(records)!={(rank,step) for rank in range(ranks) for step in range(steps)}:
        raise ValueError('missing or unexpected rank/step timing')
    return [max(records[rank,step] for rank in range(ranks)) for step in range(warmup,steps)]
