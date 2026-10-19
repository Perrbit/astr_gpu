#!/usr/bin/env python3
"""CP8 local cost attribution; no rendering, checkpoint or validation field dumps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from prepare_m12_local_case import prepare
from run_compact_flatplate import configure, GRID

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=('tgv', 'plate'), default='tgv')
    parser.add_argument('--source', type=Path, default=ROOT.parent/'M12_C13')
    parser.add_argument('--topology', default='2,1,1')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--workspace', choices=('scalar','full'), default='full')
    parser.add_argument('--compact-sync', choices=('explicit','dependency'), default='explicit')
    parser.add_argument('--batch-rhs', type=int, default=65535)
    parser.add_argument('--transport', choices=('pinned','device-aware'), default='pinned')
    parser.add_argument('--nsys', action='store_true')
    args = parser.parse_args()
    case=args.out.resolve()
    exe=args.exe.resolve(strict=True)
    if args.case=='tgv':
        subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
            '--src-case',str(ROOT/'examples/Taylor_Green_Vortex'),'--dst-case',str(case),
            '--grid','32,32,32','--use-gpu','t','--scheme','643c','--lfilter','t',
            '--diffterm','t','--deltat','1.d-3','--maxstep',str(args.steps-1)],check=True)
        primary='datin/input.tgv'
    else:
        prepare(args.source,case,'gpu',GRID)
        configure(case,'nscbc','643c',False,'t')
        controller=case/'datin/controller'
        lines=controller.read_text().splitlines()
        index=next(i for i,line in enumerate(lines) if '# maxstep,feqchkpt' in line)+1
        fields=lines[index].split(',');fields[0]=str(args.steps-1)
        lines[index]=','.join(fields)
        controller.write_text('\n'.join(lines)+'\n')
        primary='datin/input.dat'
    (case/'datin/input.output').write_text(
        '&output\n host_budget_bytes=67108864\n/\n&checkpoint\n/\n&volume\n/\n&slices\n/\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=args.topology,ASTR_GPU_COMPACT_SYNC=args.compact_sync,
        ASTR_GPU_COMPACT_BATCH_RHS=str(args.batch_rhs),ASTR_GPU_COMPACT_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_PRECISION_MODE='fp64',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_FILTER_WORKSPACE=args.workspace,ASTR_GPU_RK_TIMING='1',ASTR_GPU_RANK_RK_TIMING='1',
        ASTR_COMPLETE_STEP_TIMING='1',
        ASTR_NSCBC_FARFIELD_MODE='incoming_only',ASTR_NSCBC_FARFIELD_RHO='1',ASTR_NSCBC_FARFIELD_U='1',
        ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',ASTR_NSCBC_FARFIELD_T='1',
        OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_opal_cuda_support='0')
    if args.case=='plate':
        env['ASTR_WALL_BLOWING_MODE']='legacy_random'
    if args.transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx',UCX_TLS='self,sm,cuda_copy',UCX_MEMTYPE_CACHE='n',
            UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',UCX_CUDA_IPC_ENABLE_MNNVL='no',
            OMPI_MCA_coll_hcoll_enable='0')
        env.pop('OMPI_MCA_opal_cuda_support')
    ranks=int(np.prod(tuple(map(int,args.topology.split(',')))))
    command=['mpirun','--oversubscribe','--bind-to','none','-np',str(ranks)]
    if args.nsys:
        command+=['nsys','profile','--trace=cuda,nvtx,mpi','--mpi-impl=openmpi','--sample=none',
            '--cpuctxsw=none','--export=sqlite','--output',str(case/'trace.rank%q{OMPI_COMM_WORLD_RANK}')]
    command += [str(exe),'run',primary]
    (case/'launch.json').write_text(json.dumps(dict(command=command,environment={k:v for k,v in env.items()
        if k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}),indent=2)+'\n')
    with (case/'run.log').open('w') as stream:
        subprocess.run(command,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    text=(case/'run.log').read_text()
    rows=re.findall(r'ASTR_GPU_RANK_RK_TIMING\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',text)
    records={(int(rank),int(step)):float(total) for rank,step,prepare_time,rk,total in rows}
    if len(records)!=ranks*args.steps:
        raise ValueError('missing rank/complete-step timing records')
    sums=[sum(records[rank,step] for step in range(args.warmup,args.steps)) for rank in range(ranks)]
    report=dict(status='pass',case=args.case,topology=args.topology,steps=args.steps,warmup=args.warmup,
        compact_sync=args.compact_sync,batch_rhs=args.batch_rhs,workspace=args.workspace,transport=args.transport,
        profiled=args.nsys,rank_rk_window_seconds=sums,rk_seconds_per_step=max(sums)/(args.steps-args.warmup),
        executable_sha256=hashlib.file_digest(exe.open('rb'),'sha256').hexdigest())
    (case/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
