"""Ordinary C++ MP positivity versus the frozen repaired CUDA Fortran RK path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import compare,logged_steps,namelist


def read_rhs(path,dtype=np.float64):
    with path.open('rb') as stream:
        shape=np.fromfile(stream,np.int32,4)
        return np.fromfile(stream,dtype).reshape(tuple(shape[:3]+1)+(shape[3],),order='F')


def read_state(path,dtype):
    with path.open('rb') as stream:
        im,jm,km,hm,nq=np.fromfile(stream,np.int32,5)
        values=np.fromfile(stream,dtype)
    full=values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    return full[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--compact-transport',choices=['pinned','device-aware'],default='pinned')
    parser.add_argument('--compact-sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    dtype=np.float32 if args.precision=='fp32' else np.float64
    limit=None if args.precision=='fp32' else 2e-10
    output=args.output.resolve();reference=args.reference.resolve();case=output/'cpp'
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE=args.sync,
        ASTR_GPU_COMPACT_TRANSPORT=args.compact_transport,ASTR_GPU_COMPACT_SYNC=args.compact_sync,
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
    if args.memcheck:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
        '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
    if not args.analyze_only:
        output.mkdir(parents=True,exist_ok=False);case.mkdir()
        shutil.copytree(reference/'datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',host_budget_bytes=-1,
            device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(args.binary.resolve()),'run'],cwd=case,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    log=(case/'run.log').read_text()
    assert 'ASTR_GPU_CPP mp_positivity=flux budget=roundoff-interior-v1' in log
    compact_active=bool(re.search(r"difschm\s*=\s*'643c'",(case/'datin/input.case').read_text()))
    if compact_active:
        compact=re.search(r'ASTR_CPP_COMPACT transport/batch/explicit=(\d+)\s+(\d+)\s+(\d+)',log)
        assert compact and int(compact[1])==int(args.compact_transport=='device-aware') and int(compact[3])==int(args.compact_sync=='explicit')
    high_open='sbli-grcbc-h1 high-order boundary chain' in (reference/'run.log').read_text()
    if high_open:
        assert 'sbli-grcbc-h1 high-order boundary chain' in log
        assert 'ASTR_GPU_CPP high_open=production-h1' in log
    counts=re.findall(r'ASTR_CPP_MP_POSITIVITY step=(\d+) stage=(\d+) limited_rank_nodes=(\d+)',log)
    old_counts=re.findall(r'ASTR_MP_POSITIVITY step=\s*(\d+)\s+stage=\s*(\d+)\s+limited_rank_nodes=\s*(\d+)',
        (reference/'run.log').read_text())
    assert len(counts)==len(old_counts)==12 and [row[:2] for row in counts]==[row[:2] for row in old_counts]
    if args.precision=='fp64':assert counts==old_counts
    fields=np.zeros(5);fields_absolute=np.zeros(5)
    phases={name:np.zeros(5) for name in ['conv','post_source','mp_limited']}
    phases_absolute={name:np.zeros(5) for name in phases}
    for rank in range(args.ranks):
        for step in range(4):
            for stage in range(1,4):
                suffix=f'.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                a=read_state(reference/'outdat'/('rhs.post_update'+suffix),dtype)
                b=read_state(case/'outdat'/('rhs.post_update'+suffix),dtype)
                fields=np.maximum(fields,compare(b.astype(np.float64),a.astype(np.float64),limit=limit))
                fields_absolute=np.maximum(fields_absolute,np.max(abs(b.astype(np.float64)-a.astype(np.float64)),axis=(0,1,2)))
                for state in [a,b]:
                    wide=state.astype(np.float64)
                    e=wide[...,4]-.5*np.sum(wide[...,1:4]**2,axis=-1)/wide[...,0]
                    assert np.isfinite(state).all() and (state[...,0]>0).all() and np.isfinite(e).all() and (e>0).all()
                for name in phases:
                    old='full' if name=='mp_limited' else name
                    a=read_rhs(reference/'outdat'/('rhs.'+old+suffix),dtype)
                    b=read_rhs(case/'outdat'/('rhs.'+name+suffix),dtype)
                    phases[name]=np.maximum(phases[name],compare(b.astype(np.float64),a.astype(np.float64),limit=limit))
                    phases_absolute[name]=np.maximum(phases_absolute[name],np.max(abs(b.astype(np.float64)-a.astype(np.float64)),axis=(0,1,2)))
    scalars=compare(np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2),
        np.loadtxt(reference/'flowstate.dat',skiprows=1,ndmin=2),limit=limit)
    assert np.array_equal(logged_steps(case),logged_steps(reference))
    memory=[]
    if args.memcheck:
        logs=list(case.glob('memcheck.*.log'))
        assert len(logs)==args.ranks
        for path in logs:
            text=path.read_text()
            errors=re.search(r'^========= ERROR SUMMARY: (\d+) errors$',text,re.M)
            leaks=re.search(r'^========= LEAK SUMMARY: (\d+) bytes leaked in (\d+) allocations$',text,re.M)
            assert errors and leaks
            memory.append(dict(file=path.name,errors=int(errors[1]),leaked_bytes=int(leaks[1]),allocations=int(leaks[2])))
    report=dict(binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),topology=topology,
        precision=args.precision,real_bytes=np.dtype(dtype).itemsize,numeric_difference_limit=limit,
        transport=args.transport,sync=args.sync,limited_counts=counts,stage_field_scaled=fields.tolist(),
        stage_field_maxabs=fields_absolute.tolist(),
        compact_active=compact_active,compact_transport=args.compact_transport,compact_sync=args.compact_sync,
        phases_scaled={name:errors.tolist() for name,errors in phases.items()},scalar_scaled=scalars,
        phases_maxabs={name:errors.tolist() for name,errors in phases_absolute.items()},
        clocks_equal=True,limited_counts_equal=counts==old_counts,checkpoint_tested=False,memcheck=args.memcheck,
        memcheck_results=memory,accepted=all(row['errors']==0 and row['leaked_bytes']==0 for row in memory),
        high_open=high_open,
        scope='Ordinary MP7-LD plate short window; active anti-diffusive stage and restart remain separate')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['accepted']:raise AssertionError('numeric comparison passed; memory/tool gate failed, see recorded summaries')


if __name__=='__main__':main()
