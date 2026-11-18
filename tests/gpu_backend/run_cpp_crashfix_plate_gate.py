"""Native MP7-LD/NSCBC repair coupling, without formal output or restart admission."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import compare,logged_steps,namelist
from run_cpp_multi_tgv_gate import raw_field
from run_cpp_mp_positivity_gate import read_rhs


def history(case,rank):
    with (case/f'outdat/history.rank{rank:08d}.bin').open('rb') as stream:
        header=np.fromfile(stream,np.int32,5)
        values=np.fromfile(stream,np.int8)
    shape=tuple(header[:3]+2*header[3]+1)
    return header,values.reshape(shape,order='F')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['binary','reference','template','output','mpiexec']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--sync',choices=['explicit','dependency'],default='explicit')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp32')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not args.analyze_only:
        out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(__file__,out/'runner.py')
    binaries={name:out/name for name in ['reference_probe','cpp_probe']}
    for name,path in [('reference_probe',args.reference),('cpp_probe',args.binary)]:
        if args.analyze_only:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==hashlib.sha256(binaries[name].read_bytes()).hexdigest()
        else:shutil.copy2(path.resolve(),binaries[name])
    dtype=np.float32 if args.precision=='fp32' else np.float64
    limit=None if args.precision=='fp32' else 2e-10
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    base={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    base.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_COMPACT_TRANSPORT='pinned',ASTR_GPU_COMPACT_SYNC='explicit',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    report=dict(precision=args.precision,topology=topology,transport=args.transport,sync=args.sync,
        automatic=args.automatic,numeric_difference_limit=limit,checkpoint_tested=False,commands={},
        scope='16-cubed four-step existing incoming-only plate, seeded nonzero crashfix history; no physical validation',
        binaries={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in binaries.items()})
    report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['execution_runner_sha256']=hashlib.sha256((out/'runner.py').read_bytes()).hexdigest()

    def run(name):
        case=out/name
        if args.analyze_only:return case
        (case/'outdat/new').mkdir(parents=True)
        shutil.copytree(args.template.resolve(),case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755)
        text=(case/'datin/input.case').read_text()
        assert text.count("gpu_mp_positivity = 'flux'")==1
        (case/'datin/input.case').write_text(text.replace("gpu_mp_positivity = 'flux'","gpu_mp_positivity = 'off'"))
        timestep=(dict(mode='auto',initial_dt=1e-6,max_dt=2e-6,cfl_target=.5,growth_factor=1.1)
            if args.automatic else dict(mode='fixed',fixed_dt=1e-6))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),timestep=timestep,
            repair=dict(crashfix=True),statistics=dict(enabled=False),logging=dict(list_steps=1))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        env=base.copy()
        env['ASTR_VALIDATION_CRASHFIX_HISTORY_PREFIX']='outdat/history'
        if name!='reference':
            env.update(ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE=args.sync)
            if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.memcheck and name=='cpp':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--leak-check','full','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command += [str(binaries['reference_probe' if name=='reference' else 'cpp_probe']),'run']
        with (case/'run.log').open('w') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if name!='reference':assert (case/'run.log').read_text().count('backend=cuda')==args.ranks
        assert not list((case/'outdat/new/checkpoints').glob('step*'))
        report['commands'][name]=command
        return case

    original=run('reference')
    if not args.analyze_only:
        for path in original.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        original.chmod(0o555)
    current=run('cpp')
    errors={name:np.zeros(width) for name,width in [('pre_rhs',5),('pre_rhs_primitives',6),('post_update',5),('conv',5),('full',5)]}
    absolute={name:np.zeros_like(values) for name,values in errors.items()}
    for rank in range(args.ranks):
        for step in range(4):
            for stage in range(1,4):
                suffix=f'.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin'
                for name in errors:
                    if name in ['pre_rhs','pre_rhs_primitives','post_update']:
                        pairs=[]
                        for case in [original,current]:
                            primitive=name=='pre_rhs_primitives'
                            header,data=raw_field(case/'outdat'/('rhs.'+name+suffix),precision=args.precision,primitive=primitive)
                            h=header[3];data=data[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
                            wide=data.astype(np.float64)
                            assert np.isfinite(wide).all() and (wide[...,0]>0).all()
                            if primitive:assert (wide[...,[4,5]]>0).all()
                            else:
                                internal=wide[...,4]-.5*np.sum(wide[...,1:4]**2,axis=-1)/wide[...,0]
                                assert (internal>0).all()
                            pairs.append(wide)
                    else:pairs=[read_rhs(case/'outdat'/('rhs.'+name+suffix),dtype).astype(np.float64) for case in [original,current]]
                    errors[name]=np.maximum(errors[name],compare(pairs[1],pairs[0],limit=limit))
                    absolute[name]=np.maximum(absolute[name],np.max(abs(pairs[1]-pairs[0]),axis=(0,1,2)))
        old_h,old=history(original,rank);new_h,new=history(current,rank)
        assert np.array_equal(old_h,new_h) and old_h[4]==np.dtype(dtype).itemsize
        assert old.tobytes()==new.tobytes()
    assert sum(np.count_nonzero(history(current,rank)[1]) for rank in range(args.ranks))>0
    assert np.array_equal(logged_steps(current),logged_steps(original))
    for case in [original,current]:
        observers=re.findall(r'^CRASHFIX_HISTORY_OBSERVER_Q_WORDS_PASS rank=(\d+)$',(case/'run.log').read_text(),re.M)
        assert sorted(map(int,observers))==list(range(args.ranks))
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'));assert len(logs)==args.ranks
        for path in logs:
            text=path.read_text()
            assert 'ERROR SUMMARY: 0 errors' in text and 'LEAK SUMMARY: 0 bytes leaked in 0 allocations' in text
    report.update(phases_scaled={k:v.tolist() for k,v in errors.items()},phases_maxabs={k:v.tolist() for k,v in absolute.items()},
        critical_history_equal=True,nonzero_history_words=int(sum(np.count_nonzero(history(current,rank)[1]) for rank in range(args.ranks))),
        clocks_equal=True,history_observer_q_words_equal=True,memcheck=args.memcheck)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
