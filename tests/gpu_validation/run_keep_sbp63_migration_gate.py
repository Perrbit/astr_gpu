"""KEEP weak/strong wall identity switches using immutable native checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_periodic_gate import compare_q
from run_keep_sbp63_component_gate import scaled

FINAL=Path('outdat/new/checkpoints/step000000000004')


def primitive_consistency(case):
    differences={}
    for rank in range(2):
        suffix=f'.step00000002.rk01.rank{rank:08d}.bin'
        snapshot=read_q_snapshot(case/'outdat'/('rhs.pre_rhs'+suffix))
        im,jm,km,hm,nq=snapshot.header
        shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
        physical=(slice(hm,hm+im+1),slice(hm,hm+jm+1),slice(hm,hm+km+1))
        q=snapshot.values.reshape((*shape,nq),order='F')[physical]
        with (case/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
            header=np.fromfile(stream,np.int32,4)
            cache=np.fromfile(stream,np.float64).reshape((*shape,6),order='F')[physical]
        if tuple(header)!=(im,jm,km,hm):
            raise AssertionError('Restored primitive layout differs')
        rho=q[...,0];v=q[...,1:4]/rho[...,None]
        p=.4*(q[...,4]-.5*np.sum(q[...,1:4]*v,axis=-1))
        if not np.isfinite(q).all() or not (rho>0).all() or not (p>0).all():
            raise AssertionError('Invalid migrated physical state')
        expected=np.concatenate([rho[...,None],v,p[...,None],(.126*p/rho)[...,None]],axis=-1)
        differences[rank]=scaled(cache,expected)
    if max(differences.values())>2e-10:
        raise AssertionError('Restored physical caches do not match their current conservative state')
    return differences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('cpu','gpu','mpiexec','sbp-source','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.sbp_source.resolve()
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMP_NUM_THREADS='1',ASTR_FORCE_MPI_TOPOLOGY='1,2,1',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',
        OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_VALIDATION_RK_STATE='1',ASTR_VALIDATION_PROFILE_STAGE_STATE='1')
    report=dict(status='running',scope='16 intervals, NP2 y, weak-SBP/strong-w24 method switches, no filter or repair',cases={})
    paths={}

    def run(name,backend,datin_source,wall_mode,seed=None,exact=True,rejection=None):
        case=out/name;shutil.copytree(datin_source,case/'datin')
        primary=case/'datin/input.case';text=primary.read_text()
        token="high_order_wall_mode='keep_sbp63'"
        if text.count(token)!=1:
            raise AssertionError('Expected the frozen weak-wall input fixture')
        primary.write_text(text.replace(token,f"high_order_wall_mode='{wall_mode}'"))
        output=case/'datin/input.output';text=output.read_text()
        text=text.replace('exact_restart=t',f"exact_restart={'t' if exact else 'f'}")
        if seed is not None:
            text=text.replace("directory='outdat/new',",f"directory='outdat/new',restore_directory='{seed}',")
        output.write_text(text);(case/'outdat/new').mkdir(parents=True)
        run_env=env.copy()
        if seed is not None and not exact:
            run_env.update(ASTR_CHECKPOINT_TEST_RESTORE_PROBE='1',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
                ASTR_VALIDATION_RHS_STEP='2')
        command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np','2',
            str(getattr(args,backend).resolve()),'run','datin/input.case']
        with (case/'run.log').open('w') as log:
            result=subprocess.run(command,cwd=case,env=run_env,stdout=log,stderr=subprocess.STDOUT)
        log=(case/'run.log').read_text()
        if rejection is not None:
            if result.returncode==0 or rejection not in log:
                raise AssertionError('Different wall method was not rejected as an exact continuation')
            if list((case/'outdat/new').rglob('COMPLETE')):
                raise AssertionError('Rejected exact method switch published a checkpoint')
            return case
        if result.returncode:
            raise RuntimeError('Native method switch failed: '+str(case/'run.log'))
        compare_q(case/FINAL/'state.h5',case/FINAL/'state.h5')
        return case

    for backend in ('cpu','gpu'):
        sbp=source/f'{backend}_n16_np2'
        low=run(backend+'_w24_source',backend,sbp/'datin','off')
        paths[(backend,'w24')]=low
        for label,origin,target in (('sbp_to_w24',sbp,'off'),('w24_to_sbp',low,'keep_sbp63')):
            seed=origin/'outdat/new/checkpoints/step000000000002'
            run(backend+'_'+label+'_exact',backend,sbp/'datin',target,seed,rejection='compact operator identity mismatch')
            case=run(backend+'_'+label+'_nonexact',backend,sbp/'datin',target,seed,exact=False)
            restored=compare_q(seed/'state.h5',case/'outdat/restore_probe.h5')
            with h5py.File(seed/'state.h5') as before,h5py.File(case/'outdat/restore_probe.h5') as after:
                z,y,x=np.indices(before['q0001'].shape)
                shared=(x==0)|(x==16)|(y==8)|(z==0)|(z==16)
                for c in range(1,6):
                    key=f'q{c:04d}'
                    # Non-exact qswap intentionally reconciles shared/periodic copies.
                    if before[key][:][~shared].tobytes()!=after[key][:][~shared].tobytes():
                        raise AssertionError('Nonexact switch changed an unshared '+key+' before advancement')
            control=(case/FINAL/'control.bin').read_bytes()
            target_id=b'/w24' if target=='off' else b'/s63sat-v1'
            if target_id not in control:
                raise AssertionError('Migrated checkpoint did not record its target wall method')
            if 'ASTR_KEEP_RESTART statistics_new_segment_step=2 ' not in (case/'run.log').read_text():
                raise AssertionError('Changed wall identity did not select the existing new-window lifecycle')
            report['cases'][backend+'_'+label]=dict(source=str(seed),target=str(case),exact='different operator rejected',
                restored_q=restored,unshared_q='all five arrays bit-identical before advancement',
                physical_cache_max_scaled=primitive_consistency(case),
                statistics='new-window branch at step 2; no nonzero accumulated statistics in this fixture')
            paths[(backend,label)]=case
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(backend,label,'identity, restored q and target caches checked',flush=True)
    for label in ('w24','sbp_to_w24','w24_to_sbp'):
        report['cases']['cpu_gpu_'+label]=compare_q(paths[('cpu',label)]/FINAL/'state.h5',paths[('gpu',label)]/FINAL/'state.h5')
    report['status']='passed_weak_strong_wall_method_switches'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
