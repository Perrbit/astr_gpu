"""Freeze existing wall-forcing trajectories before new C++ consumers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_high_order_wall_runtime import prepare
from prepare_upwind_components import prepare_flat
from run_cpp_tgv_gate import compare,snapshot,logged_steps,namelist,convert_case
from prepare_ldc_reference import compare_stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--family',choices=['channel','plate'],required=True)
    parser.add_argument('--forcing',choices=['hash','modal','legacy_random'],required=True)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--axis',choices=list('xyz'),default='x')
    parser.add_argument('--joint-family',choices=('keep','me6','mp7ld_compact','high_open'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.forcing=='legacy_random' and args.family!='plate':parser.error('the original legacy generator requires bl')
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    cases={}
    for backend,binary in [('cpu','astr_cpu'),('gpu','astr_cuda_fortran')]:
        case=out/backend;cases[backend]=case;dt=1e-5 if args.family=='channel' else 1e-6
        if args.family=='channel':
            prepare(case,backend,4,None,grid=(16,16,16),filtering=False)
            primary='input.chl';xa,xb,xc=1.,3.,5.
        else:
            prepare_flat(case,backend=='gpu');primary='input.flatplate';xa,xb,xc=2.,5.,8.
        groups=convert_case((case/'datin'/primary).read_text())
        groups['numerics'].update(conschm='643e' if args.family=='channel' else '543e',difschm='643e',
            convection_form='standard',viscous_form='standard',recon_schem=3,lchardecomp=False,lfilter=False,filter_scheme='explicit10')
        if args.family=='plate':groups['boundary']['bctype'][3]=51
        groups['precision_options']=dict(gpu_precision_mode='fp64',gpu_mp_positivity='off')
        groups['boundary_options']=dict(high_order_wall_mode='off',
            wall_blowing_mode='legacy_random' if args.forcing=='legacy_random' else 'current')
        if args.joint_family:
            if args.joint_family in ('mp7ld_compact','high_open'):
                groups['numerics'].update(conschm='743e',difschm='643e' if args.joint_family=='high_open' else '643c',
                    recon_schem=5,lchardecomp=True)
            else:
                groups['numerics'].update(conschm='643e',difschm='643e',recon_schem=3,lchardecomp=False,
                    convection_form='keep' if args.joint_family=='keep' else 'standard',
                    viscous_form='me6_opti' if args.joint_family=='me6' else 'standard')
            groups['boundary']['bctype']=[11,22,41,52,1,1]
            groups['boundary_options'].update(nscbc_farfield_mode='incoming_only' if args.joint_family=='mp7ld_compact' else 'nonreflecting',
                profile_inflow_mode='complete_state')
            if args.joint_family=='high_open':
                groups['boundary_options'].update(nscbc_farfield_mode='sbli_grcbc',nscbc_farfield_shock_x=15.,
                    nscbc_farfield_shock_angle_deg=40.,nscbc_farfield_rho=1.,nscbc_farfield_u=1.,
                    nscbc_farfield_v=0.,nscbc_farfield_w=0.,nscbc_farfield_t=1.)
        groups['transport_options']=dict(channel_force_mode='fixed',channel_force_fixed=0.)
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/wallbs.dat').write_text(f'# wall forcing\n# amplitude beta xa xb xc\n1.d-4 2.d0 {xa} {xb} {xc}\n# time/span modes\n{2 if args.forcing=="modal" else 0} 2\n')
        if args.forcing=='modal':
            (case/'datin/wallbs_phase.dat').write_text(' '.join(str(.03*i) for i in range(1,16))+'\n')
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),
            timestep=dict(mode='fixed',fixed_dt=dt),logging=dict(list_steps=1),repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks),
                str((args.reference/'bin'/binary).resolve()),'run'],cwd=case,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    report=dict(family=args.family,forcing=args.forcing,joint_family=args.joint_family,topology=topology,dt=dt,steps=4,
        binaries={name:hashlib.sha256((args.reference/'bin'/name).read_bytes()).hexdigest() for name in ('astr_cpu','astr_cuda_fortran')},
        stages=compare_stages(cases['cpu'],cases['gpu'],ranks=args.ranks),updated={})
    for rank in range(args.ranks):
        paths=sorted((cases['cpu']/'outdat').glob(f'rhs.post_update.*.rank{rank:08d}.bin'))
        if len(paths)!=12:raise AssertionError('missing original four-step wall trajectory')
        report['updated'][str(rank)]=max(max(compare(snapshot(cases['gpu']/'outdat'/p.name),snapshot(p))) for p in paths)
    report['scalars']=compare(np.loadtxt(cases['gpu']/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(cases['cpu']/'flowstate.dat',skiprows=1,ndmin=2))
    if not np.array_equal(logged_steps(cases['cpu']),logged_steps(cases['gpu'])):raise AssertionError('forcing reference clock')
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,out/'preparation.py')
    for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
