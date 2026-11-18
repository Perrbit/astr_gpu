"""N7: original CPU/CUDA Fortran filtered bc52 plate, with real MPI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from prepare_upwind_components import prepare_flat
from run_cpp_tgv_gate import convert_case,namelist,compare,logged_steps,snapshot
from run_cpp_multi_tgv_gate import raw_field


def read_phase(path,kind):
    if kind in ('pre_rhs','pre_rhs_primitives','post_update'):
        header,value=raw_field(path,primitive=kind=='pre_rhs_primitives');h=header[3]
        return value[tuple(slice(h,h+n+1) for n in header[:3])+(slice(None),)]
    with path.open('rb') as f:
        h=np.fromfile(f,np.int32,4)
        return np.fromfile(f,np.float64).reshape(tuple(h[:3]+1)+(h[3],),order='F')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','mpiexec','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--policy',choices=('compatibility','incoming_only','sbli_shock','nonreflecting'),default='compatibility')
    parser.add_argument('--joint-family',choices=('keep','me6','mp7ld_compact'))
    parser.add_argument('--compact',action='store_true')
    parser.add_argument('--filter',choices=('off','explicit10','compact10'),default='off')
    parser.add_argument('--mpi-axes',nargs='*',choices=tuple('xyz'),default=['y','z'])
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='3')
    report=dict(policy=args.policy,joint_family=args.joint_family,compact=args.compact,filter=args.filter,cases={})
    previous=None
    cases=[('cpu_np1',False,1,0),('gpu_np1',True,1,0)]
    cases += [(f'gpu_np2_{axis}',True,2,'xyz'.index(axis)) for axis in args.mpi_axes]
    for name,gpu,ranks,axis in cases:
        case=out/name;prepare_flat(case,gpu);(case/'outdat/new').mkdir()
        groups=convert_case((case/'datin/input.flatplate').read_text())
        groups['numerics'].update(conschm='543e',difschm='643e',recon_schem=3,lchardecomp=False,lfilter=False,
            convection_form='standard',viscous_form='standard',filter_scheme='explicit10')
        groups['boundary']['bctype']=[11,21,41,52,1,1]
        if args.joint_family:
            groups['boundary']['bctype'][1]=22
            if args.joint_family=='mp7ld_compact':
                groups['numerics'].update(conschm='743e',difschm='643c',recon_schem=5,lchardecomp=True)
            else:
                groups['numerics'].update(conschm='643e',convection_form='keep' if args.joint_family=='keep' else 'standard',
                    viscous_form='me6_opti' if args.joint_family=='me6' else 'standard')
        if args.compact:groups['numerics'].update(conschm='643c',difschm='643c')
        if args.filter!='off':groups['numerics'].update(lfilter=True,filter_scheme=args.filter)
        groups['boundary_options']=dict(nscbc_farfield_mode=args.policy,profile_inflow_mode='complete_state')
        if args.policy=='sbli_shock':
            groups['boundary_options'].update(nscbc_farfield_shock_x=5.,nscbc_farfield_shock_angle_deg=36.)
        groups['precision_options']=dict(gpu_precision_mode='fp64')
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),
            timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        topology=[1,1,1];topology[axis]=ranks
        binary=args.reference.resolve()/'bin'/('astr_cuda_fortran' if gpu else 'astr_cpu')
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks),str(binary),'run'],
                cwd=case,env=env|dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology))),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        if previous is None:
            previous=case
            report['cases'][name]=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest());continue
        errors={}
        for phase in ('pre_rhs','pre_rhs_primitives','post_source','post_update'):
            paths=sorted((previous/'outdat').glob(f'rhs.{phase}.*.bin'))
            if len(paths)!=12:raise AssertionError(('missing phase',phase,len(paths)))
            maximum=0.
            for file in paths:
                whole=read_phase(file,phase)
                for rank in range(ranks):
                    actual=read_phase(case/'outdat'/file.name.replace('rank00000000',f'rank{rank:08d}'),phase)
                    body=[slice(None)]*3
                    body[axis]=slice(rank*(16//ranks),rank*(16//ranks)+actual.shape[axis])
                    maximum=max(maximum,max(compare(actual,whole[tuple(body)+(slice(None),)])))
            errors[phase]=maximum
        scalars=compare(np.loadtxt(case/'flowstate.dat',skiprows=1),np.loadtxt(previous/'flowstate.dat',skiprows=1))
        if not np.array_equal(logged_steps(case),logged_steps(previous)):raise AssertionError('filtered-top clocks changed')
        report['cases'][name]=dict(topology=topology,phases=errors,scalars= scalars)
    shutil.copyfile(__file__,out/'runner.py');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
