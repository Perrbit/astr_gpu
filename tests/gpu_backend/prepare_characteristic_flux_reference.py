"""Freeze two periodic or bc50 Shu-Osher steps from original FP64/MP3 executables."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import compare,namelist,snapshot,logged_steps,convert_case


def sensor(path):
    with path.open('rb') as stream:
        upper=np.fromfile(stream,np.int32,3)
        nodes=int(np.prod(upper+1))
        return np.fromfile(stream,np.float64,nodes),np.fromfile(stream,np.int8,nodes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','template','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--boundary',choices=('periodic','xphysical'),default='periodic')
    parser.add_argument('--original-executable',type=Path,help='explicit dual CPU/CUDA-Fortran executable for a current corrected reference')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'preparation.py')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_MP_POSITIVITY='off',ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='2',ASTR_VALIDATION_GRADIENTS='1',
        ASTR_GEOMETRY_DUMP='outdat/geometry.h5')
    report=dict(scope='16-cubed periodic Shu-Osher interface fixture, Mach3, dt1e-5, two RK3 steps; not a physical benchmark',
        original_binaries={},candidate_field_atol=2e-6,candidate_statistics_atol=2e-6)
    if args.boundary=='xphysical':
        report['scope']='16-cubed Shu-Osher xphysical bc50 interface fixture, Mach3, dt1e-5, two RK3 steps; not a physical benchmark'
    report['boundary']=args.boundary
    fields={};scalars={};cases={}
    for name,binary,gpu,mixed in (('cpu','astr_cpu',False,False),('old_gpu_fp64','astr_cuda_fortran',True,False),
        ('old_gpu_mixed','astr_cuda_fortran',True,True)):
        case=out/name;cases[name]=case;(case/'datin').mkdir(parents=True);(case/'outdat/new').mkdir(parents=True)
        groups=convert_case(args.template.read_text())
        groups['case_options'].update(flowtype='shuosher',use_gpu=gpu)
        groups['mesh'].update(ia=16,ja=16,ka=16,lihomo=args.boundary=='periodic',ljhomo=True,lkhomo=True)
        groups['physics'].update(nondimen=True,diffterm=False,mach=3.,num_species=0)
        groups['numerics'].update(conschm='543e',difschm='643e',convection_form='standard',viscous_form='standard',
            recon_schem=3,lchardecomp=True,lfilter=False,filter_scheme='explicit10',shkcrt=.001)
        groups['boundary']['bctype']=[1]*6 if args.boundary=='periodic' else [50,50,1,1,1,1]
        groups['precision_options']=dict(gpu_precision_mode='mixed_workspace' if mixed else 'fp64')
        if mixed:groups['precision_options']['gpu_mixed_candidate']='characteristic_flux'
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=2),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        executable=(args.original_executable if args.original_executable else args.reference/'bin'/binary).resolve()
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(executable),'run'],
                cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,check=True)
        if mixed and 'ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux' not in (case/'run.log').read_text():
            raise AssertionError('old characteristic candidate was not selected')
        q=snapshot(case/'outdat/rhs.post_update.step00000001.rk03.rank00000000.bin')
        internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
        if not np.isfinite(q).all() or not (q[...,0]>0).all() or not (internal>0).all():
            raise AssertionError('original periodic shock state')
        fields[name]=q;scalars[name]=np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
        report['original_binaries'][name]=dict(path=str(executable),sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
    report['cpu_old_gpu_fp64']=compare(fields['cpu'],fields['old_gpu_fp64'])
    field=np.max(abs(fields['old_gpu_mixed']-fields['old_gpu_fp64']),axis=(0,1,2))
    stats=np.max(abs(scalars['old_gpu_mixed']-scalars['old_gpu_fp64']),axis=0)
    if not np.isfinite(stats).all() or max(field)>2e-6 or max(stats)>2e-6:
        raise AssertionError('old candidate exceeds its frozen periodic MP3-A field/statistics gates')
    if not np.array_equal(logged_steps(cases['old_gpu_fp64'],2),logged_steps(cases['old_gpu_mixed'],2)):
        raise AssertionError('old candidate step clock differs')
    observations=[]
    for step in range(2):
        for stage in (1,2,3):
            name=f'outdat/rhs.sensor.step{step:08d}.rk{stage:02d}.rank00000000.bin'
            raw,mask=sensor(cases['old_gpu_fp64']/name);candidate,marks=sensor(cases['old_gpu_mixed']/name)
            observations.append(dict(step=step,stage=stage,raw_max_abs=float(np.max(abs(raw-candidate))),
                raw_bits_equal=raw.tobytes()==candidate.tobytes(),mask_mismatches=int(np.count_nonzero(mask!=marks)),
                marked=int(np.count_nonzero(mask)),unmarked=int(np.count_nonzero(mask==0))))
    if not observations[0]['raw_bits_equal'] or observations[0]['mask_mismatches']:
        raise AssertionError('same initial state changed the old MP3 sensor')
    report.update(field_max_abs=field.tolist(),statistics_max_abs=stats.tolist(),sensors=observations)
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
