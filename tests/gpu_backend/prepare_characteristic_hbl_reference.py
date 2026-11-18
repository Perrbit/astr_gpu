"""Original MP3-HBL1 input/field reference; no replacement CFD integrator."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np

from run_cpp_tgv_gate import compare,convert_case,namelist,logged_steps

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from compare_shock_sensor import read_sensor_dump

FIELD_NAMES=('density','velocity_x','velocity_y','velocity_z','pressure','temperature')
FIELD_ATOL=5e-7
STAT_ATOL=SENSOR_ATOL=1e-12
CONST6=1.0/(1.4-1.0)


def read_volume(case):
    path=case/'outdat/new/fields/segment00000000/step000000000002/data.h5'
    with h5py.File(path) as data:
        if data['metadata'][1]!=2 or data['metadata'][2:5].view(np.float64)[0]!=2e-5:
            raise AssertionError('HBL complete-step volume identity')
        if any(data[name].dtype!=np.dtype('float64') for name in FIELD_NAMES):
            raise AssertionError('MP3-HBL1 publication must retain FP64 physical fields')
        return np.stack([data[name][...] for name in FIELD_NAMES],axis=-1)


def conservative(p):
    rho=p[...,0];velocity=p[...,1:4];pressure=p[...,4]
    return np.concatenate((rho[...,None],rho[...,None]*velocity,
        (CONST6*pressure+.5*rho*np.sum(velocity**2,axis=-1))[...,None]),axis=-1)


def positive(p):
    if not np.isfinite(p).all() or not all((p[...,c]>0).all() for c in (0,4,5)):
        raise AssertionError('HBL finite positive rho/p/T')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','mpiexec','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'preparation.py')
    template=out/'template'
    command=[sys.executable,str(ROOT/'tests/gpu_validation/prepare_s1_flatplate_case.py'),
        '--dst-case',str(template),'--use-gpu','f','--im','192','--jm','192','--km','8',
        '--diffterm','t','--lchardecomp','t','--shock-threshold','.001','--conschm','543e',
        '--reynolds','1.83052e6','--mach','5','--reference-temperature','226.65',
        '--wall-temperature','5.191440547760865','--upper-bctype','51','--x-min-bctype','11',
        '--ninit','3','--x-min','-1','--x-max','10','--y-stretch','5','--z-length','.25',
        '--maxstep','2','--feqchkpt','2','--sponge-im','0','--deltat','1e-5']
    subprocess.run(command,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    command=[sys.executable,str(ROOT/'tests/gpu_validation/generate_compressible_blasius_profile.py'),
        '--grid',str(template/'datin/grid.flatplate.h5'),'--output',str(template/'datin/inlet.prof'),
        '--mach','5','--reynolds','1.83052e6','--reference-temperature','226.65',
        '--wall-temperature','5.191440547760865','--station-x','1','--density-mode','provided',
        '--pressure-mode','provided','--field-output',str(template/'datin/flowini3d.h5'),
        '--virtual-leading-edge','-2','--shock-angle-deg','35','--shock-x0','-1','--shock-y0','.18',
        '--shock-y-min','.02','--profile-oblique-shock','--profile-shock-y-min','.18','--field-oblique-shock']
    with (out/'profile_generation.log').open('w') as log:
        subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,text=True)
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_VALIDATION_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_SHOCK_SENSOR_DUMP='outdat/initial_sensor.dat',ASTR_GPU_MP_POSITIVITY='off')
    fields={};scalars={};sensors={};report=dict(scope='Original MP3-HBL1 controlled numerical fixture, not steady HBL/SBLI physics',
        grid=[192,192,8],steps=2,dt=1e-5,mach=5.,reynolds=1.83052e6,reference_temperature=226.65,
        wall_temperature=5.191440547760865,field_atol=FIELD_ATOL,statistics_atol=STAT_ATOL,sensor_atol=SENSOR_ATOL,
        binaries={})
    for name,binary,gpu,mixed in (('cpu','astr_cpu',False,False),('old_gpu_fp64','astr_cuda_fortran',True,False),
        ('old_gpu_mixed','astr_cuda_fortran',True,True)):
        case=out/name;case.mkdir();shutil.copytree(template/'datin',case/'datin')
        (case/'outdat/new').mkdir(parents=True)
        groups=convert_case((template/'datin/input.flatplate').read_text())
        groups['case_options']['use_gpu']=gpu
        groups['numerics'].update(convection_form='standard',viscous_form='standard',filter_scheme='explicit10')
        groups['precision_options']=dict(gpu_precision_mode='mixed_workspace' if mixed else 'fp64',gpu_mp_positivity='off')
        if mixed:groups['precision_options']['gpu_mixed_candidate']='characteristic_flux'
        (case/'datin/input.case').write_text(namelist(groups))
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=2),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),checkpoint=dict(enabled=False),
            volume=dict(enabled=True,mode='steps',interval_steps=2,initial_frame=False,final_frame=False),slices=dict(enabled=False))))
        executable=(args.reference/'bin'/binary).resolve()
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',str(executable),'run'],
                cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if mixed and 'ASTR_GPU_ACTIVE_MIXED_WORKSPACE=characteristic_flux' not in (case/'run.log').read_text():
            raise AssertionError('old HBL MP3 candidate was not active')
        fields[name]=read_volume(case);positive(fields[name])
        scalars[name]=np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.isfinite(scalars[name]).all():raise AssertionError('original HBL scalar state')
        sensors[name]=read_sensor_dump(case/'outdat/initial_sensor.dat')
        report['binaries'][name]=dict(path=str(executable),sha256=hashlib.sha256(executable.read_bytes()).hexdigest())
    report['cpu_fp64_scaled']=compare(fields['old_gpu_fp64'],fields['cpu'])
    report['cpu_fp64_q_scaled']=compare(conservative(fields['old_gpu_fp64']),conservative(fields['cpu']))
    field=np.max(abs(fields['old_gpu_mixed']-fields['old_gpu_fp64']),axis=(0,1,2))
    q=np.max(abs(conservative(fields['old_gpu_mixed'])-conservative(fields['old_gpu_fp64'])),axis=(0,1,2))
    stat=np.max(abs(scalars['old_gpu_mixed']-scalars['old_gpu_fp64']),axis=0)
    raw=float(np.max(abs(sensors['old_gpu_mixed'].sensor-sensors['old_gpu_fp64'].sensor)))
    masks=int(np.count_nonzero(sensors['old_gpu_mixed'].mask!=sensors['old_gpu_fp64'].mask))
    if max(field)>FIELD_ATOL or max(q)>FIELD_ATOL or max(stat)>STAT_ATOL or raw>SENSOR_ATOL or masks:
        raise AssertionError('original MP3-HBL1 frozen gates failed')
    if not np.array_equal(logged_steps(out/'old_gpu_fp64',2),logged_steps(out/'old_gpu_mixed',2)):
        raise AssertionError('original HBL complete-step clock')
    report.update(candidate_primitive_max_abs=field.tolist(),candidate_q_max_abs=q.tolist(),
        candidate_statistics_max_abs=stat.tolist(),candidate_initial_sensor_max_abs=raw,mask_mismatches=masks)
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
