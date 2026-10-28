"""Explicit incident-shock top and pressure outlet in the actual ASTR integrator."""
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from generate_compressible_blasius_profile import oblique_shock_ratios
from compare_flowstate import read_flowstate
from run_m12_mp7_nscbc_performance import runtime,mpi_command
from run_output_restart_validation import compare_fields

ROOT=Path(__file__).resolve().parents[2]
EXE=os.environ.get('ASTR_SBLI_JOINT_EXE')
MPIEXEC='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'
PRESSURE_RATIO=float(oblique_shock_ratios(2.,32.6)[0])
DT=2.5e-4
pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_JOINT_EXE to the root-CMake GPU/CPU executable')


def prepare(case,backend,scheme,steps,restore=None,checkpoint=True,outlet_pressure=PRESSURE_RATIO,backflow_mach=None,
            km=16):
    subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/prepare_s1_flatplate_case.py'),
        '--dst-case',str(case),'--use-gpu','t' if backend=='gpu' else 'f',
        '--im','64','--jm','48','--km',str(km),'--conschm',scheme,
        '--lchardecomp','t','--diffterm','t','--lfilter','f',
        '--x-min-bctype','11','--x-max-bctype','22','--upper-bctype','52',
        '--outlet-pressure',str(outlet_pressure),'--ninit','3',
        '--mach','2','--reynolds','296000','--reference-temperature','148.9',
        '--wall-temperature','.8','--x-min','-1','--x-max','10','--y-stretch','5',
        '--z-length','.25','--maxstep',str(steps-1),'--deltat',str(DT),
        '--feqchkpt','1','--feqlist','1'],check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/generate_compressible_blasius_profile.py'),
        '--grid',str(case/'datin/grid.flatplate.h5'),'--output',str(case/'datin/inlet.prof'),
        '--field-output',str(case/'datin/flowini3d.h5'),'--virtual-leading-edge','-2',
        '--mach','2','--reynolds','296000','--reference-temperature','148.9',
        '--wall-temperature','.8','--station-x','1','--density-mode','provided',
        '--pressure-mode','reconstruct','--shock-angle-deg','32.6',
        '--shock-x0','1','--shock-y0','1','--shock-y-min','0'],
        check=True,capture_output=True,text=True)
    if backflow_mach is not None:
        with h5py.File(case/'datin/grid.flatplate.h5') as mesh:
            x=mesh['x'][...]
        s=np.clip(((x-x.min())/(x.max()-x.min())-.8)/.1,0.,1.)
        eta=np.linspace(0.,1.,49)[None,:,None]
        weight=s*s*(3.-2.*s)*np.sin(np.pi*eta)**2
        weight[:,[0,-1],:]=0.
        with h5py.File(case/'datin/flowini3d.h5','r+') as field:
            u=field['u1'][...]
            field['u1'][...]=(1.-weight)*u-weight*backflow_mach*np.sqrt(field['t'][...])/2.
    (case/'outdat/new').mkdir(parents=True)
    (case/'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',restart_output='saved',exact_restart=t,
 host_budget_bytes=134217728,device_budget_bytes=134217728,buffer_bytes=4194304,device_reserve_bytes=0
/
&checkpoint
 enabled={'t' if checkpoint else 'f'},interval_steps=5,keep=2,initial_frame=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")


def launch(case,backend,scheme,steps,topology,restore=None,checkpoint=True,memcheck=False,shock_x='1',reject=None,
           transport='pinned',outlet_pressure=PRESSURE_RATIO,backflow_mach=None,km=16):
    prepare(case,backend,scheme,steps,restore,checkpoint,outlet_pressure,backflow_mach,km=km)
    ranks=int(np.prod(topology))
    env=runtime(','.join(map(str,topology)),transport=transport)
    env.pop('ASTR_WALL_BLOWING_MODE',None)
    env.update(ASTR_GPU_MP_POSITIVITY='off',ASTR_GPU_MP_SPLIT_CACHE='off',
        ASTR_NSCBC_FARFIELD_MODE='sbli_shock',ASTR_NSCBC_FARFIELD_SHOCK_X=shock_x,
        ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='32.6',ASTR_NSCBC_FARFIELD_RHO='1',
        ASTR_NSCBC_FARFIELD_U='1',ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',
        ASTR_NSCBC_FARFIELD_T='1')
    command=mpi_command(MPIEXEC,ranks,transport)
    if memcheck:
        command+=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']
    command += [str(Path(EXE).resolve()),'run','datin/input.flatplate']
    with (case/'run.log').open('w') as log:
        result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    text=(case/'run.log').read_text()
    if reject:
        assert reject in text and 'The job is done!' not in text,text[-6000:]
        return case
    assert result.returncode==0 and 'The job is done!' in text,text[-6000:]
    if memcheck:
        assert text.count('ERROR SUMMARY: 0 errors')==ranks,text[-6000:]
    return case


def fields(case,steps):
    with h5py.File(case/f'outdat/new/checkpoints/step{steps:012d}/state.h5') as state:
        q=np.stack([state[f'q{i:04d}'][...] for i in range(1,6)],axis=-1)
        primitives=np.stack([state[f'q{i:04d}'][...] for i in range(6,12)],axis=-1)
    assert np.isfinite(q).all()
    assert np.isfinite(primitives).all()
    rho=q[...,0]
    assert rho.min()>0
    velocity=q[...,1:4]/rho[...,None]
    internal=q[...,4]-.5*np.sum(q[...,1:4]*velocity,axis=-1)
    assert internal.min()>0
    pressure=.4*internal
    return {'q':q,'rho':rho,'velocity':velocity,'pressure':pressure,'temperature':pressure*5.6/rho,
            'primitive_cache':primitives}


def compare_cpu_gpu(cpu,gpu,steps=12):
    reference,actual=fields(cpu,steps),fields(gpu,steps)
    errors={}
    for name,expected in reference.items():
        errors[name]=float(np.max(np.abs(actual[name]-expected)/(1.+np.abs(expected))))
        assert errors[name]<=2.e-10,(name,errors[name])
    header,expected=read_flowstate(cpu)
    other,actual=read_flowstate(gpu)
    assert header==other and expected.shape==actual.shape
    assert np.isfinite(expected).all() and np.isfinite(actual).all()
    assert np.array_equal(expected[:,0],actual[:,0])
    for column,name in enumerate(header[1:],1):
        errors['statistics.'+name]=float(np.max(np.abs(actual[:,column]-expected[:,column])))
        assert errors['statistics.'+name]<=2.e-10,(name,errors['statistics.'+name])
    return errors


@pytest.mark.parametrize('scheme',['543e','743e'])
@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1),
                                     (4,1,1),(2,1,2),(1,2,2),(2,2,2)])
def test_same_phase_cpu_gpu(tmp_path,scheme,topology,record_property):
    steps=12
    cpu=launch(tmp_path/'cpu','cpu',scheme,steps,topology)
    gpu=launch(tmp_path/'gpu','gpu',scheme,steps,topology)
    errors=compare_cpu_gpu(cpu,gpu,steps)
    (tmp_path/'field_errors.json').write_text(json.dumps(errors,indent=2))
    record_property('scaled_maximum_field_errors',json.dumps(errors))


@pytest.mark.parametrize('scheme',['543e','743e'])
def test_cuda_aware_cpu_gpu(tmp_path,scheme):
    cpu=launch(tmp_path/'cpu','cpu',scheme,12,(2,1,1))
    gpu=launch(tmp_path/'gpu','gpu',scheme,12,(2,1,1),transport='device-aware')
    compare_cpu_gpu(cpu,gpu)


@pytest.mark.parametrize('scheme',['543e','743e'])
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_exact_checkpoint_continuation(tmp_path,scheme,backend):
    topology=(2,1,1)
    full=launch(tmp_path/'full',backend,scheme,12,topology)
    part=launch(tmp_path/'part',backend,scheme,5,topology)
    source=part/'outdat/new/checkpoints/step000000000005'
    tail=launch(tmp_path/'tail',backend,scheme,12,topology,restore=source)
    final='outdat/new/checkpoints/step000000000012'
    compare_fields(full/final/'state.h5',tail/final/'state.h5')
    assert (full/final/'control.bin').read_bytes()==(tail/final/'control.bin').read_bytes()
    fields(tail,12)


@pytest.mark.parametrize('scheme',['543e','743e'])
def test_shock_target_changes_actual_top_state(tmp_path,scheme):
    active=launch(tmp_path/'active','gpu',scheme,12,(1,1,1))
    control=launch(tmp_path/'control','gpu',scheme,12,(1,1,1),shock_x='20')
    a,b=fields(active,12),fields(control,12)
    x=np.linspace(-1.,10.,65)
    selected=(x>2.)&(x<5.)
    assert np.mean(a['pressure'][:,-1,selected]-b['pressure'][:,-1,selected])>0
    assert np.mean(a['velocity'][:,-1,selected,1]-b['velocity'][:,-1,selected,1])<0


@pytest.mark.parametrize('scheme',['543e','743e'])
def test_outlet_pressure_changes_subsonic_exit_state(tmp_path,scheme):
    active=launch(tmp_path/'active','gpu',scheme,12,(1,1,1))
    control=launch(tmp_path/'control','gpu',scheme,12,(1,1,1),outlet_pressure=1.)
    a,b=fields(active,12),fields(control,12)
    speed=b['velocity'][:,:,-1,0]
    sound=np.sqrt(1.4*b['pressure'][:,:,-1]/b['rho'][:,:,-1])
    subsonic=(speed>0)&(speed<sound)
    assert subsonic.any()
    assert np.mean((a['pressure'][:,:,-1]-b['pressure'][:,:,-1])[subsonic])>0


@pytest.mark.parametrize('scheme',['543e','743e'])
@pytest.mark.parametrize('backflow_mach',[.5,1.5])
def test_local_outlet_backflow_cpu_gpu(tmp_path,scheme,backflow_mach):
    cpu=launch(tmp_path/'cpu','cpu',scheme,12,(2,1,1),backflow_mach=backflow_mach)
    gpu=launch(tmp_path/'gpu','gpu',scheme,12,(2,1,1),backflow_mach=backflow_mach)
    compare_cpu_gpu(cpu,gpu)
    actual=fields(gpu,12)
    normal_mach=actual['velocity'][:,24,-1,0]/np.sqrt(1.4*actual['pressure'][:,24,-1]/actual['rho'][:,24,-1])
    if backflow_mach<1.:
        assert np.all((normal_mach>-1.)&(normal_mach<0.))
    else:
        assert np.all(normal_mach<-1.)


@pytest.mark.parametrize('scheme',['543e','743e'])
def test_shock_parameter_change_rejects_exact_restart(tmp_path,scheme):
    part=launch(tmp_path/'part','gpu',scheme,5,(1,1,1))
    launch(tmp_path/'changed','gpu',scheme,12,(1,1,1),
        restore=part/'outdat/new/checkpoints/step000000000005',shock_x='1.5',
        reject='boundary/driver configuration mismatch')


@pytest.mark.parametrize('scheme',['543e','743e'])
def test_memcheck(tmp_path,scheme):
    case=launch(tmp_path/'gpu','gpu',scheme,6,(2,1,1),memcheck=True)
    fields(case,6)
