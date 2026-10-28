"""Bounded complete-RHS GRCBC integration; not a developed-SBLI certificate."""
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from run_m12_mp7_nscbc_performance import runtime,mpi_command
from run_output_restart_validation import compare_fields
from test_sbli_joint_nscbc import prepare,fields,compare_cpu_gpu,MPIEXEC

EXE=os.environ.get('ASTR_SBLI_GRCBC_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake CPU/GPU executable')


def launch(case,backend,steps=12,topology=(1,1,1),restore=None,mode='sbli_grcbc',reject=None,memcheck=False,
           transport='pinned',checkpoint_interval=5,diagnostics=False,viscous=True):
    prepare(case,backend,'743e',steps,restore)
    if not viscous:
        path=case/'datin/input.flatplate'
        path.write_text(path.read_text().replace('\nt,t,f,t,f,f,f,f,','\nt,f,f,t,f,f,f,f,',1))
    if checkpoint_interval!=5:
        configuration=case/'datin/input.output'
        configuration.write_text(configuration.read_text().replace('interval_steps=5',
                                   f'interval_steps={checkpoint_interval}'))
    env=runtime(','.join(map(str,topology)),transport=transport)
    env.pop('ASTR_WALL_BLOWING_MODE',None)
    env.update(ASTR_GPU_MP_POSITIVITY='off',ASTR_GPU_MP_SPLIT_CACHE='off',
        ASTR_NSCBC_FARFIELD_MODE=mode,ASTR_NSCBC_FARFIELD_SHOCK_X='1',
        ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='32.6',ASTR_NSCBC_FARFIELD_RHO='1',
        ASTR_NSCBC_FARFIELD_U='1',ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',
        ASTR_NSCBC_FARFIELD_T='1')
    if diagnostics:
        (case/'diagnostics').mkdir()
        env.update(ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',ASTR_VALIDATION_RHS_STEP='0',
                   ASTR_VALIDATION_GRADIENTS='1')
    command=mpi_command(MPIEXEC,int(np.prod(topology)),transport)
    if memcheck:
        command+=['compute-sanitizer','--tool','memcheck','--error-exitcode','99']
    command += [str(Path(EXE).resolve()),'run','datin/input.flatplate']
    with (case/'run.log').open('w') as log:
        result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    text=(case/'run.log').read_text()
    if reject:
        assert reject in text and 'The job is done!' not in text,text[-8000:]
        return case
    assert result.returncode==0 and 'The job is done!' in text,text[-8000:]
    if mode=='sbli_grcbc':
        assert 'complete-RHS closure, top transverse filter off' in text
        assert 'sbli-grcbc-h1 high-order boundary chain' in text
    if memcheck:
        assert text.count('ERROR SUMMARY: 0 errors')==int(np.prod(topology)),text[-8000:]
    fields(case,steps)
    return case


@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1)])
def test_same_phase_cpu_gpu(tmp_path,topology):
    cpu=launch(tmp_path/'cpu','cpu',topology=topology)
    gpu=launch(tmp_path/'gpu','gpu',topology=topology)
    errors=compare_cpu_gpu(cpu,gpu)
    (tmp_path/'field_errors.json').write_text(json.dumps(errors,indent=2))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_exact_checkpoint_continuation(tmp_path,backend):
    topology=(2,1,1)
    full=launch(tmp_path/'full',backend,topology=topology)
    part=launch(tmp_path/'part',backend,steps=5,topology=topology)
    source=part/'outdat/new/checkpoints/step000000000005'
    tail=launch(tmp_path/'tail',backend,topology=topology,restore=source)
    final=Path('outdat/new/checkpoints/step000000000012')
    compare_fields(full/final/'state.h5',tail/final/'state.h5')
    assert (full/final/'control.bin').read_bytes()==(tail/final/'control.bin').read_bytes()


def test_mode_change_is_not_exact_restart(tmp_path):
    part=launch(tmp_path/'old','gpu',steps=5,mode='sbli_shock')
    source=part/'outdat/new/checkpoints/step000000000005'
    launch(tmp_path/'new','gpu',restore=source,reject='boundary/driver configuration mismatch')


def test_top_state_response(tmp_path):
    old=launch(tmp_path/'old','gpu',mode='sbli_shock')
    new=launch(tmp_path/'new','gpu')
    old_fields,new_fields=fields(old,12),fields(new,12)
    selected=(np.linspace(-1.,10.,65)>2.)&(np.linspace(-1.,10.,65)<5.)
    summary={}
    for label,data in [('old',old_fields),('grcbc',new_fields)]:
        summary[label]={
            'top_mean_pressure_ratio':float(np.mean(data['pressure'][:,-1,selected]*5.6)),
            'top_mean_v_over_Uinf':float(np.mean(data['velocity'][:,-1,selected,1]))}
    (tmp_path/'top_response.json').write_text(json.dumps(summary,indent=2))
    assert summary['grcbc']['top_mean_pressure_ratio']>summary['old']['top_mean_pressure_ratio']
    assert summary['grcbc']['top_mean_v_over_Uinf']<summary['old']['top_mean_v_over_Uinf']


def test_two_rank_memcheck(tmp_path):
    launch(tmp_path/'gpu','gpu',steps=6,topology=(2,1,1),memcheck=True)


def test_cuda_aware_cpu_gpu(tmp_path):
    cpu=launch(tmp_path/'cpu','cpu',topology=(2,1,1))
    gpu=launch(tmp_path/'gpu','gpu',topology=(2,1,1),transport='device-aware')
    compare_cpu_gpu(cpu,gpu)


def test_legacy_mode_cpu_gpu(tmp_path):
    cpu=launch(tmp_path/'cpu','cpu',topology=(2,1,1),mode='sbli_shock')
    gpu=launch(tmp_path/'gpu','gpu',topology=(2,1,1),mode='sbli_shock')
    compare_cpu_gpu(cpu,gpu)


def test_driving_window_is_finite(tmp_path):
    case=launch(tmp_path/'gpu','gpu',steps=2000,topology=(2,1,1),checkpoint_interval=2000)
    state=fields(case,2000)
    x=np.linspace(-1.,10.,65)
    selected=(x>2.)&(x<5.)
    summary={
        'complete_steps':2000,'time_nondimensional':.5,
        'top_mean_pressure_ratio':float(np.mean(state['pressure'][:,-1,selected]*5.6)),
        'top_mean_v_over_Uinf':float(np.mean(state['velocity'][:,-1,selected,1])),
        'scope':'finite driving window; not an angle/jump/reflection or developed-SBLI acceptance'}
    (tmp_path/'driving_window.json').write_text(json.dumps(summary,indent=2))
