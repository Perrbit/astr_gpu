"""Small real M12 sampling window, including wall output and restart lifecycle."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

from run_m12_hd_coverage import prepare_case,check_compact,MPIEXEC
from run_m12_boundary_compare import environment,configuration
from run_output_restart_validation import compare_fields

ROOT=Path(__file__).resolve().parents[2]
CAMPAIGN=ROOT/'tests/gpu_validation/out/runtime_overhead_cleanup_20261018'
CURRENT=Path(os.environ.get('ASTR_M12_DIAGNOSTIC_EXE',ROOT/'build_m12_device_render/bin/astr'))
SHAPE=(64,32,24)


def launch(case,executable,steps,checkpoint=False,restore=None,trace=False):
    prepare_case(ROOT.parent/'M12_C13',case,SHAPE,steps,False)
    if checkpoint:
        configuration(case,'extrapolation',SHAPE,steps,archive=False,checkpoint_interval=50,restore=restore)
    env=environment('2,1,1')
    env.update(ASTR_GPU_MP_POSITIVITY='flux',ASTR_GPU_RK_TIMING='1',ASTR_INSITU_CONFIG='datin/input.insitu',
               ASTR_INSITU_TEST_ORACLE_IO='0',ASTR_M12_RENDER_ISOLATION='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
        env.pop(key,None)
    solver=[str(executable),'run','datin/input.dat']
    if trace:
        wrapper=case/'trace.sh'
        wrapper.write_text('#!/bin/sh\nexec strace -e trace=openat -o '+
            shlex.quote(str(case/'opens'))+'.rank"${OMPI_COMM_WORLD_RANK}" '+
            ' '.join(map(shlex.quote,solver))+'\n')
        wrapper.chmod(0o755)
        solver=[str(wrapper)]
    command=[str(MPIEXEC),'--mca','pml','ob1','--mca','btl','self,vader,tcp','--bind-to','none','-np','2',*solver]
    with (case/'run.log').open('w') as log:
        result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    text=(case/'run.log').read_text()
    assert result.returncode==0 and 'The job is done!' in text,case/'run.log'
    assert not list((case/'outdat/render').glob('*.jpeg'))
    return text


def test_m12_probe_and_wall_outputs_unchanged_with_one_open(tmp_path):
    results={}
    for name,binary in [('frozen',CAMPAIGN/'frozen/astr_m12_render'),('current',CURRENT)]:
        case=tmp_path/name
        text=launch(case,binary,52,trace=True)
        check_compact(case,52)
        assert text.count('q_bits_rng_unchanged=1')==4
        count=0
        for path in case.glob('opens.rank*'):
            count+=sum('"outdat/render/probes.csv"' in line and any(flag in line for flag in ('O_WRONLY','O_RDWR'))
                       for line in path.read_text().splitlines())
        results[name]=dict(probe_opens=count)
    assert results['frozen']['probe_opens']==52 and results['current']['probe_opens']==1,results
    for name in ('probes.csv','wall.step00000050.rank0000.csv','wall.step00000050.rank0001.csv'):
        assert (tmp_path/'frozen/outdat/render'/name).read_bytes()==(tmp_path/'current/outdat/render'/name).read_bytes(),name
    (tmp_path/'file_opens.json').write_text(json.dumps(results,indent=2)+'\n')


def test_m12_restart_starts_a_separate_probe_segment(tmp_path):
    whole,first,resumed=(tmp_path/name for name in ('whole','first','resumed'))
    launch(whole,CURRENT,52,checkpoint=True)
    launch(first,CURRENT,50,checkpoint=True)
    launch(resumed,CURRENT,52,checkpoint=True,
           restore=first/'outdat/new/checkpoints/step000000000050')
    source=whole/'outdat/render/probes.csv'
    continuation=resumed/'outdat/render/probes.from_step000000000050.csv'
    expected=source.read_text().splitlines()
    assert continuation.read_text().splitlines()==[expected[0],*expected[1+50*72:]]
    for path in (whole,first,resumed):
        assert not list((path/'outdat/render').glob('*.partial'))
    final='outdat/new/checkpoints/step000000000052/state.h5'
    compare_fields(whole/final,resumed/final)
