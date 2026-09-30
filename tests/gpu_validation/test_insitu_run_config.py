"""Execute the Fortran namelist parser, not a Python duplicate of its rules."""
import os
from pathlib import Path
import subprocess

import pytest

PROBE = os.environ.get('ASTR_INSITU_CONFIG_PROBE')
COLLECTIVE = os.environ.get('ASTR_INSITU_COLLECTIVE_PROBE')
MPIEXEC = os.environ.get('ASTR_INSITU_MPIEXEC')

VALID = """&insitu_run
enabled=t, statistics=t, render=t,
statistics_window=0.0005,0.0035,
schedule_mode='steps', step_interval=2,
host_budget_bytes=4294967296, device_budget_bytes=2147483648,
device_reserve_bytes=1073741824,
implementation_path='lib/catalyst', pipeline_file='tgv.py', output_directory='outdat/insitu'
/
"""


@pytest.mark.skipif(not PROBE,reason='Set ASTR_INSITU_CONFIG_PROBE')
@pytest.mark.parametrize('content,accepted',[
    ('&insitu_run /',True),
    (VALID,True),
    (VALID.replace('\n','\r\n'),True),
    (VALID.replace("schedule_mode='steps', step_interval=2", "schedule_mode='time', time_interval=0.002"),True),
    (VALID.replace('step_interval=2','step_interval=2,time_interval=0.002'),False),
    (VALID.replace('0.0005,0.0035','0.0035,0.0005'),False),
    (VALID.replace('0.0005,0.0035','NaN,0.0035'),False),
    (VALID.replace('host_budget_bytes=4294967296','host_budget_bytes=0'),False),
    (VALID.replace('device_reserve_bytes=1073741824','device_reserve_bytes=-1'),False),
    (VALID.replace("pipeline_file='tgv.py'","pipeline_file=''"),False),
    (VALID.replace("output_directory='outdat/insitu'", "output_directory='"+'x'*1100+"'"),False),
    (VALID.replace('step_interval=2','misspelled_interval=2'),False),
    (VALID.replace('step_interval=2',"step_interval=2,batch_prefix='outdat/pair'"),True),
    ("&insitu_run restore_batch='old' /",False),
    (VALID.replace('step_interval=2',"step_interval=2,restore_batch='old'"),True),
])
def test_options(tmp_path,content,accepted):
    path=tmp_path/'insitu.nml'
    path.write_bytes(content.encode())
    result=subprocess.run([str(Path(PROBE).resolve()),str(path)],capture_output=True,text=True,timeout=10)
    assert (result.returncode==0)==accepted,result.stdout+result.stderr
    if accepted:
        assert 'PASS: parsed' in result.stdout


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC,reason='Set collective probe and MPI launcher')
@pytest.mark.parametrize('other,accepted',[
    (VALID,True),
    ('! rank-local comment\n'+VALID.replace('\n','\r\n'),True),
    (VALID.replace('step_interval=2','step_interval=3'),False),
    (VALID.replace('host_budget_bytes=4294967296','host_budget_bytes=4294967297'),False),
    (VALID.replace('0.0005,0.0035','0.0005,0.0036'),False),
    (VALID.replace('render=t','render=f'),False),
    (VALID.replace('tgv.py','other.py'),False),
    (VALID.replace('enabled=t','enabled=f'),False),
    (VALID.replace('step_interval=2',"step_interval=2,batch_prefix='outdat/pair'"),False),
    (VALID.replace('step_interval=2',"step_interval=2,restore_batch='old'"),False),
    (VALID.replace('step_interval=2','step_interval=-1'),False),
    (None,False),
])
def test_collective_options(tmp_path,other,accepted):
    (tmp_path/'rank0.nml').write_text(VALID)
    if other is not None:
        (tmp_path/'rank1.nml').write_bytes(other.encode())
    result=subprocess.run([MPIEXEC,'-np','2',str(Path(COLLECTIVE).resolve()),
                           str(tmp_path/'rank')],capture_output=True,text=True,timeout=30)
    output=result.stdout+result.stderr
    assert (result.returncode==0)==accepted,output
    if accepted:
        assert output.count('PASS collective rank ')==2,output
    else:
        assert output.count('REJECT rank ')==2,output
