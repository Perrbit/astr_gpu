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
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu'"),True),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='unknown'"),False),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu',products='q_streamlines'"),True),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu',products='velocity_slice',slice_axis='x',slice_index=16"),True),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu',products='velocity_slice',slice_axis='bad'"),False),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu',products='velocity_slice',slice_index=-1"),False),
    (VALID.replace('step_interval=2', "step_interval=2,slice_index=16"),False),
    (VALID.replace('step_interval=2', "step_interval=2,products='unknown'"),False),
    (VALID.replace('statistics=t','statistics=f').replace('step_interval=2',
        "step_interval=2,products='channel_walls'"),True),
    (VALID.replace('step_interval=2',"step_interval=2,products='channel_walls'"),True),
    (VALID.replace('statistics=t','statistics=f').replace('step_interval=2',
        "step_interval=2,products='channel_walls',derivative_backend='gpu'"),False),
    (VALID.replace('statistics=t','statistics=f').replace('step_interval=2',
        "step_interval=2,products='air5_walls'"),True),
    (VALID.replace('step_interval=2',"step_interval=2,products='air5_walls'"),True),
    (VALID.replace('step_interval=2',"step_interval=2,products='air5_walls',wall_separation=t"),True),
    (VALID.replace('step_interval=2',"step_interval=2,products='channel_walls',wall_separation=t"),False),
    (VALID.replace('step_interval=2',"step_interval=2,products='air5_walls',wall_mean_render=t"),True),
    (VALID.replace('step_interval=2','step_interval=2,wall_mean_render=t'),False),
    (VALID.replace('statistics=t','statistics=f').replace('step_interval=2',
        "step_interval=2,products='channel_walls',wall_mean_render=t"),False),
    (VALID.replace('step_interval=2',"step_interval=2,products='air5_walls',air5_volume_statistics=t"),True),
    (VALID.replace('step_interval=2','step_interval=2,air5_volume_statistics=t'),False),
    (VALID.replace('step_interval=2',"step_interval=2,products='air5_walls',air5_volume_statistics=t,air5_volume_reduction=t"),True),
    (VALID.replace('step_interval=2','step_interval=2,air5_volume_reduction=t'),False),
    (VALID.replace('statistics=t','statistics=f').replace('step_interval=2',
        "step_interval=2,products='air5_walls',air5_volume_statistics=t"),False),
    (VALID.replace('statistics=t','statistics=f').replace('step_interval=2',
        "step_interval=2,products='air5_walls',derivative_backend='gpu'"),False),
    (VALID.replace('step_interval=2', "step_interval=2,products='streamlines'"),False),
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
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu'"),False),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu',products='streamlines'"),False),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu',products='velocity_slice',slice_axis='x',slice_index=16"),False),
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


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC,reason='Set collective probe and MPI launcher')
@pytest.mark.parametrize('change',["slice_axis='x'",'slice_index=16'])
def test_collective_plane_identity(tmp_path,change):
    text=VALID.replace('step_interval=2',
        "step_interval=2,derivative_backend='gpu',products='velocity_slice',slice_axis='z',slice_index=4")
    (tmp_path/'rank0.nml').write_text(text)
    key=change.split('=')[0]
    other=text.replace("slice_axis='z'" if key=='slice_axis' else 'slice_index=4',change)
    (tmp_path/'rank1.nml').write_text(other)
    result=subprocess.run([MPIEXEC,'-np','2',str(Path(COLLECTIVE).resolve()),str(tmp_path/'rank')],
        capture_output=True,text=True,timeout=30)
    output=result.stdout+result.stderr
    assert result.returncode!=0 and output.count('REJECT rank ')==2,output


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC,reason='Set collective probe and MPI launcher')
def test_collective_volume_selection(tmp_path):
    text=VALID.replace('step_interval=2',"step_interval=2,products='air5_walls',air5_volume_statistics=t")
    (tmp_path/'rank0.nml').write_text(text)
    (tmp_path/'rank1.nml').write_text(text.replace('air5_volume_statistics=t','air5_volume_statistics=f'))
    result=subprocess.run([MPIEXEC,'-np','2',str(Path(COLLECTIVE).resolve()),str(tmp_path/'rank')],
        capture_output=True,text=True,timeout=30)
    output=result.stdout+result.stderr
    assert result.returncode!=0 and output.count('REJECT rank ')==2,output


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC,reason='Set collective probe and MPI launcher')
@pytest.mark.parametrize('flag',('air5_volume_reduction','wall_mean_render','wall_separation'))
def test_collective_statistic_products(tmp_path,flag):
    text=VALID.replace('step_interval=2',
        "step_interval=2,products='air5_walls',air5_volume_statistics=t,air5_volume_reduction=t,wall_mean_render=t,wall_separation=t")
    (tmp_path/'rank0.nml').write_text(text)
    (tmp_path/'rank1.nml').write_text(text.replace(flag+'=t',flag+'=f'))
    result=subprocess.run([MPIEXEC,'-np','2',str(Path(COLLECTIVE).resolve()),str(tmp_path/'rank')],
        capture_output=True,text=True,timeout=30)
    output=result.stdout+result.stderr
    assert result.returncode!=0 and output.count('REJECT rank ')==2,output


DEVICE = VALID.replace('step_interval=2',
    "step_interval=2,derivative_backend='gpu',processing_backend='device',postprocess_transport='pinned'")


@pytest.mark.skipif(not PROBE, reason='Set ASTR_INSITU_CONFIG_PROBE')
@pytest.mark.parametrize('content,accepted', [
    (DEVICE, True),
    (DEVICE.replace("'pinned'", "'device-aware'"), True),
    (DEVICE.replace(",postprocess_transport='pinned'", ''), False),
    (DEVICE.replace("'pinned'", "'automatic'"), False),
    (DEVICE.replace("processing_backend='device'", "processing_backend='host'"), False),
    (DEVICE.replace("processing_backend='device'", "processing_backend='unknown'"), False),
    (DEVICE.replace("derivative_backend='gpu'", "derivative_backend='cpu'"), False),
    (DEVICE.replace('render=t', 'render=f'), False),
    (DEVICE.replace('device_budget_bytes=2147483648', 'device_budget_bytes=0'), False),
    (DEVICE.replace('step_interval=2', "step_interval=2,products='channel_walls'"), False),
])
def test_device_transport_options(tmp_path, content, accepted):
    path = tmp_path / 'device.nml'
    path.write_text(content)
    env = dict(os.environ, ASTR_GPU_HALO_TRANSPORT='device-aware')
    result = subprocess.run([str(Path(PROBE).resolve()), str(path)], env=env,
                            capture_output=True, text=True, timeout=10)
    assert (result.returncode == 0) == accepted, result.stdout + result.stderr


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC, reason='Set collective probe and MPI launcher')
@pytest.mark.parametrize('other,accepted', [
    (DEVICE, True),
    (DEVICE.replace("'pinned'", "'device-aware'"), False),
    (DEVICE.replace(",postprocess_transport='pinned'", ''), False),
    (VALID.replace('step_interval=2', "step_interval=2,derivative_backend='gpu'"), False),
])
def test_collective_device_transport(tmp_path, other, accepted):
    (tmp_path / 'rank0.nml').write_text(DEVICE)
    (tmp_path / 'rank1.nml').write_text(other)
    result = subprocess.run([MPIEXEC, '-np', '2', str(Path(COLLECTIVE).resolve()), str(tmp_path / 'rank')],
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == accepted, output
    assert output.count('PASS collective rank ' if accepted else 'REJECT rank ') == 2, output
