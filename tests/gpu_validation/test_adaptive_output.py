"""Native adaptive core and configuration contracts, without a Python scheduler."""
import os
from pathlib import Path
import subprocess

import pytest

from test_output_config import VALID, run

PROBE = os.environ.get('ASTR_ADAPTIVE_OUTPUT_PROBE')
CONFIG_PROBE = os.environ.get('ASTR_OUTPUT_CONFIG_PROBE')
COLLECTIVE = os.environ.get('ASTR_OUTPUT_COLLECTIVE_PROBE')
MPIEXEC = os.environ.get('ASTR_OUTPUT_MPIEXEC')
ADAPTIVE = '''&adaptive_output
 enabled=t, monitor_steps=1, event_ids='energy',s_ref=0.125,t_ref=1,
 r_on=0.01,r_off=0.001,hold_time=0.002,
 window_ids='important',window_modes='time',window_times(:,1)=0.004,0.008
/
'''
CONFIG = VALID.replace('vorticity=t', "vorticity=t,adaptive=t,dense_interval_time=0.002,event_ids='energy'") + ADAPTIVE


@pytest.mark.skipif(not PROBE, reason='Set ASTR_ADAPTIVE_OUTPUT_PROBE')
def test_native_events_clocks_missing_restart(tmp_path):
    result = subprocess.run([PROBE, str(tmp_path/'state.bin')], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS adaptive' in result.stdout


@pytest.mark.skipif(not PROBE, reason='Set ASTR_ADAPTIVE_OUTPUT_PROBE')
@pytest.mark.parametrize('scenario',['sparse_hold','overflow','step_overflow','time_and_crossing',
                                   'missing_wait','selective_override','corrupt_shared','overlap'])
def test_native_adaptive_edges(tmp_path,scenario):
    result=subprocess.run([PROBE,str(tmp_path/'state.bin'),scenario],capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS adaptive '+scenario in result.stdout


@pytest.mark.skipif(not CONFIG_PROBE, reason='Set ASTR_OUTPUT_CONFIG_PROBE')
@pytest.mark.parametrize('content', [CONFIG, CONFIG.replace('\n', '\r\n'),
    VALID + ADAPTIVE, VALID + '&adaptive_output\n/\n',
    VALID + ADAPTIVE.replace("monitor_steps=1, event_ids='energy',s_ref=0.125,t_ref=1,\n r_on=0.01,r_off=0.001,hold_time=0.002,", '')])
def test_adaptive_parse_without_catalyst(tmp_path, content):
    result = run(tmp_path, content)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not CONFIG_PROBE, reason='Set ASTR_OUTPUT_CONFIG_PROBE')
@pytest.mark.parametrize('before,after', [
    ('s_ref=0.125','s_ref=0'),('s_ref=0.125','s_ref=-1'),('s_ref=0.125','s_ref=NaN'),
    ('t_ref=1','t_ref=0'),('r_off=0.001','r_off=0.01'),('r_off=0.001','r_off=-1'),
    ('hold_time=0.002','hold_time=-1'),('monitor_steps=1','monitor_steps=0'),
    ('monitor_steps=1','monitor_steps=1,monitor_time=0.1'),
    ('enabled=t, monitor','enabled=f, monitor'),('dense_interval_time=0.002','dense_interval_time=0.01'),
    ('dense_interval_time=0.002','dense_interval_time=-1'),("event_ids='energy'\n", "event_ids='missing'\n"),
    ("window_modes='time'","window_modes='bad'"),('0.004,0.008','0.008,0.004'),
    ("window_ids='important'","indicator='unsupported',window_ids='important'"),
    ("event_ids='energy',s_ref", "event_ids='energy','energy',s_ref"),
])
def test_invalid_adaptive_configuration_rejected(tmp_path, before, after):
    content = CONFIG.replace(before, after)
    assert content != CONFIG
    result = run(tmp_path, content)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'REJECT unchanged' in result.stdout


@pytest.mark.skipif(not CONFIG_PROBE,reason='Set ASTR_OUTPUT_CONFIG_PROBE')
def test_checkpoint_cannot_be_adaptive(tmp_path):
    result=run(tmp_path,CONFIG.replace('keep=2','keep=2,adaptive=t'))
    assert result.returncode!=0 and 'REJECT unchanged' in result.stdout


@pytest.mark.skipif(not CONFIG_PROBE,reason='Set ASTR_OUTPUT_CONFIG_PROBE')
def test_documented_adaptive_native_example(tmp_path):
    root=Path(__file__).resolve().parents[2]
    result=run(tmp_path,(root/'scripts/output/input.output.tgv.adaptive.example').read_text())
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.skipif(not os.environ.get('ASTR_INSITU_CONFIG_PROBE'),reason='Select native in-situ parser')
def test_documented_adaptive_insitu_template(tmp_path):
    root=Path(__file__).resolve().parents[2]
    path=tmp_path/'input.insitu'
    path.write_text((root/'scripts/insitu/presets/tgv32/adaptive.nml.in').read_text())
    result=subprocess.run([os.environ['ASTR_INSITU_CONFIG_PROBE'],str(path)],text=True,capture_output=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC,reason='Select native MPI probes')
@pytest.mark.parametrize('mode',['adaptive_scale','adaptive_dense','adaptive_permutation'])
def test_adaptive_typed_collective_and_canonical_ids(tmp_path,mode):
    content=CONFIG.replace("event_ids='energy'", "event_ids='energy','slow'").replace(
        "event_ids='energy','slow',s_ref=0.125,t_ref=1", "event_ids='energy','slow',s_ref=0.125,0.25,t_ref=1,2").replace(
        'r_on=0.01,r_off=0.001,hold_time=0.002','r_on=0.01,0.02,r_off=0.001,0.002,hold_time=0.002,0.004')
    path=tmp_path/'input.output'; path.write_text(content)
    result=subprocess.run([MPIEXEC,'--mca','coll_hcoll_enable','0','-np','2',COLLECTIVE,str(path),mode],
                          text=True,capture_output=True,timeout=30)
    accepted=mode=='adaptive_permutation'
    assert (result.returncode==0)==accepted,result.stdout+result.stderr
    assert result.stdout.count('PASS collective rank ' if accepted else 'REJECT adaptive rank ')==2


@pytest.mark.skipif(not os.environ.get('ASTR_INSITU_COLLECTIVE_PROBE') or not MPIEXEC,reason='Select native in-situ MPI probe')
@pytest.mark.parametrize('change',['dense','association_order'])
def test_insitu_adaptive_bindings_collective(tmp_path,change):
    from test_insitu_run_config import VALID as INSITU
    content=INSITU.replace('step_interval=2',
        "product_ids='q_surface.image',product_modes='steps',product_steps=5,product_adaptive=t,product_dense_steps=1,product_events(1:2,1)='a','z'")
    (tmp_path/'rank0.nml').write_text(content)
    changed=content.replace('product_dense_steps=1','product_dense_steps=2') if change=='dense' else content.replace("'a','z'","'z','a'")
    (tmp_path/'rank1.nml').write_text(changed)
    result=subprocess.run([MPIEXEC,'--mca','coll_hcoll_enable','0','-np','2',
        os.environ['ASTR_INSITU_COLLECTIVE_PROBE'],str(tmp_path/'rank')],text=True,capture_output=True,timeout=30)
    accepted=change=='association_order'
    assert (result.returncode==0)==accepted,result.stdout+result.stderr
    assert result.stdout.count('PASS collective rank ' if accepted else 'REJECT rank ')==2
