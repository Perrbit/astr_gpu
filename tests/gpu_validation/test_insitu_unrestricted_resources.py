"""Unlimited opt-in resource accounting; numerical and real allocation checks remain separate."""
import io
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import insitu_resource_monitor as monitor


@pytest.fixture
def sampled_runtime(monkeypatch):
    gib=1024**3
    class Process:
        pid=123456789
        def __init__(self):self.polls=0
        def poll(self):
            self.polls+=1
            return None if self.polls==1 else 0
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(monitor.subprocess,'Popen',lambda *a,**kw:Process())
    tree=SimpleNamespace(pid=Process.pid,children=lambda recursive:[],memory_info=lambda:SimpleNamespace(rss=9*gib))
    monkeypatch.setattr(monitor.psutil,'Process',lambda pid:tree)
    for name in ('nvmlInit','nvmlShutdown'):
        monkeypatch.setattr(monitor.nvml,name,lambda:None)
    monkeypatch.setattr(monitor.nvml,'nvmlDeviceGetCount',lambda:1)
    monkeypatch.setattr(monitor.nvml,'nvmlDeviceGetHandleByIndex',lambda i:0)
    monkeypatch.setattr(monitor.nvml,'nvmlDeviceGetUUID',lambda h:'test-gpu')
    query=lambda h:[SimpleNamespace(pid=Process.pid,usedGpuMemory=8*gib)]
    monkeypatch.setattr(monitor.nvml,'nvmlDeviceGetComputeRunningProcesses',query)
    monkeypatch.setattr(monitor.nvml,'nvmlDeviceGetGraphicsRunningProcesses',query)
    monkeypatch.setattr(monitor.nvml,'nvmlDeviceGetMemoryInfo',lambda h:SimpleNamespace(free=8*gib))
    ticks=iter([0.,1000.,1001.,1002.,1003.])
    monkeypatch.setattr(monitor.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(monitor.time,'sleep',lambda seconds:None)
    killed=[];monkeypatch.setattr(monitor.os,'killpg',lambda pid,sig:killed.append((pid,sig)))
    return dict(host_rss_peak_bytes=0,devices={'test-gpu':dict(peak_bytes=0)}),killed


def test_monitor_records_without_artificial_limits(tmp_path,sampled_runtime):
    baseline,killed=sampled_runtime
    path=tmp_path/'resources.json'
    monitor.run_monitored([],tmp_path,{},io.StringIO(),path,baseline=baseline,
        device_extra_budget_bytes=None,host_extra_budget_bytes=None,device_reserve_bytes=0,timeout_seconds=None)
    record=json.loads(path.read_text())
    assert record['status']=='passed-sampled-observation-not-allocation-proof'
    assert record['limits']==dict(device_extra_bytes=None,host_extra_bytes=None,device_reserve_bytes=0,timeout_seconds=None)
    assert record['host_rss_peak_bytes']==9*1024**3 and record['devices']['test-gpu']['peak_bytes']==8*1024**3
    assert record['samples']==2 and not killed
    assert record['elapsed_seconds']>=1000


def test_explicit_limits_still_apply_when_requested(tmp_path,sampled_runtime):
    baseline,killed=sampled_runtime
    with pytest.raises(ValueError,match='device-memory budget exceeded'):
        monitor.run_monitored([],tmp_path,{},io.StringIO(),tmp_path/'resources.json',baseline=baseline)
    assert killed


@pytest.mark.skipif(not os.environ.get('ASTR_INSITU_CONFIG_PROBE'),reason='Select the root parser probe')
def test_catalog_stats_unlimited_input(tmp_path):
    config=tmp_path/'input.insitu'
    config.write_text("&insitu_run\n enabled=t,statistics=t,render=f,products='catalog_stats',statistics_window=0,1,"
        "host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0,output_directory='outdat'\n/\n")
    result=subprocess.run([os.environ['ASTR_INSITU_CONFIG_PROBE'],str(config),'catalog'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.skipif(not os.environ.get('ASTR_WALL_FOOTPRINT_RENDER_EXE'),reason='Select the strict Catalyst/device solver')
def test_np4_unlimited_keeps_shared_accounting(tmp_path):
    from test_insitu_device_products import resident_configuration
    from test_insitu_sbli_statistics import launch,WINDOW
    from test_insitu_wall_footprints import wall_image_definitions
    config=resident_configuration('standard-device',profile='catalog_stats').replace('0.0005,0.0115',f'{WINDOW[0]},{WINDOW[1]}')
    config=config.replace("output_directory='outdat/render'","output_directory='outdat'")
    for old,new in [('host_budget_bytes=4294967296','host_budget_bytes=-1'),
        ('device_budget_bytes=2147483648','device_budget_bytes=-1'),('device_reserve_bytes=1073741824','device_reserve_bytes=0')]:
        assert old in config;config=config.replace(old,new)
    case=launch(tmp_path/'gpu','gpu',(2,2,1),steps=2,records=wall_image_definitions(('reynolds',)),
        insitu_config=config,executable=os.environ['ASTR_WALL_FOOTPRINT_RENDER_EXE'],unrestricted=True,monitor=True)
    text=(case/'run.log').read_text()
    assert text.count('ASTR_INSITU_SHARED_ALLOCATION rank=')==4
    output=(case/'datin/input.output').read_text()
    assert 'host_budget_bytes=-1' in output and 'device_budget_bytes=-1' in output
    expected=(2**63-1)//4096
    reports=list((case/'outdat').glob('resources.rank*.csv'));assert len(reports)==4
    for report in reports:
        header=report.read_text().splitlines()[0]
        assert f'host_limit={expected}' in header and f'device_limit={expected}' in header and 'reserve=0' in header
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert all(p['state']=='active' for p in state['products'])
    observation=json.loads((case/'resources.sampled.json').read_text())
    assert observation['limits']==dict(device_extra_bytes=None,host_extra_bytes=None,device_reserve_bytes=0,timeout_seconds=None)
