"""Timing aggregation is not a sum of parallel work or nested stages."""
import json
import importlib.util
from pathlib import Path
import pytest

from run_insitu_acceptance_timing import read_timing,preset
from test_insitu_curve_derivatives import arguments,ROOT
from run_output_restart_validation import run_case


def fixture_log(ranks=2):
    rows=[]
    for rank in range(ranks):
        for stage,value in [('solver_initialization',rank+1),('solver_total_after_mpi',10+rank),('completed_window',8+rank)]:
            rows.append(f'ASTR_INSITU_STAGE_TIMING {stage} -1 {rank} {value}')
        for step in range(1,5):
            rows.append(f'ASTR_INSITU_STAGE_TIMING advance_inclusive {step} {rank} {rank+1}')
    return rows


def test_maximum_of_rank_windows_not_sum_of_event_maxima(tmp_path):
    rows=fixture_log()
    rows+=['ASTR_INSITU_STAGE_TIMING statistics_inclusive 1 0 4',
           'ASTR_INSITU_STAGE_TIMING statistics_inclusive 2 0 0',
           'ASTR_INSITU_STAGE_TIMING statistics_inclusive 1 1 0',
           'ASTR_INSITU_STAGE_TIMING statistics_inclusive 2 1 4']
    rows+=['ASTR_INSITU_PIPELINE_TIMING '+json.dumps(dict(rank=0,step=2,seconds=dict(image_encoding=.1)))]
    (tmp_path/'run.log').write_text('\n'.join(rows))
    result=read_timing(tmp_path,2)
    assert result['statistics_inclusive']['seconds']==4
    assert result['advance_inclusive']['seconds']==8
    assert result['completed_window']['seconds']==9
    assert result['image_encoding']['per_rank_events']=={'0':1,'1':0}


@pytest.mark.parametrize('defect',('duplicate','missing','nan','negative','bad_rank','short_window'))
def test_reject_incomplete_or_invalid_timing(tmp_path,defect):
    rows=fixture_log()
    if defect=='duplicate':rows.append(rows[0])
    elif defect=='missing':rows.pop(0)
    elif defect=='short_window':rows.pop()
    else:
        value={'nan':'nan','negative':'-1','bad_rank':'1'}[defect]
        rows.append(f'ASTR_INSITU_STAGE_TIMING invalid 1 {9 if defect=="bad_rank" else 0} {value}')
    (tmp_path/'run.log').write_text('\n'.join(rows))
    with pytest.raises(ValueError):read_timing(tmp_path,2)


@pytest.mark.parametrize('enabled',(False,True))
def test_standalone_preset_has_no_validation_imports_or_private_defaults(tmp_path,enabled):
    path=Path(__file__).resolve().parents[2]/'scripts/insitu/start_tgv_acceptance.py'
    spec=importlib.util.spec_from_file_location('standalone_insitu_preset',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    library=tmp_path/'library';library.mkdir()
    output=tmp_path/'case'
    module.prepare(output,library,enabled)
    config=(output/'insitu.nml').read_text()
    assert '${' not in config and f'enabled={"t" if enabled else "f"}' in config
    assert 'statistics_window=0.003,0.004' in config and 'step_interval=2' in config
    assert str(library.resolve()) in config
    assert {p.name for p in (output/'datin').iterdir()}=={'input.tgv','controller','input.output'}
    assert (output/'outdat/render').is_dir()
    assert (output/'outdat/new').is_dir()
    primary=(output/'datin/input.tgv').read_text().splitlines()
    assert primary[5]=='tgv' and primary[8]=='32,32,32'
    assert primary[11]=='t,t,t' and primary[14]=='t,t,t,f,f,f,t,f,t'
    control=(output/'datin/controller').read_text().splitlines()
    assert control[5]=='f,f,f,f' and control[8]=='3,1,100,50,1,50' and control[11]=='1.d-3'
    assert 'tests.gpu_validation' not in path.read_text()
    assert '/home/dell' not in path.read_text()
    with pytest.raises(FileExistsError):module.prepare(output,library,enabled)


@pytest.mark.parametrize('processor,transport,accepted',[
    ('device','pinned',True),('device','device-aware',True),('host',None,True),
    ('device',None,False),('host','pinned',False),('device','automatic',False)])
def test_standalone_device_transport_is_explicit(tmp_path,processor,transport,accepted):
    spec=importlib.util.spec_from_file_location('standalone_insitu_preset',
        ROOT/'scripts/insitu/start_tgv_acceptance.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    library=tmp_path/'library';library.mkdir()
    output=tmp_path/'case'
    if not accepted:
        with pytest.raises(ValueError):module.prepare(output,library,True,processor,transport)
        assert not output.exists()
    else:
        module.prepare(output,library,True,processor,transport)
        content=(output/'insitu.nml').read_text()
        if processor=='device':
            assert "processing_backend='device'" in content
            assert f"postprocess_transport='{transport}'" in content


@pytest.mark.parametrize('kind',('host','device','shared_device'))
def test_native_observed_budget_rejects_without_fallback(tmp_path,kind,monkeypatch):
    config=preset();ranks=1
    if kind=='host':
        config=config.replace('host_budget_bytes=4294967296','host_budget_bytes=134217728')
        message='node host increment'
    else:
        config=config.replace('device_budget_bytes=2147483648','device_budget_bytes=67108864')
        message='physical GPU increment'
        if kind=='shared_device':
            ranks=2;monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    run_case(arguments(tmp_path,'gpu','x'),ROOT,'gpu',ranks,'budget_'+kind,4,
        grid='32,32,32',checkpoint_interval=1,insitu_config=config,
        insitu_timing=True,reject=message,failure_after_start=True)
