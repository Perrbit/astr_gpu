"""Window aggregation must not add ranks or double-count nested stages."""
import importlib.util
from pathlib import Path

import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/insitu/benchmark_tgv256_visualization.py'
SPEC=importlib.util.spec_from_file_location('tgv256_visualization',PATH)
BENCHMARK=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


def log(steps=2):
    lines=[]
    for rank in range(2):
        for stage,value in (('solver_initialization',4+rank),('completed_window',10+rank),
                            ('solver_total_after_mpi',15+rank)):
            lines.append(f'ASTR_INSITU_STAGE_TIMING {stage} -1 {rank} {value}')
        for step in range(steps):
            for stage,value in (('advance_inclusive',1+rank),('insitu_sample_inclusive',2+rank)):
                lines.append(f'ASTR_INSITU_STAGE_TIMING {stage} {step+1} {rank} {value}')
            lines.append(f'ASTR_GPU_RANK_RK_TIMING {rank} {step} 0.1 0.2 0.3')
    return '\n'.join(lines)


def test_whole_window_not_nested_sum(tmp_path):
    (tmp_path/'run.log').write_text(log())
    actual=BENCHMARK.timing(tmp_path,2)
    assert actual['completed_window']['seconds']==11
    assert actual['advance_inclusive']['seconds']==4
    assert actual['pure_rk']['seconds']==.6
    assert actual['completed_window']['per_rank_seconds']=={'0':10,'1':11}


def test_missing_rank_step_rejected(tmp_path):
    (tmp_path/'run.log').write_text(log().replace('ASTR_GPU_RANK_RK_TIMING 1 1 0.1 0.2 0.3',''))
    with pytest.raises(AssertionError,match='Incomplete step count'):
        BENCHMARK.timing(tmp_path,2)


@pytest.mark.parametrize('pipeline',['compatible','standard-device','direct-device'])
def test_preflight_explicit_pipeline_and_no_field_io(tmp_path,monkeypatch,pipeline):
    def prepare(command,**kwargs):
        case=Path(command[command.index('--dst-case')+1])
        (case/'datin').mkdir(parents=True)
        assert command[command.index('--maxstep')+1]=='1'
        assert command[command.index('--feqlist')+1]=='1'
    monkeypatch.setattr(BENCHMARK.subprocess,'run',prepare)
    case=tmp_path/'case'
    BENCHMARK.prepare(case,2,True,tmp_path/'lib','device-aware',pipeline)
    content=(case/'insitu.nml').read_text()
    assert f"rendering_pipeline='{pipeline}'" in content
    script='tgv_pipeline.py' if pipeline=='compatible' else 'device_render_pipeline.py'
    assert f"pipeline_file='{BENCHMARK.ROOT/'scripts/insitu'/script}'" in content
    assert 'statistics=f' in content and 'initial_frame=f, final_frame=f' in content
    assert (case/'datin/input.output').read_text().count('enabled=f')==3
