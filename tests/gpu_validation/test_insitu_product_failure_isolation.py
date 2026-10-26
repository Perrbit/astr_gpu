"""Optional products fail closed; the solver and sibling products keep running."""
import json
import os
from pathlib import Path
import pytest

from run_output_restart_validation import run_case
from test_output_insitu_restart import ROOT,arguments,configuration
from test_insitu_device_products import resident_configuration,field_difference

pytestmark=pytest.mark.skipif(not os.environ.get('ASTR_OUTPUT_INSITU_EXE') or
    not os.environ.get('ASTR_OUTPUT_INSITU_BACKEND'),reason='Select the native rendering build explicitly')


def config(pipeline,independent=True):
    if pipeline=='compatible':
        value=configuration(interval=2)
        filename='tgv_pipeline.py'
        ids="'q_surface.geometry','q_surface.image','velocity_slice.image'"
        modes="'steps','steps','steps'";steps='2,2,2'
    else:
        value=resident_configuration(pipeline,interval=2)
        filename='device_render_pipeline.py'
        ids="'q_surface.image','velocity_slice.image'"
        modes="'steps','steps'";steps='2,2'
    value=value.replace(str(ROOT/'scripts/insitu'/filename),
        str(ROOT/'tests/gpu_validation/insitu_product_failure_pipeline.py'))
    if independent:
        value=value.replace('step_interval=2,',f'product_ids={ids}, product_modes={modes}, product_steps={steps},')
    return value


@pytest.mark.parametrize('pipeline',['compatible','standard-device','direct-device'])
@pytest.mark.parametrize('fault',['image','capture'])
def test_sibling_products_solver_and_exact_restart(tmp_path,pipeline,fault):
    args=arguments(tmp_path)
    args.runtime_timeout_seconds=180
    args.directory_budget_bytes=256*1024**2
    cfg=config(pipeline)
    healthy,_=run_case(args,ROOT,'gpu',2,'healthy',4,grid='32,32,32',
        checkpoint_interval=2,insitu_config=cfg)
    failed,_=run_case(args,ROOT,'gpu',2,'fault_'+fault,4,grid='32,32,32',
        checkpoint_interval=2,insitu_config=cfg)
    output=failed/'outdat/render'
    assert 'ASTR_INSITU_PRODUCT_DISABLED' in (failed/'run.log').read_text()
    assert not (output/'q_surface.step00000004.jpeg').exists()
    assert (output/'velocity_slice.step00000004.jpeg').is_file()
    if pipeline=='compatible':
        assert (output/'q_surface.step00000004.pvtp').is_file()
    for rank in (0,1):
        assert json.loads((output/f'lifecycle_rank{rank}.json').read_text())['finalized']
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        assert field_difference(failed/final/name,healthy/final/name)==0
    resumed,_=run_case(args,ROOT,'gpu',2,'resumed',4,grid='32,32,32',
        restore=failed/'outdat/new/checkpoints/step000000000002',
        checkpoint_interval=2,insitu_config=cfg)
    assert not (resumed/'outdat/render/q_surface.step00000004.jpeg').exists()
    for name in ('state.h5','statistics.h5'):
        assert field_difference(failed/final/name,resumed/final/name)==0


def test_geometry_failure_keeps_image_and_other_products(tmp_path):
    args=arguments(tmp_path); args.directory_budget_bytes=256*1024**2
    failed,_=run_case(args,ROOT,'gpu',2,'fault_geometry',4,grid='32,32,32',
        checkpoint_interval=2,insitu_config=config('compatible'))
    output=failed/'outdat/render'
    assert not (output/'q_surface.step00000004.pvtp').exists()
    assert (output/'q_surface.step00000004.jpeg').is_file()
    assert (output/'velocity_slice.step00000004.jpeg').is_file()
    assert 'ASTR_INSITU_PRODUCT_DISABLED' in (failed/'run.log').read_text()


@pytest.mark.parametrize('pipeline',['standard-device','direct-device'])
def test_rank_local_device_extraction_failure_keeps_slice(tmp_path,pipeline):
    args=arguments(tmp_path); args.directory_budget_bytes=256*1024**2
    cfg=config(pipeline)
    healthy,_=run_case(args,ROOT,'gpu',2,'healthy',6,grid='32,32,32',checkpoint_interval=2,insitu_config=cfg)
    failed,_=run_case(args,ROOT,'gpu',2,'fault_extraction',6,grid='32,32,32',checkpoint_interval=2,insitu_config=cfg)
    output=failed/'outdat/render'
    assert (output/'q_surface.step00000002.jpeg').is_file()
    assert not (output/'q_surface.step00000004.jpeg').exists()
    assert not (output/'q_surface.step00000006.jpeg').exists()
    assert (output/'velocity_slice.step00000006.jpeg').is_file()
    assert 'injected rank-local product extraction failure' in (failed/'run.log').read_text()
    final='outdat/new/checkpoints/step000000000006'
    for name in ('state.h5','statistics.h5'):
        assert field_difference(failed/final/name,healthy/final/name)==0


@pytest.mark.parametrize('pipeline',['compatible','standard-device','direct-device'])
def test_common_clock_disabled_state_survives_restart(tmp_path,pipeline):
    args=arguments(tmp_path); args.directory_budget_bytes=256*1024**2
    cfg=config(pipeline,independent=False)
    full,_=run_case(args,ROOT,'gpu',2,'fault_image',4,grid='32,32,32',checkpoint_interval=2,insitu_config=cfg)
    resumed,_=run_case(args,ROOT,'gpu',2,'resumed',4,grid='32,32,32',checkpoint_interval=2,
        restore=full/'outdat/new/checkpoints/step000000000002',insitu_config=cfg)
    output=resumed/'outdat/render'
    assert not (output/'q_surface.step00000004.jpeg').exists()
    assert (output/'velocity_slice.step00000004.jpeg').is_file()
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        assert field_difference(full/final/name,resumed/final/name)==0
    assert (full/final/'insitu_control.bin').read_bytes()==(resumed/final/'insitu_control.bin').read_bytes()


@pytest.mark.parametrize('pipeline',['compatible','standard-device'])
def test_optional_diagnostic_write_failure_is_not_a_second_fatal_error(tmp_path,pipeline):
    args=arguments(tmp_path); args.directory_budget_bytes=256*1024**2
    cfg=config(pipeline)
    healthy,_=run_case(args,ROOT,'gpu',2,'healthy',4,grid='32,32,32',checkpoint_interval=2,insitu_config=cfg)
    failed,_=run_case(args,ROOT,'gpu',2,'fault_diagnostics',4,grid='32,32,32',checkpoint_interval=2,insitu_config=cfg)
    log=(failed/'run.log').read_text()
    assert 'ASTR_INSITU_DIAGNOSTIC_WRITE_FAILED' in log and 'logging disabled' in log
    assert (failed/'outdat/render/velocity_slice.step00000004.jpeg').is_file()
    final='outdat/new/checkpoints/step000000000004'
    for name in ('state.h5','statistics.h5'):
        assert field_difference(failed/final/name,healthy/final/name)==0


def test_m12_main_slice_failure_keeps_temperature_and_derived_views(tmp_path):
    from test_m12_insitu_memory import launch
    script=ROOT/'tests/gpu_validation/insitu_product_failure_pipeline.py'
    case=tmp_path/'m12_fault_velocity'
    healthy=tmp_path/'m12_healthy'
    launch(healthy,Path(os.environ['ASTR_OUTPUT_INSITU_EXE']),'slice_only',steps=4,pipeline_file=script)
    launch(case,Path(os.environ['ASTR_OUTPUT_INSITU_EXE']),'slice_only',steps=4,pipeline_file=script)
    output=case/'outdat/render'
    assert not (output/'velocity_slice.step00000004.jpeg').exists()
    for name in ('temperature_slice','spanwise_slice','velocity_slice.x0_100'):
        assert (output/f'{name}.step00000004.jpeg').is_file()
    log=(case/'run.log').read_text()
    assert 'ASTR_INSITU_PRODUCT_DISABLED' in log and 'The job is done!' in log
    checkpoint='outdat/new/checkpoints/step000000000004'
    # Exact halo caches include deliberately unusable NaN sentinels. Their bits
    # must match; the physical q/cache datasets must still all be finite.
    import h5py
    import numpy as np
    from run_output_restart_validation import compare_fields
    with h5py.File(case/checkpoint/'state.h5') as state:
        for name in state:
            if name.startswith('q'):
                assert np.isfinite(state[name][...]).all(),name
    compare_fields(case/checkpoint/'state.h5',healthy/checkpoint/'state.h5')
    resumed=tmp_path/'m12_resumed'
    launch(resumed,Path(os.environ['ASTR_OUTPUT_INSITU_EXE']),'slice_only',steps=6,pipeline_file=script,
        restore=case/checkpoint)
    assert not (resumed/'outdat/render/velocity_slice.step00000006.jpeg').exists()
    assert (resumed/'outdat/render/temperature_slice.step00000006.jpeg').is_file()
