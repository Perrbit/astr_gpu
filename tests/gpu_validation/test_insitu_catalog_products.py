"""Typed runtime image/monitor definitions, without a rendering dependency."""
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_insitu_catalog import HEADER, PROBE, product, run

pytestmark = pytest.mark.skipif(not PROBE, reason='Set ASTR_CATALOG_PROBE from root CMake')


def test_shipped_tgv_image_example_has_complete_explicit_records(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.tgv.images.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('valid=T')==6,result.stdout
    assert 'valid=F' not in result.stdout


def test_shipped_tgv_geometry_example_has_complete_explicit_records(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.tgv.geometry.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('valid=T')==4 and 'valid=F' not in result.stdout,result.stdout


def test_shipped_channel_wall_example_has_complete_explicit_records(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.channel.walls.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('valid=T')==4 and 'valid=F' not in result.stdout,result.stdout


def test_shipped_channel_wall_chart_example_has_complete_explicit_records(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.channel.wall-charts.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('valid=T')==4 and 'valid=F' not in result.stdout,result.stdout


def test_shipped_sbli_midspan_example_has_complete_explicit_records(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.sbli.midspan.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('valid=T')==4 and 'valid=F' not in result.stdout,result.stdout


def test_shipped_m12_short_example_has_complete_explicit_records(tmp_path):
    path=Path(__file__).resolve().parents[2]/'scripts/output/input.products.m12.short.example.jsonl'
    result=run(tmp_path,path.read_text().splitlines())
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('valid=T')==6 and 'valid=F' not in result.stdout,result.stdout


def visual_product(kind='slice', name='view'):
    item = dict(id=name, kind=kind, enable_request=1, source='instantaneous',
                field='velocity_magnitude',
                sampling=dict(mode='physical_plane', origin=[0, 1, 0], normal=[0, 1, 0],
                              bounds=[0, 300, 0, 35, 0, 90]),
                display=dict(projection='parallel', camera=[150, -1800, 45],
                             focal_point=[150, 1, 45], view_up=[0, 0, 1],
                             parallel_scale=60, color_map='diverging_blue_red',
                             color_range=[0, 1.1], lighting=False, line_width=2),
                output=dict(channel='image', formats=['jpeg'], image_size=[800, 600],
                            jpeg_quality=95, dpi=300),
                schedule=dict(mode='steps', interval=100))
    if kind in ('q_surface', 'streamlines'):
        item['sampling'] = dict(mode='volume', bounds=[0, 300, 0, 35, 0, 90])
        if kind == 'q_surface':
            item['extraction'] = dict(isovalue=0.25)
        else:
            item['extraction'] = dict(seed_layout='bl-layered64', integrator='rk45',
                                      initial_step=.1, minimum_step=.01, maximum_step=.5,
                                      maximum_error=1e-8, maximum_length=300,
                                      maximum_steps=100000,direction='both')
    if kind == 'wall_curve':
        item['field'] = 'wall_pressure'
        item['sampling'] = dict(mode='wall', axis='j', side='lower',
                                spanwise_index=4,bounds=[0, 300, 0, 35, 0, 90])
        item['display']=dict(y_range=[0.,1.],line_width=2.)
    if kind == 'probe':
        item.pop('field')
        item.pop('display')
        item['fields'] = ['velocity_x', 'pressure']
        item['sampling'] = dict(mode='grid_node', indices=[25, 3, 4])
        item['output'] = dict(channel='data', format='csv')
    return item


@pytest.mark.parametrize('kind', ['slice', 'probe', 'wall_curve', 'q_surface', 'streamlines'])
def test_each_existing_product_has_explicit_typed_definition(tmp_path, kind):
    result = run(tmp_path, [HEADER, visual_product(kind), product('archive')])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRODUCT 1 valid=T id=view' in result.stdout, result.stdout
    assert 'PRODUCT 2 valid=T id=archive' in result.stdout, result.stdout


def test_index_image_reuses_global_slice_selection(tmp_path):
    item = visual_product()
    item['sampling'] = product()['sampling']
    result = run(tmp_path, [HEADER, item])
    assert 'valid=T' in result.stdout, result.stdout
    assert 'axis=2' in result.stdout and 'index=20' in result.stdout


def test_wall_curve_data_requires_explicit_spanwise_node_not_a_camera(tmp_path):
    item=visual_product('wall_curve');item.pop('display')
    item['output']=dict(channel='data',format='csv')
    result=run(tmp_path,[HEADER,item])
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout
    del item['sampling']['spanwise_index']
    result=run(tmp_path,[HEADER,item,product('healthy')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T' in result.stdout,result.stdout


def test_wall_chart_x_axis_needs_width_but_data_can_select_one_x(tmp_path):
    item=visual_product('wall_curve');item['sampling']['bounds'][1]=item['sampling']['bounds'][0]
    result=run(tmp_path,[HEADER,item,product('healthy')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T' in result.stdout,result.stdout
    item.pop('display');item['output']=dict(channel='data',format='csv')
    result=run(tmp_path,[HEADER,item])
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout


def test_physical_midspan_is_explicit_and_does_not_accept_a_node_index(tmp_path):
    item=visual_product('wall_curve');item['sampling']['mode']='wall_midspan'
    del item['sampling']['spanwise_index']
    item['field']='wall_heat_into_wall_normalized'
    result=run(tmp_path,[HEADER,item])
    assert 'PRODUCT 1 valid=T' in result.stdout,result.stdout
    item['sampling']['spanwise_index']=4
    result=run(tmp_path,[HEADER,item,product('healthy')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T' in result.stdout,result.stdout


def test_midspan_definition_survives_the_typed_control_reader(tmp_path):
    probe=os.environ.get('ASTR_CATALOG_STATE_PROBE')
    if not probe: pytest.skip('Select the root-built control probe')
    item=visual_product('wall_curve');item['sampling']['mode']='wall_midspan'
    del item['sampling']['spanwise_index'];item['field']='wall_heat_into_wall_normalized'
    path=tmp_path/'midspan.jsonl';path.write_text(json.dumps(HEADER)+'\n'+json.dumps(item)+'\n')
    result=subprocess.run([probe,str(path),'visual-checkpoint'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def geometry_product(kind='q_surface'):
    item=visual_product(kind)
    item.pop('display');item.pop('field')
    item['fields']=['velocity_x','velocity_y','velocity_z','Q_rs'] if kind!='streamlines' else [
        'velocity_x','velocity_y','velocity_z']
    item['output']=dict(channel='geometry',format='vtk-xml')
    return item


@pytest.mark.parametrize('kind',['slice','q_surface','streamlines'])
def test_geometry_definition_has_fields_but_no_image_settings(tmp_path,kind):
    result=run(tmp_path,[HEADER,geometry_product(kind)])
    assert result.returncode==0 and 'PRODUCT 1 valid=T' in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize('scenario',['typed-fields','visual-checkpoint'])
def test_geometry_fields_have_a_native_typed_query_and_checkpoint(tmp_path,scenario):
    probe=os.environ.get('ASTR_CATALOG_STATE_PROBE')
    if not probe: pytest.skip('Select the root-built state probe')
    path=tmp_path/'geometry.jsonl'
    path.write_text(json.dumps(HEADER)+'\n'+json.dumps(geometry_product())+'\n')
    result=subprocess.run([probe,str(path),scenario],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('ranks',[1,2,4])
def test_geometry_parameters_are_broadcast_without_worker_parsing(tmp_path,ranks):
    probe,launcher=os.environ.get('ASTR_CATALOG_MPI_PROBE'),os.environ.get('ASTR_CATALOG_MPIEXEC')
    if not probe or not launcher: pytest.skip('Select the catalog MPI probe and launcher')
    path=tmp_path/'geometry.jsonl';path.write_text(json.dumps(HEADER)+'\n'+json.dumps(geometry_product())+'\n')
    result=subprocess.run([launcher,'--oversubscribe','-n',str(ranks),probe,str(path),str(path),'geometry'],
        capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    for rank in range(ranks): assert f'RANK {rank} GEOMETRY_OK' in result.stdout


@pytest.mark.parametrize('fault',['display','field','format','empty-fields','wall'])
def test_bad_geometry_configuration_does_not_hide_the_next_record(tmp_path,fault):
    item=geometry_product()
    if fault=='display': item['display']=visual_product()['display']
    elif fault=='field': item['field']='velocity_x'
    elif fault=='format': item['output']['format']='csv'
    elif fault=='empty-fields': item['fields']=[]
    elif fault=='wall':
        item['kind']='wall_curve';item['sampling']=visual_product('wall_curve')['sampling']
        item.pop('extraction')
    result=run(tmp_path,[HEADER,item,product('healthy')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T' in result.stdout,result.stdout


@pytest.mark.parametrize('kind', ['slice', 'q_surface'])
def test_non_line_image_does_not_require_line_width(tmp_path, kind):
    item = visual_product(kind)
    del item['display']['line_width']
    result = run(tmp_path, [HEADER, item])
    assert 'valid=T' in result.stdout, result.stdout


def test_perspective_image_does_not_require_parallel_scale(tmp_path):
    item = visual_product()
    item['display']['projection'] = 'perspective'
    del item['display']['parallel_scale']
    result = run(tmp_path, [HEADER, item])
    assert 'valid=T' in result.stdout, result.stdout


@pytest.mark.parametrize('change', ['missing_camera', 'zero_normal', 'bad_range', 'unknown_map',
                                  'nonfinite_camera', 'bad_size', 'bad_format', 'unknown_quantity',
                                  'missing_integrator', 'bad_integration_steps', 'missing_threshold',
                                  'missing_maximum_steps','fractional_probe', 'unknown_wall'])
def test_bad_visual_record_is_local_not_a_catalog_failure(tmp_path, change):
    kind = ('streamlines' if change in ('missing_integrator', 'bad_integration_steps','missing_maximum_steps')
            else 'q_surface' if change == 'missing_threshold'
            else 'probe' if change == 'fractional_probe'
            else 'wall_curve' if change == 'unknown_wall' else 'slice')
    item = visual_product(kind, 'bad')
    if change == 'missing_camera': del item['display']['camera']
    elif change == 'zero_normal': item['sampling']['normal'] = [0, 0, 0]
    elif change == 'bad_range': item['display']['color_range'] = [1, 0]
    elif change == 'unknown_map': item['display']['color_map'] = 'guessed'
    elif change == 'nonfinite_camera': item['display']['camera'][0] = float('inf')
    elif change == 'bad_size': item['output']['image_size'][0] = 0
    elif change == 'bad_format': item['output']['formats'] = ['png']
    elif change == 'unknown_quantity': item['field'] = 'not_a_quantity'
    elif change == 'missing_integrator': del item['extraction']['integrator']
    elif change == 'missing_maximum_steps': del item['extraction']['maximum_steps']
    elif change == 'bad_integration_steps': item['extraction']['minimum_step'] = 1
    elif change == 'missing_threshold': del item['extraction']['isovalue']
    elif change == 'fractional_probe': item['sampling']['indices'][0] = 1.5
    elif change == 'unknown_wall': item['sampling']['side'] = 'guessed'
    result = run(tmp_path, [HEADER, item, product('good')])
    assert result.returncode == 0, result.stdout + result.stderr
    # Infinity is not JSON: a rejected syntax tree has no trustworthy parsed ID.
    expected = 'PRODUCT 1 valid=F' if change == 'nonfinite_camera' else 'PRODUCT 1 valid=F id=bad'
    assert expected in result.stdout, result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout, result.stdout


@pytest.mark.parametrize('kind', ['slice', 'probe', 'wall_curve', 'q_surface', 'streamlines'])
def test_visual_definition_survives_control_checkpoint(tmp_path, kind):
    probe = os.environ.get('ASTR_CATALOG_STATE_PROBE')
    if not probe:
        pytest.skip('Set ASTR_CATALOG_STATE_PROBE from root CMake')
    path = tmp_path / 'products.jsonl'
    path.write_text(json.dumps(HEADER) + '\n' + json.dumps(visual_product(kind)) + '\n')
    result = subprocess.run([str(Path(probe).resolve()), str(path), 'visual-checkpoint'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: runtime catalog visual-checkpoint' in result.stdout


@pytest.mark.parametrize('ranks', [1, 2, 4])
def test_visual_parameters_are_broadcast_without_reparsing(tmp_path, ranks):
    probe, launcher = os.environ.get('ASTR_CATALOG_MPI_PROBE'), os.environ.get('ASTR_CATALOG_MPIEXEC')
    if not probe or not launcher:
        pytest.skip('Set the catalog MPI probe and launcher')
    path = tmp_path / 'products.jsonl'
    path.write_text(json.dumps(HEADER) + '\n' + json.dumps(visual_product('q_surface')) + '\n')
    result = subprocess.run([launcher, '--oversubscribe', '-n', str(ranks), probe,
                             str(path), str(path), 'visual'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(ranks):
        assert f'RANK {rank} VISUAL_OK' in result.stdout
