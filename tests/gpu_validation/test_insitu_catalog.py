"""Host JSONL interface; no flow advancement, GPU or Catalyst."""
import json
import os
from pathlib import Path
import subprocess

import pytest

PROBE = os.environ.get('ASTR_CATALOG_PROBE')
pytestmark = pytest.mark.skipif(not PROBE, reason='Set ASTR_CATALOG_PROBE from root CMake')
HEADER = {'catalog_schema': 'astr-insitu-products-v1'}


def product(name='nearwall'):
    return dict(id=name, kind='slice', enable_request=1, source='instantaneous',
                fields=['velocity_x', 'density_gradient_magnitude'],
                sampling=dict(mode='grid_index', axis='j', index=20,
                              index_ranges=dict(i=[0, 2250], k=[0, 480])),
                output=dict(channel='data', format='hdf5-xdmf'),
                schedule=dict(mode='steps', interval=100, events=[], windows=[]))


def run(tmp_path, records, crlf=False, numq=None):
    path = tmp_path / 'products.jsonl'
    lines = [line if isinstance(line, str) else json.dumps(line) for line in records]
    path.write_bytes((('\r\n' if crlf else '\n').join(lines) + '\n').encode())
    command = [str(Path(PROBE).resolve()), str(path)]
    if numq is not None:
        command.append(str(numq))
    return subprocess.run(command, capture_output=True, text=True)


def test_valid_index_product(tmp_path):
    result = run(tmp_path, [HEADER, product()])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRODUCT 1 valid=T id=nearwall' in result.stdout
    assert 'axis=2' in result.stdout and 'index=20' in result.stdout
    assert 'field=velocity_x' in result.stdout
    assert 'field=density_gradient_magnitude' in result.stdout


@pytest.mark.parametrize('source,fields', [('reynolds', ['velocity_x', 'pressure', 'temperature']),
                                          ('favre', ['velocity_x', 'velocity_y', 'velocity_z'])])
def test_fixed_j_fluctuation_definition(tmp_path, source, fields):
    item = product('fluctuation')
    item.update(source=source, statistics_source='selected-time', fluctuation=True, fields=fields)
    result = run(tmp_path, [HEADER, item])
    assert 'PRODUCT 1 valid=T' in result.stdout and 'fluctuation=T' in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize('change', [dict(fluctuation=1), dict(fluctuation='true'),
    dict(source='instantaneous'), dict(statistics_source='velocity-time'),
    dict(fields=['density']), dict(fields=['Q_rs']), dict(source='favre', fields=['pressure']),
    dict(source='favre', fields=['temperature']), dict(sampling=dict(mode='grid_index', axis='k', index=2,
        index_ranges=dict(i=[0,2250],j=[0,260])))])
def test_bad_fluctuation_is_product_local(tmp_path, change):
    item = product('bad-fluctuation')
    item.update(source='reynolds', statistics_source='selected-time', fluctuation=True, fields=['velocity_x'])
    item.update(change)
    result = run(tmp_path, [HEADER, item, product('good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


def test_fluctuation_is_not_a_fluctuation_of_speed(tmp_path):
    from test_insitu_catalog_products import visual_product
    item = visual_product(name='speed')
    item.update(source='reynolds',statistics_source='selected-time',fluctuation=True)
    item['sampling'] = product()['sampling']
    result = run(tmp_path,[HEADER,item,product('good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.skipif(not os.environ.get('ASTR_CATALOG_STATE_PROBE'),reason='Select the root control probe')
def test_fluctuation_definition_checkpoint_roundtrip(tmp_path):
    item = product('fluctuation')
    item.update(source='reynolds',statistics_source='selected-time',fluctuation=True,fields=['velocity_x'])
    path = tmp_path/'fluctuation.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in [HEADER,item])+'\n')
    result = subprocess.run([os.environ['ASTR_CATALOG_STATE_PROBE'],str(path),'visual-checkpoint'],
                            capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS: runtime catalog visual-checkpoint' in result.stdout


def test_complete_wall_transport_and_temperature_gradient_fields(tmp_path):
    item = product('wall-full')
    item['sampling'].update(index=0)
    item['fields'] = ['temperature_gradient_x', 'temperature_gradient_y', 'temperature_gradient_z',
                      'wall_shear_x', 'wall_shear_z', 'wall_heat_into_wall', 'dynamic_viscosity']
    result = run(tmp_path, [HEADER, item])
    assert result.returncode == 0 and 'PRODUCT 1 valid=T id=wall-full' in result.stdout, result.stdout + result.stderr
    assert [line[6:] for line in result.stdout.splitlines() if line.startswith('field=')] == item['fields']


@pytest.mark.parametrize('mode,krange', [('wall_span', [0, 480]), ('wall_line', [480, 480])])
def test_reynolds_complete_wall_profile(tmp_path, mode, krange):
    item = product('wall-profile')
    item.update(kind='wall_profile', source='reynolds', statistics_source='selected-time',
                fields=['density', 'temperature', 'wall_shear_x', 'wall_shear_z', 'wall_heat_into_wall', 'dynamic_viscosity'])
    item['sampling'] = dict(mode=mode, axis='j', index=0, index_ranges=dict(i=[0, 2250], k=krange))
    item['output'] = dict(channel='data', format='csv')
    result = run(tmp_path, [HEADER, item])
    assert result.returncode == 0 and 'PRODUCT 1 valid=T id=wall-profile' in result.stdout, result.stdout + result.stderr


def test_wall_flux_is_not_a_flux_of_primitive_means(tmp_path):
    item = product('bad-wall')
    item.update(source='reynolds', statistics_source='velocity-time', fields=['wall_shear_x'])
    item['sampling']['index'] = 0
    result = run(tmp_path, [HEADER, item, product('good')])
    assert 'PRODUCT 1 valid=F id=bad-wall' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


def test_selected_wall_footprint_is_a_mean_of_wall_quantities(tmp_path):
    item = product('wall-footprint')
    item.update(source='reynolds',statistics_source='selected-time',
                fields=['density','temperature','wall_shear_x','wall_shear_z','wall_heat_into_wall','dynamic_viscosity'])
    item['sampling']['index'] = 0
    result = run(tmp_path,[HEADER,item])
    assert 'PRODUCT 1 valid=T id=wall-footprint' in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize('source,provider,index,fields',[('favre','selected-time',0,['wall_shear_x']),
    ('reynolds','velocity-time',0,['wall_shear_x']),('reynolds','selected-time',1,['wall_shear_x']),
    ('reynolds','selected-time',0,['wall_shear_x','velocity_x'])])
def test_invalid_wall_mean_support_is_product_local(tmp_path,source,provider,index,fields):
    item=product('bad-wall-mean')
    item.update(source=source,statistics_source=provider,fields=fields)
    item['sampling']['index']=index
    result=run(tmp_path,[HEADER,item,product('good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


def normalized_wall(reference):
    item = product('wall-reference')
    item.update(kind='wall_profile', source='reynolds', statistics_source='selected-time', fields=['wall_shear_x'])
    item['sampling'] = dict(mode='wall_span', axis='j', index=0, index_ranges=dict(i=[0,2250], k=[0,480]))
    item['output'] = dict(channel='data', format='csv')
    item['wall_reference'] = reference
    return item


@pytest.mark.parametrize('reference,mode', [({}, 0), ({'recovery_temperature': 26.92}, 1),
                                         ({'recovery_factor': .9}, 2), ({'recovery_factor': 0}, 2)])
def test_explicit_wall_references(tmp_path, reference, mode):
    result = run(tmp_path, [HEADER, normalized_wall(reference)])
    assert 'PRODUCT 1 valid=T' in result.stdout, result.stdout + result.stderr
    assert f'wall_reference_flags=1 {mode}' in result.stdout


@pytest.mark.parametrize('reference', [False, 12, [], {'unknown': 1}, {'recovery_temperature': 0},
                                     {'recovery_temperature': -1}, {'recovery_factor': -1},
                                     {'recovery_factor': True}, {'recovery_temperature': '20'},
                                     {'recovery_temperature': 20, 'recovery_factor': .9}])
def test_bad_wall_reference_is_product_local(tmp_path, reference):
    result = run(tmp_path, [HEADER, normalized_wall(reference), product('good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.parametrize('ranks', (1, 2, 4))
def test_wall_reference_broadcast(tmp_path, ranks):
    probe = os.environ.get('ASTR_CATALOG_MPI_PROBE')
    launcher = os.environ.get('ASTR_CATALOG_MPIEXEC')
    if not probe or not launcher:
        pytest.skip('Set the root-CMake MPI catalog probe and launcher')
    formal = tmp_path / 'wall.jsonl'
    formal.write_text('\n'.join(json.dumps(row) for row in [HEADER, normalized_wall({'recovery_factor': .9})])+'\n')
    candidate, temperature = tmp_path / 'factor.jsonl', tmp_path / 'temperature.jsonl'
    candidate.write_text('\n'.join(json.dumps(row) for row in [HEADER, normalized_wall({'recovery_factor': .8})])+'\n')
    temperature.write_text('\n'.join(json.dumps(row) for row in [HEADER, normalized_wall({'recovery_temperature': 26.92})])+'\n')
    result = subprocess.run([launcher, '--mca', 'coll_hcoll_enable', '0', '--oversubscribe', '-np', str(ranks),
                             str(Path(probe).resolve()), str(formal), str(candidate), 'wall-reference', str(temperature)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count('WALL_REFERENCE_OK') == ranks


@pytest.mark.parametrize('mode,krange', [('station_span', [0, 480]), ('station_line', [17, 17])])
def test_valid_selected_time_station_profile(tmp_path, mode, krange):
    item = product('station')
    item.update(kind='station_profile', source='reynolds', statistics_source='selected-time',
                fields=['velocity_x', 'pressure', 'temperature'])
    item['sampling'] = dict(mode=mode, axis='i', index=700, index_ranges=dict(j=[0, 260], k=krange))
    item['output'] = dict(channel='data', format='csv')
    result = run(tmp_path, [HEADER, item])
    assert result.returncode == 0 and 'PRODUCT 1 valid=T id=station' in result.stdout, result.stdout + result.stderr
    assert 'axis=1' in result.stdout and 'index=700' in result.stdout


def thickness_product():
    item = product('thickness')
    item.update(kind='station_profile', source='favre', statistics_source='selected-time',
                fields=['velocity_x', 'pressure', 'temperature'], thickness=dict(edge_band=[220, 240]))
    item['sampling'] = dict(mode='station_span', axis='i', index=700,
                            index_ranges=dict(j=[0, 260], k=[0, 480]))
    item['output'] = dict(channel='data', format='csv')
    return item


@pytest.mark.parametrize('upper', [None, 12.5])
def test_thickness_profile_options(tmp_path, upper):
    item = thickness_product()
    if upper is not None:
        item['thickness']['explicit_upper'] = upper
    result = run(tmp_path, [HEADER, item])
    assert result.returncode == 0 and 'PRODUCT 1 valid=T' in result.stdout, result.stdout + result.stderr
    assert 'thickness_band=220 240' in result.stdout
    assert f'thickness_flags=1 {int(upper is not None)}' in result.stdout


@pytest.mark.parametrize('change', ['null', 'not_object', 'missing_band', 'bad_key', 'float_index',
                                  'bool_index', 'reversed', 'negative', 'wall_band', 'outside', 'off_wall',
                                  'reynolds', 'missing_field', 'upper_nonfinite', 'upper_bool'])
def test_invalid_thickness_retains_other_product(tmp_path, change):
    item = thickness_product()
    if change == 'null': item['thickness'] = None
    elif change == 'not_object': item['thickness'] = []
    elif change == 'missing_band': item['thickness'] = {}
    elif change == 'bad_key': item['thickness']['estimate'] = 'auto'
    elif change == 'float_index': item['thickness']['edge_band'] = [220.5, 240]
    elif change == 'bool_index': item['thickness']['edge_band'] = [True, 240]
    elif change == 'reversed': item['thickness']['edge_band'] = [240, 220]
    elif change == 'negative': item['thickness']['edge_band'] = [-1, 220]
    elif change == 'wall_band': item['thickness']['edge_band'] = [0, 0]
    elif change == 'outside': item['thickness']['edge_band'] = [220, 261]
    elif change == 'off_wall': item['sampling']['index_ranges']['j'] = [1, 260]
    elif change == 'reynolds': item['source'] = 'reynolds'
    elif change == 'missing_field': item['fields'] = ['velocity_x', 'temperature']
    elif change == 'upper_nonfinite': item['thickness']['explicit_upper'] = 'nan'
    elif change == 'upper_bool': item['thickness']['explicit_upper'] = True
    result = run(tmp_path, [HEADER, item, product('good')])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRODUCT 1 valid=F id=thickness' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.parametrize('ranks', [1, 2, 4])
def test_thickness_broadcast_version_and_restore(tmp_path, ranks):
    probe = os.environ.get('ASTR_CATALOG_MPI_PROBE')
    if not probe:
        pytest.skip('Select the root-CMake MPI catalog probe')
    item = thickness_product()
    formal, candidate = tmp_path/'formal.jsonl', tmp_path/'candidate.jsonl'
    formal.write_text('\n'.join(json.dumps(row) for row in [HEADER, item])+'\n')
    item['thickness'] = dict(edge_band=[221, 241], explicit_upper=12.5)
    candidate.write_text('\n'.join(json.dumps(row) for row in [HEADER, item])+'\n')
    mpi = os.environ.get('ASTR_CATALOG_MPIEXEC', 'mpirun')
    result = subprocess.run([mpi, '--mca', 'coll_hcoll_enable', '0', '--oversubscribe', '-np', str(ranks),
                             str(Path(probe).resolve()), str(formal), str(candidate), 'thickness'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('THICKNESS_OK') == ranks


@pytest.mark.parametrize('source', ['reynolds', 'favre'])
def test_station_wall_units_reference(tmp_path, source):
    item = product('station')
    item.update(kind='station_profile', source=source, statistics_source='selected-time',
                fields=['velocity_x', 'pressure', 'temperature'], wall_units='wall-reference')
    item['sampling'] = dict(mode='station_span', axis='i', index=700,
                            index_ranges=dict(j=[0, 260], k=[0, 480]))
    item['output'] = dict(channel='data', format='csv')
    result = run(tmp_path, [HEADER, normalized_wall({}), item])
    assert result.returncode == 0 and 'PRODUCT 2 valid=T id=station' in result.stdout, result.stdout + result.stderr
    assert 'wall_units=wall-reference' in result.stdout


@pytest.mark.parametrize('value', [None, {}, [], 7, True, '', 'bad\nID', 'bad\x00ID'])
def test_invalid_wall_units_is_product_local(tmp_path, value):
    item = product('bad')
    item.update(kind='station_profile', source='reynolds', statistics_source='selected-time',
                fields=['velocity_x'], wall_units=value)
    item['sampling'] = dict(mode='station_span', axis='i', index=700,
                            index_ranges=dict(j=[0, 260], k=[0, 480]))
    item['output'] = dict(channel='data', format='csv')
    result = run(tmp_path, [HEADER, item, product('good')])
    assert 'PRODUCT 1 valid=F id=bad' in result.stdout and 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.parametrize('ranks', [1, 2, 4])
def test_wall_units_broadcast_version_and_restore(tmp_path, ranks):
    probe = os.environ.get('ASTR_CATALOG_MPI_PROBE')
    if not probe:
        pytest.skip('Set ASTR_CATALOG_MPI_PROBE')
    item = product('station')
    item.update(kind='station_profile', source='favre', statistics_source='selected-time',
                fields=['velocity_x'], wall_units='reference-a')
    item['sampling'] = dict(mode='station_span', axis='i', index=700,
                            index_ranges=dict(j=[0, 260], k=[0, 480]))
    item['output'] = dict(channel='data', format='csv')
    formal, candidate = tmp_path/'formal.jsonl', tmp_path/'candidate.jsonl'
    formal.write_text('\n'.join(json.dumps(row) for row in [HEADER, item])+'\n')
    item['wall_units'] = 'reference-b'
    candidate.write_text('\n'.join(json.dumps(row) for row in [HEADER, item])+'\n')
    mpi = os.environ.get('ASTR_CATALOG_MPIEXEC', 'mpirun')
    result = subprocess.run([mpi, '--mca', 'coll_hcoll_enable', '0', '--oversubscribe', '-np', str(ranks),
                             str(Path(probe).resolve()), str(formal), str(candidate), 'wall-units'],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('WALL_UNITS_OK') == ranks


@pytest.mark.parametrize('change', ['instantaneous', 'favre_density', 'display_field', 'point_interval',
                                  'zero_span', 'wrong_axis', 'image'])
def test_invalid_station_does_not_disable_valid_slice(tmp_path, change):
    item = product('station')
    item.update(kind='station_profile', source='reynolds', statistics_source='selected-time', fields=['pressure'])
    item['sampling'] = dict(mode='station_span', axis='i', index=700,
                            index_ranges=dict(j=[0, 260], k=[0, 480]))
    item['output'] = dict(channel='data', format='csv')
    if change == 'instantaneous': item['source'] = 'instantaneous'
    elif change == 'favre_density': item.update(source='favre', fields=['density'])
    elif change == 'display_field': item['field'] = 'pressure'
    elif change == 'point_interval': item['sampling']['mode'] = 'station_line'
    elif change == 'zero_span': item['sampling']['index_ranges']['k'] = [1, 1]
    elif change == 'wrong_axis': item['sampling'].update(axis='j', index_ranges=dict(i=[0, 2250], k=[0, 480]))
    elif change == 'image': item['output'] = dict(channel='image', format='jpeg')
    result = run(tmp_path, [HEADER, item, product('good')])
    assert result.returncode == 0 and 'PRODUCT 1 valid=F id=station' in result.stdout, result.stdout + result.stderr
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


def test_bad_record_does_not_swallow_later_record(tmp_path):
    result = run(tmp_path, [HEADER, '{"id":"broken",', product('good')])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRODUCT 1 valid=F' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.parametrize('records', [[], [''], [{}], [dict(catalog_schema='unknown')],
                                     ['{"catalog_schema":'], [HEADER | {'unknown': 1}]])
def test_bad_header_is_file_error(tmp_path, records):
    result = run(tmp_path, records)
    assert result.returncode != 0 and 'FILE_ERROR' in result.stdout


def test_header_only_is_explicit_empty_catalog(tmp_path):
    result = run(tmp_path, [HEADER], crlf=True)
    assert result.returncode == 0 and 'FILE_OK 0' in result.stdout


@pytest.mark.parametrize('request_number', [1, 9007199254740993, 9223372036854775807])
def test_request_integer_is_not_rounded_through_double(tmp_path, request_number):
    item = product()
    item['enable_request'] = request_number
    result = run(tmp_path, [HEADER, item])
    assert 'valid=T' in result.stdout, result.stdout + result.stderr
    assert f'request={request_number}' in result.stdout


@pytest.mark.parametrize('request_number', [0, -1, 1.5, 9223372036854775808, '1', True])
def test_bad_request_rejects_only_that_product(tmp_path, request_number):
    item = product('bad')
    item['enable_request'] = request_number
    result = run(tmp_path, [HEADER, item, product('good')])
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PRODUCT 1 valid=F id=bad' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


def test_duplicate_ids_disable_all_ambiguous_records(tmp_path):
    result = run(tmp_path, [HEADER, product('same'), product('same'), product('good')])
    assert 'PRODUCT 1 valid=F id=same' in result.stdout
    assert 'PRODUCT 2 valid=F id=same' in result.stdout
    assert 'PRODUCT 3 valid=T id=good' in result.stdout


@pytest.mark.parametrize('change', ['unknown_field', 'unknown_key', 'missing_source', 'backwards_range',
                                  'fixed_axis_range', 'empty_fields', 'bad_format', 'bad_source', 'stride'])
def test_invalid_identifiable_product(tmp_path, change):
    item = product('bad')
    if change == 'unknown_field': item['fields'] = ['not_a_field']
    elif change == 'unknown_key': item['guessed_default'] = 1
    elif change == 'missing_source': del item['source']
    elif change == 'backwards_range': item['sampling']['index_ranges']['i'] = [3, 1]
    elif change == 'fixed_axis_range': item['sampling']['index_ranges']['j'] = [0, 1]
    elif change == 'empty_fields': item['fields'] = []
    elif change == 'bad_format': item['output']['format'] = 'jpeg'
    elif change == 'bad_source': item['source'] = 'unavailable'
    elif change == 'stride': item['sampling']['stride'] = 2
    result = run(tmp_path, [HEADER, item, product('good')])
    assert 'PRODUCT 1 valid=F id=bad' in result.stdout, result.stdout + result.stderr
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


def test_field_order_and_duplicates_are_canonical(tmp_path):
    item = product()
    item['fields'] = ['density_gradient_magnitude', 'velocity_x', 'velocity_x']
    result = run(tmp_path, [HEADER, item])
    assert result.returncode == 0 and 'valid=T' in result.stdout
    assert [x for x in result.stdout.splitlines() if x.startswith('field=')] == [
        'field=velocity_x', 'field=density_gradient_magnitude']


def test_physical_time_and_mean_source(tmp_path):
    item = product()
    item.update(source='favre', statistics_source='insitu_velocity')
    item['schedule'] = dict(mode='time', interval=.125, events=[], windows=[])
    result = run(tmp_path, [HEADER, item])
    assert result.returncode == 0 and 'valid=T' in result.stdout
    assert 'source=favre' in result.stdout
    assert 'steps=0' in result.stdout and float(result.stdout.split('time=')[1].split()[0]) == .125


def test_long_bad_line_does_not_truncate_next_record(tmp_path):
    item = product('bad')
    item['unknown'] = 'x' * 12000
    result = run(tmp_path, [HEADER, item, product('good')], crlf=True)
    assert 'PRODUCT 1 valid=F id=bad' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


def test_unicode_id_and_strict_json(tmp_path):
    result = run(tmp_path, [HEADER, product('截面'), '// not JSON', product('good')])
    assert 'PRODUCT 1 valid=T id=截面' in result.stdout
    assert 'PRODUCT 2 valid=F' in result.stdout
    assert 'PRODUCT 3 valid=T id=good' in result.stdout


@pytest.mark.parametrize('axis,index', [('i', 2251), ('j', 261), ('k', 481)])
def test_global_grid_not_rank_local_bounds(tmp_path, axis, index):
    item = product('bad')
    item['sampling'].update(axis=axis, index=index,
                            index_ranges={a: [0, 1] for a in 'ijk' if a != axis})
    result = run(tmp_path, [HEADER, item, product('good')])
    assert 'PRODUCT 1 valid=F id=bad' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


def test_air5_fields_require_actual_source_layout(tmp_path):
    item = product()
    item['fields'] = ['mass_fraction_NO', 'vibrational_temperature']
    absent = run(tmp_path, [HEADER, item], numq=5)
    assert 'valid=F' in absent.stdout and 'AIR5 field' in absent.stdout
    present = run(tmp_path, [HEADER, item], numq=11)
    assert 'valid=T' in present.stdout, present.stdout + present.stderr


def test_duplicate_json_key_is_not_silently_overwritten(tmp_path):
    record = json.dumps(product()).replace('"kind": "slice"', '"kind": "slice", "kind": "unknown"')
    result = run(tmp_path, [HEADER, record, product('good')])
    assert 'PRODUCT 1 valid=F' in result.stdout and 'duplicate JSON object key' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.parametrize('interval', [0, -1, float('nan'), float('inf'), True])
def test_invalid_physical_interval(tmp_path, interval):
    item = product('bad')
    item['schedule'] = dict(mode='time', interval=interval)
    result = run(tmp_path, [HEADER, item, product('good')])
    assert 'PRODUCT 1 valid=F' in result.stdout
    assert 'PRODUCT 2 valid=T id=good' in result.stdout


@pytest.mark.parametrize('scenario', ['first-frame', 'cadence', 'definition', 'invalid-recovery',
                                    'remove-readd', 'resource-retry', 'waiting', 'file-rejection',
                                    'checkpoint', 'checkpoint-empty', 'checkpoint-bad-format','adaptive'])
def test_runtime_state_transitions(tmp_path, scenario):
    probe = os.environ.get('ASTR_CATALOG_STATE_PROBE')
    if not probe:
        pytest.skip('Set ASTR_CATALOG_STATE_PROBE from root CMake')
    path = tmp_path / 'products.jsonl'
    path.write_text(json.dumps(HEADER) + '\n' + json.dumps(product()) + '\n')
    result = subprocess.run([str(Path(probe).resolve()), str(path), scenario], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: runtime catalog ' + scenario in result.stdout


@pytest.mark.parametrize('np', [1, 2, 4])
@pytest.mark.parametrize('mode', ['change', 'bad-header'])
def test_root_only_file_watch_and_typed_mpi_metadata(tmp_path, np, mode):
    probe, launcher = os.environ.get('ASTR_CATALOG_MPI_PROBE'), os.environ.get('ASTR_CATALOG_MPIEXEC')
    if not probe or not launcher:
        pytest.skip('Set ASTR_CATALOG_MPI_PROBE and ASTR_CATALOG_MPIEXEC')
    formal, candidate = tmp_path / 'products.jsonl', tmp_path / 'candidate.jsonl'
    formal.write_text(json.dumps(HEADER) + '\n' + json.dumps(product()) + '\n')
    changed = product()
    changed['sampling']['index'] = 21
    candidate.write_text(json.dumps(HEADER) + '\n' + json.dumps(changed) + '\n' if mode == 'change'
                         else '{"catalog_schema":"unsupported"}\n')
    result = subprocess.run([launcher, '--oversubscribe', '-n', str(np), probe,
                             str(formal), str(candidate), mode], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'RANK 0 READS 2 CHECKS 3' in result.stdout
    if mode=='bad-header':
        status=json.loads(Path(str(formal)+'.status.json').read_text())
        assert status['catalog_error']
    for rank in range(1, np):
        assert f'RANK {rank} READS 0 CHECKS 0' in result.stdout


@pytest.mark.parametrize('np', [1, 2, 4])
@pytest.mark.parametrize('mode', ['restart-ignore', 'feedback'])
def test_checkpoint_authority_and_status_feedback(tmp_path, np, mode):
    probe, launcher = os.environ.get('ASTR_CATALOG_MPI_PROBE'), os.environ.get('ASTR_CATALOG_MPIEXEC')
    if not probe or not launcher:
        pytest.skip('Set ASTR_CATALOG_MPI_PROBE and ASTR_CATALOG_MPIEXEC')
    formal, candidate, repeated = [tmp_path / name for name in ('products.jsonl', 'candidate.jsonl', 'repeat.jsonl')]
    formal.write_text(json.dumps(HEADER) + '\n' + json.dumps(product()) + '\n')
    item = product()
    item['sampling']['index'] = 21
    changed = json.dumps(HEADER) + '\n' + json.dumps(item) + '\n'
    candidate.write_text(changed)
    repeated.write_text(changed)
    result = subprocess.run([launcher, '--oversubscribe', '-n', str(np), probe, str(formal),
                             str(candidate), mode, str(repeated)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    if mode == 'feedback':
        status = json.loads(Path(str(formal) + '.status.json').read_text())
        assert status['catalog_error'] == '' and status['step'] == 2
        entry, = status['products']
        assert entry['id'] == 'nearwall' and entry['version'] == 1
        assert entry['state'] == 'resource-disabled' and entry['last_output_step'] == 1
        assert entry['reason'] == 'injected "disk" error\nline2'
        assert not list(tmp_path.glob('*.tmp.*'))
    else:
        for rank in range(np):
            assert f'RANK {rank} RESTORE_OK' in result.stdout
