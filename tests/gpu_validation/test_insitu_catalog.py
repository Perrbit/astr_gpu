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
