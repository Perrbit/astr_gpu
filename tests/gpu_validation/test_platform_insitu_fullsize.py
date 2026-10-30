"""Focused checks for scale preparation, honest timing and nonfatal image gaps."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import run_platform_insitu_fullsize as gate
from prepare_m12_local_case import FULL_INTERVALS, selected_nodes


def test_actual_image_publisher_preflight(tmp_path):
    assert gate.publication_preflight(tmp_path)['status'] == 'passed'


@pytest.mark.parametrize('value', ['0', '1'])
def test_preflight_uses_selected_image_formats(tmp_path, monkeypatch, value):
    monkeypatch.setenv('ASTR_INSITU_WRITE_EPS', value)
    result = gate.publication_preflight(tmp_path)
    assert result['image_formats'] == (['jpeg'] if value == '0' else ['jpeg', 'eps'])
    assert bool(list(tmp_path.rglob('*.eps'))) is (value == '1')
    assert not list(tmp_path.rglob('*.partial'))


def test_explicit_full_gpu_budget_allows_zero_reserve(tmp_path, monkeypatch):
    output = tmp_path/'zero_reserve'
    monkeypatch.setattr(gate.sys, 'argv', [__file__, '--preflight-only', '--output', str(output),
        '--device-budget', str(80*gate.GIB), '--reserve', '0'])
    gate.main()
    report = json.loads((output/'report.json').read_text())
    assert report['status'] == 'completed'
    assert report['resource_limits']['device_extra_budget_bytes'] == 80*gate.GIB
    assert report['resource_limits']['device_reserve_bytes'] == 0
    config = gate.render_configuration(SimpleNamespace(host_budget=32*gate.GIB,
        device_budget=80*gate.GIB, reserve=0, library='/catalyst'))
    assert 'device_budget_bytes=85899345920,device_reserve_bytes=0' in config


def test_negative_gpu_reserve_is_still_invalid(tmp_path, monkeypatch):
    output = tmp_path/'invalid_reserve'
    monkeypatch.setattr(gate.sys, 'argv', [__file__, '--preflight-only', '--output', str(output),
        '--reserve', '-1'])
    with pytest.raises(SystemExit) as error:
        gate.main()
    assert error.value.code == 2
    assert not output.exists()


def test_original_grid_selection_is_identity():
    ix, iy, x = selected_nodes({'x': np.linspace(0., 1100., 2251)[None, :]}, FULL_INTERVALS)
    np.testing.assert_array_equal(ix, np.arange(2251))
    np.testing.assert_array_equal(iy, np.arange(261))
    assert x.size == 2251


def test_mpi_launcher_preserves_dispatch_symlink_name(tmp_path, monkeypatch):
    import run_platform_insitu_acceptance as acceptance
    target = tmp_path/'env.sh'
    target.write_text('#!/bin/sh\nexit 0\n')
    target.chmod(0o700)
    launcher = tmp_path/'mpiexec'
    launcher.symlink_to(target.name)
    captured = []

    def dependency(args):
        captured.append(args.mpiexec)
        raise RuntimeError('stop before GPU probes')

    monkeypatch.setattr(acceptance, 'dependency_gate', dependency)
    monkeypatch.setattr(gate, 'publication_preflight', lambda directory: {'status': 'passed'})
    monkeypatch.setattr(gate.sys, 'argv', [__file__, '--source', str(tmp_path),
        '--output', str(tmp_path/'run'), '--executable', str(target),
        '--diagnostic-executable', str(target), '--mpiexec', str(launcher),
        '--library', str(tmp_path), '--probe-directory', str(tmp_path),
        '--product-catalog',str(tmp_path/'approved.jsonl')])
    with pytest.raises(RuntimeError, match='stop before GPU probes'):
        gate.main()
    assert captured == [launcher.absolute()]


def test_checkpoint_preparation_creates_private_empty_root(tmp_path, monkeypatch):
    def prepare(source, case, backend, shape):
        (case/'datin').mkdir(parents=True)
        (case/'outdat').mkdir()
        (case/'fixture.json').write_text('{}')
        (case/'datin/controller').write_text('# maxstep,feqchkpt\n9,1,9999,9999,1,9999\n')
        (case/'datin/input.output').write_text('&checkpoint\n enabled=f\n/\n')
    monkeypatch.setattr(gate, 'prepare', prepare)
    args = SimpleNamespace(source=tmp_path, host_budget=1, device_budget=1, reserve=1, library='/lib')
    case = tmp_path/'small'
    gate.prepare_case(args, case, (64,32,24), 10, False, checkpoint=True)
    root = case/'outdat/new'
    assert root.is_dir() and not list(root.iterdir())
    assert root.stat().st_mode & 0o777 == 0o700
    assert 'interval_steps=10' in (case/'datin/input.output').read_text()


def test_complete_iteration_excludes_warmup_not_rendering():
    lines = []
    for rank in range(4):
        for step in range(1, 11):
            elapsed = 100. if step <= 5 else rank+1.
            lines += [f'ASTR_INSITU_STAGE_TIMING completed_step_inclusive {step} {rank} {elapsed}',
                      f'ASTR_GPU_RANK_RK_TIMING {rank} {step-1} 0.25 0.5 0.75']
    record = gate.timing_summary('\n'.join(lines), 10)
    assert record['complete_iteration_seconds'] == 20.
    assert record['complete_iteration_median_seconds'] == 4.
    assert record['rk_integration_seconds'] == 2.5
    assert record['rk_prepare_plus_integration_seconds'] == 3.75
    with pytest.raises(ValueError, match='Incomplete four-rank'):
        gate.timing_summary('\n'.join(lines[:-2]), 10)


def product_case(tmp_path):
    case = tmp_path/'case'
    case.mkdir()
    definitions=gate.m12_short_products(interval=10)
    gate.write_catalog(case/'input.products.jsonl',definitions)
    out=case/'outdat/new';out.mkdir(parents=True)
    (out/'products.status.json').write_text(json.dumps(dict(products=[dict(id=row['id'],state='active') for row in definitions])))
    for row in definitions:
        if row['output']['channel']!='image': continue
        directory=out/'render_products'/row['id']/'version000000000001/segment000000000001'
        directory.mkdir(parents=True)
        for step in (0,10):
            path=directory/f'frame.step{step:012d}.jpeg'
            Image.new('RGB',tuple(row['output']['image_size']),'white').save(path)
            receipt=dict(id=row['id'],definition=row,phase='completed_step',geometry_host_bytes=0,
                field_d2h_bytes=0,processing_backend='device',product=dict(image='published'),
                geometry_empty=row['kind']=='q_surface')
            for rank in range(4):
                path.with_name(f'{path.stem}.rank{rank:08d}.json').write_text(json.dumps(receipt))
    return case


def test_empty_q_and_pale_main_image_are_not_scientific_failures(tmp_path):
    record = gate.inspect_products(product_case(tmp_path), 10)
    assert record['publication_complete']
    assert any('empty geometry' in warning for warning in record['warnings'])


def test_missing_image_is_recorded_not_promoted_to_pass_or_abort(tmp_path):
    case = product_case(tmp_path)
    next((case/'outdat/new/render_products/section').rglob('*000000000010.jpeg')).unlink()
    record = gate.inspect_products(case, 10)
    assert not record['publication_complete']
    assert any('Missing JPEG image' in warning for warning in record['warnings'])


def test_product_inspection_respects_jpeg_only(tmp_path, monkeypatch):
    case = product_case(tmp_path)
    for path in case.rglob('*.eps'):
        path.unlink()
    monkeypatch.setenv('ASTR_INSITU_WRITE_EPS', '0')
    assert gate.inspect_products(case, 10)['publication_complete']
    next((case/'outdat/new/render_products/section').rglob('*000000000010.jpeg')).unlink()
    assert not gate.inspect_products(case, 10)['publication_complete']


def test_default_product_inspection_does_not_require_eps(tmp_path, monkeypatch):
    monkeypatch.delenv('ASTR_INSITU_WRITE_EPS', raising=False)
    case = product_case(tmp_path)
    assert gate.inspect_products(case, 10)['publication_complete']


def test_host_geometry_still_fails(tmp_path):
    case = product_case(tmp_path)
    path=next((case/'outdat/new/render_products/section').rglob('*.rank00000000.json'))
    record = json.loads(path.read_text())
    record['geometry_host_bytes'] = 8
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='host geometry'):
        gate.inspect_products(case, 10)


def test_fullsize_configuration_keeps_optional_images_optional():
    args = SimpleNamespace(host_budget=32*gate.GIB, device_budget=32*gate.GIB,
                           reserve=8*gate.GIB, library='/catalyst')
    assert 'product_steps' not in gate.render_configuration(args)
    on=gate.render_catalog(args,(64,32,24),True);off=gate.render_catalog(args,(64,32,24),False)
    assert len(on)==6 and len(off)==2 and all(row['output']['channel']=='data' for row in off)
    assert 'statistics=f' in gate.render_configuration(args)


def test_full_grid_requires_its_own_reviewed_catalog(tmp_path):
    args=SimpleNamespace(m12_products='all')
    with pytest.raises(ValueError,match='explicitly reviewed --product-catalog'):
        gate.render_catalog(args,FULL_INTERVALS)
    records=gate.m12_short_products()[:1]
    records[0]['sampling']['index']=192
    records[0]['sampling']['index_ranges']=dict(i=[0,2250],j=[0,260])
    path=tmp_path/'reviewed.jsonl';gate.write_catalog(path,records);args.product_catalog=path
    assert gate.render_catalog(args,FULL_INTERVALS)==records
    assert gate.render_catalog(args,FULL_INTERVALS,False)==[]
    empty=tmp_path/'off.jsonl';gate.write_catalog(empty,gate.render_catalog(args,FULL_INTERVALS,False))
    assert empty.read_text().splitlines()==['{"catalog_schema": "astr-insitu-products-v1"}']


def test_slices_wall_configuration_excludes_volume_geometry():
    args = SimpleNamespace(host_budget=32*gate.GIB, device_budget=80*gate.GIB,
        reserve=0, library='/catalyst', m12_products='slices-wall')
    config = gate.render_configuration(args)
    assert 'product_ids' not in config
    assert all(row['kind'] not in ('q_surface','streamlines') for row in gate.render_catalog(args,(64,32,24)))


def test_slices_wall_publication_does_not_require_disabled_products(tmp_path):
    case = product_case(tmp_path)
    directory=case/'outdat/new/render_products'
    for name in ('vortices','trajectories'):
        for path in (directory/name).rglob('frame.*'):
            path.unlink()
    assert not gate.inspect_products(case, 10)['publication_complete']
    definitions=gate.render_catalog(SimpleNamespace(m12_products='slices-wall'),(64,32,24))
    for row in definitions: row['schedule']['interval']=10
    gate.write_catalog(case/'input.products.jsonl',definitions)
    assert gate.inspect_products(case,10,'slices-wall')['publication_complete']


def test_nscbc_preparation_changes_only_outlet_and_top(tmp_path, monkeypatch):
    original = '# bctype\n11, prof\n21\n41, 10.d0\n50\n1\n1\n'
    def prepare(source, case, backend, shape):
        (case/'datin').mkdir(parents=True)
        (case/'outdat').mkdir()
        (case/'fixture.json').write_text('{}')
        (case/'datin/input.dat').write_text(original)
        (case/'datin/controller').write_text('# maxstep,feqchkpt\n9,1,9999,9999,1,9999\n')
    monkeypatch.setattr(gate, 'prepare', prepare)
    args = SimpleNamespace(source=tmp_path, host_budget=1, device_budget=1, reserve=0,
        library='/lib', m12_boundary='nscbc', m12_products='slices-wall')
    case = tmp_path/'nscbc'
    gate.prepare_case(args, case, (64,32,24), 10, True)
    assert (case/'datin/input.dat').read_text() == original.replace('\n21\n', '\n22,1.d0\n').replace('\n50\n', '\n52\n')
    assert json.loads((case/'fixture.json').read_text())['boundary'] == 'nscbc'
    env = gate.runtime_environment('device-aware', 'nscbc')
    assert env['ASTR_NSCBC_FARFIELD_MODE'] == 'compatibility'
    assert env['ASTR_GPU_HALO_TRANSPORT'] == 'device-aware'
