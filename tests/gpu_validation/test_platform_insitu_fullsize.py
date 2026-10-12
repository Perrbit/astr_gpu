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
        '--library', str(tmp_path), '--probe-directory', str(tmp_path)])
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
    monkeypatch.setattr(gate, 'prepare_probe_map', lambda *args: None)
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
    out = case/'outdat/render'
    out.mkdir(parents=True)
    names = {'q_surface', 'q_surface.local', 'instantaneous_streamlines', 'velocity_slice',
             'temperature_slice', 'spanwise_slice'} | {
        f'{field}.x{100*n}_{100*(n+1)}' for field in ('velocity_slice', 'temperature_slice') for n in range(3)}
    receipt = dict(geometry_host_bytes=0, processing_backend='device',
        products={name: dict(image='published', empty=name.startswith('q_surface')) for name in names})
    for rank in range(4):
        (out/f'mesh_step00000010_rank{rank}.json').write_text(json.dumps(receipt))
    for name in names:
        image = Image.new('RGB', (8, 8), 'white')
        image.save(out/f'{name}.step00000010.jpeg')
        image.save(out/f'{name}.step00000010.eps')
    return case


def test_empty_q_and_pale_main_image_are_not_scientific_failures(tmp_path):
    record = gate.inspect_products(product_case(tmp_path), 10)
    assert record['publication_complete']
    assert any('empty geometry' in warning for warning in record['warnings'])


def test_missing_image_is_recorded_not_promoted_to_pass_or_abort(tmp_path):
    case = product_case(tmp_path)
    (case/'outdat/render/velocity_slice.step00000010.jpeg').unlink()
    record = gate.inspect_products(case, 10)
    assert not record['publication_complete']
    assert any('Missing image pair' in warning for warning in record['warnings'])


def test_host_geometry_still_fails(tmp_path):
    case = product_case(tmp_path)
    path = case/'outdat/render/mesh_step00000010_rank0.json'
    record = json.loads(path.read_text())
    record['geometry_host_bytes'] = 8
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='host geometry'):
        gate.inspect_products(case, 10)


def test_fullsize_configuration_keeps_optional_images_optional():
    args = SimpleNamespace(host_budget=32*gate.GIB, device_budget=32*gate.GIB,
                           reserve=8*gate.GIB, library='/catalyst')
    assert 'product_steps=10,10,10' in gate.render_configuration(args)
    assert 'product_steps=1000000000,1000000000,1000000000' in gate.render_configuration(args, False)
    assert 'statistics=f' in gate.render_configuration(args)
