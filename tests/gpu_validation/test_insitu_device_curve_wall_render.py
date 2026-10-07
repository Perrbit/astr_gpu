"""Bounded y-wavy CURVE wall images, no field or geometry reference downloads.

Geometry measures come from an independent checkpoint coordinate integration;
same-state fields are covered separately by the CURVE wall fields gate.
"""
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields
from run_insitu_gpu_derivatives import check_resources
from test_insitu_device_channel_render import configuration
from test_insitu_device_curve_wall_fields import current_arguments, ROOT, FINAL
from test_insitu_geometry import integrate_numpy

PRODUCTS = {'wall_pressure': [80., 120.], 'wall_shear_x': [-.2, .2],
            'wall_heat_into_gas': [0., 15000.]}


def measures(case):
    with h5py.File(case / 'outdat/new/resources/geometry.h5') as geometry:
        xyz = np.stack([geometry[f'q{c:04d}'][:].transpose(2, 1, 0) for c in (1, 2, 3)])
    _, area = integrate_numpy(xyz)
    facets = 0.
    for wall in (0, 32):
        for i, k in np.ndindex((32, 32)):
            a, b = xyz[:, i, wall, k], xyz[:, i + 1, wall, k]
            c, d = xyz[:, i, wall, k + 1], xyz[:, i + 1, wall, k + 1]
            facets += .5 * (np.linalg.norm(np.cross(c - a, d - a)) +
                            np.linalg.norm(np.cross(d - a, b - a)))
    return facets, float(area.sum())


def validate(case, ranks, steps, pipeline, mode):
    log = (case / 'run.log').read_text()
    assert 'ASTR_INSITU_DEVICE_WALL_CHECK' not in log
    assert 'ASTR_INSITU_WALL rank=' not in log
    frames = re.findall(r'ASTR_INSITU_DEVICE_WALL_FRAME rank=(\d+) step=(\d+) transport=(\S+) '
                        r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+) '
                        r'field_download_bytes=0 geometry_host_bytes=0', log)
    assert len(frames) == ranks * len(steps), log
    assert {row[2] for row in frames} == {mode}
    if mode == 'device-aware':
        assert all(int(row[3]) == int(row[4]) == 0 for row in frames)
    facets, quadrature = measures(case)
    geometry = re.findall(r'ASTR_INSITU_DEVICE_WALL_GEOMETRY rank=(\d+) step=(\d+) product=(\S+) '
                          r'points=(\d+) triangles=(\d+) local_area=(\S+) global_area=(\S+) '
                          r'invalid_triangles=0 geometry_host_bytes=0', log)
    areas = re.findall(r'ASTR_INSITU_DEVICE_WALL_MEASURE rank=(\d+) step=(\d+) product=(\S+) curve=1 '
                       r'local_quad_area=(\S+) global_quad_area=(\S+)', log)
    assert len(geometry) == len(areas) == ranks * len(steps) * 3
    scalars = re.findall(r'ASTR_INSITU_DEVICE_WALL_SCALAR rank=(\d+) step=(\d+) product=(\S+) '
                        r'minimum=(\S+) maximum=(\S+) display_minimum=(\S+) display_maximum=(\S+) '
                        r'control_values=2 field_download_bytes=0', log)
    assert len(scalars) == ranks * len(steps) * 3
    for rank, step, name, minimum, maximum, lower, upper in scalars:
        assert name in PRODUCTS and int(step) in steps and int(rank) < ranks
        assert [float(lower), float(upper)] == PRODUCTS[name]
        assert float(lower) <= float(minimum) <= float(maximum) <= float(upper), (
            'Frozen fixture color range clips actual GPU wall values', name, step, minimum, maximum)
    assert abs(facets - quadrature) > 1e-6, 'The two distinct surface measures were conflated'
    for step in steps:
        for name in PRODUCTS:
            rows = [row for row in geometry if int(row[1]) == step and row[2] == name]
            assert len(rows) == ranks and sum(int(row[4]) for row in rows) == 4096
            assert all(abs(float(row[6]) - facets) <= 2e-10 for row in rows)
            assert abs(sum(float(row[5]) for row in rows) - facets) <= 2e-10
            rows = [row for row in areas if int(row[1]) == step and row[2] == name]
            assert len(rows) == ranks
            assert all(abs(float(row[4]) - quadrature) <= 2e-10 for row in rows)
            assert abs(sum(float(row[3]) for row in rows) - quadrature) <= 2e-10
            path = case / f'outdat/render/{name}.step{step:08d}.jpeg'
            assert path.with_suffix('.eps').read_bytes().startswith(b'%!PS-Adobe')
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
                assert pixels.shape == (600, 800, 3)
                assert np.count_nonzero(np.any(pixels < 220, axis=2)) > 10000
                border = np.concatenate((pixels[:10].reshape(-1, 3), pixels[-10:].reshape(-1, 3),
                                         pixels[:, :10].reshape(-1, 3), pixels[:, -10:].reshape(-1, 3)))
                assert np.all(border >= 245), 'The physical wall surface is clipped'
        for rank in range(ranks):
            receipt = json.loads((case / f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert receipt['processing_backend'] == 'device' and receipt['rendering_pipeline'] == pipeline
            assert receipt['geometry_host_bytes'] == 0 and set(receipt['products']) == set(PRODUCTS)
            for name, colors in PRODUCTS.items():
                item = receipt['products'][name]
                assert item['color_field'] == name and item['color_range'] == colors
                assert item['local_cells'] > 0 and item['local_points'] > 0
    assert not list((case / 'outdat/render').glob('*.vtp'))
    assert not list((case / 'outdat/render').glob('*.pvtp'))
    check_resources(case, ranks, steps=steps, capture=False)


@pytest.mark.parametrize('ranks,axis', [(1, 'x'), (2, 'x'), (2, 'y'), (2, 'z')],
                         ids=['single', 'x', 'y', 'z'])
@pytest.mark.parametrize('mode', ['pinned', 'device-aware'])
@pytest.mark.parametrize('pipeline', ['standard-device', 'direct-device'])
def test_real_curve_device_wall_render(tmp_path, ranks, axis, mode, pipeline, record_property):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 300
    config = configuration(pipeline, mode)
    kwargs = dict(grid='32,32,32', tgv_mapping='y-wavy', checkpoint_interval=1)
    actual, size = run_case(args, ROOT, 'gpu', ranks, 'render', 4,
                            insitu_config=config, postprocess_transport=mode, **kwargs)
    validate(actual, ranks, (2, 4), pipeline, mode)
    record_property('directory_bytes', size)
    off, _ = run_case(args, ROOT, 'gpu', ranks, 'off', 4, **kwargs)
    compare_fields(actual / FINAL / 'state.h5', off / FINAL / 'state.h5')
    source = actual / 'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    resumed, _ = run_case(args, ROOT, 'gpu', ranks, 'restart', 4, restore=source,
                         insitu_config=config, postprocess_transport=mode, **kwargs)
    validate(resumed, ranks, (4,), pipeline, mode)
    compare_fields(actual / FINAL / 'state.h5', resumed / FINAL / 'state.h5')
    for name in ('control.bin', 'insitu_control.bin'):
        assert (actual / FINAL / name).read_bytes() == (resumed / FINAL / name).read_bytes()
    for name in PRODUCTS:
        for suffix in ('jpeg', 'eps'):
            filename = f'{name}.step00000004.{suffix}'
            assert (actual / 'outdat/render' / filename).read_bytes() == (resumed / 'outdat/render' / filename).read_bytes()
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


@pytest.mark.parametrize('pipeline,mode,axis', [
    ('standard-device', 'device-aware', 'z'), ('direct-device', 'pinned', 'x')])
def test_curve_device_wall_render_memcheck(tmp_path, pipeline, mode, axis):
    args = current_arguments(tmp_path, axis=axis, samples=False)
    args.runtime_timeout_seconds = 300
    case, _ = run_case(args, ROOT, 'gpu', 2, 'memcheck', 2,
                       grid='32,32,32', tgv_mapping='y-wavy', checkpoint_interval=1,
                       insitu_config=configuration(pipeline, mode), postprocess_transport=mode, memcheck=True)
    validate(case, 2, (2,), pipeline, mode)
    reports = list(case.glob('memcheck.*.log'))
    assert len(reports) == 2
    for report in reports:
        assert 'ERROR SUMMARY: 0 errors' in report.read_text(), report


def test_curve_wall_partition_image_seams(record_property):
    root = os.environ.get('ASTR_INSITU_CURVE_WALL_IMAGE_ROOT')
    if not root:
        pytest.skip('Select the immutable completed sixteen-case CURVE wall matrix')
    from scipy.ndimage import distance_transform_edt
    cases = sorted(p for p in Path(root).glob('test_real_curve_device_wall_re*/gpu_np*_render')
                   if not p.parent.is_symlink())
    assert len(cases) == 16
    maxima, entry_difference = {}, 0
    for name in PRODUCTS:
        worst = 0.
        for step in (2, 4):
            reference, pairs = None, {}
            for case in cases:
                log = (case / 'run.log').read_text()
                topology = re.search(r'forced mpi topology=\s*(\d+)\s+(\d+)\s+(\d+)', log).groups()
                transport = re.search(r'ASTR_INSITU_DEVICE_WALL_FRAME .*?transport=(\S+)', log).group(1)
                with Image.open(case / f'outdat/render/{name}.step{step:08d}.jpeg') as image:
                    pixels = np.asarray(image.convert('RGB')).astype(int)[50:550, 50:620]
                mask = pixels.max(2) - pixels.min(2) > 40
                assert mask.sum() > 10000, (case, name, step)
                if reference is None:
                    reference = mask
                error = max(float(distance_transform_edt(~reference)[mask].max()),
                            float(distance_transform_edt(~mask)[reference].max()))
                assert error <= 1., (case, name, step, error)
                worst = max(worst, error)
                key = topology, transport
                if key in pairs:
                    entry_difference = max(entry_difference, int(abs(pixels - pairs.pop(key)).max()))
                else:
                    pairs[key] = pixels
            assert not pairs
        maxima[name] = worst
    record_property('partition_seam_max_pixels', json.dumps(maxima, sort_keys=True))
    record_property('cross_entry_pixel_maxabs', entry_difference)
