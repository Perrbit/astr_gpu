"""Small host-side checks of the approved HD test configuration and probe map."""
import ast
import itertools
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from run_m12_hd_coverage import expected_images, finish_comparison, insitu_configuration, prepare_probe_map, visible_field_region


@pytest.mark.parametrize('name,rgb,expected', [
    ('spanwise_slice', [240, 248, 248], True),
    ('spanwise_slice', [255, 255, 255], False),
    ('spanwise_slice_weak', [240, 248, 248], False),
    ('spanwise_slice_weak', [150, 30, 30], True),
])
def test_fixed_range_weak_main_can_be_pale_but_not_blank(name, rgb, expected):
    pixels = np.broadcast_to(np.array(rgb, dtype=np.uint8), (10, 10, 3))
    assert visible_field_region(name, pixels) == expected


@pytest.mark.parametrize('filename', ['device_render_pipeline.py', 'tgv_pipeline.py'])
@pytest.mark.parametrize('name,profile,bounds,expected', [
    ('velocity_slice', 'all', None, True),
    ('spanwise_slice_weak', 'm12_diagnostics', None, True),
    ('velocity_slice', 'physical_plane', [0., 1.]*3, True),
    ('pressure', 'air5_walls', [0., 1., 0., 0., 0., 1.], True),
    ('mean_wall_shear_x', 'curve_walls', [0., 1., -.1, .1, 0., 1.], False),
    ('q_surface', 'm12_diagnostics', [0., 1.]*3, False),
    ('instantaneous_streamlines', 'all', [0., 1.]*3, False),
])
def test_flat_color_policy_preserves_three_dimensional_shading(filename, name, profile, bounds, expected):
    pipeline = Path(__file__).resolve().parents[2]/'scripts/insitu'/filename
    tree = ast.parse(pipeline.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'is_two_dimensional_product')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(pipeline), 'exec'), namespace)
    assert namespace['is_two_dimensional_product'](name, profile, bounds) == expected


@pytest.mark.skipif(not os.environ.get('ASTR_OUTPUT_INSITU_EXE') or
                    not os.environ.get('ASTR_OUTPUT_INSITU_BACKEND'),
                    reason='Select the Catalyst executable and implementation explicitly')
@pytest.mark.parametrize('pipeline', ['compatible', 'standard-device', 'direct-device'])
def test_slice_flat_color_actual_catalyst(tmp_path, pipeline):
    from run_output_restart_validation import run_case
    from test_output_insitu_restart import ROOT, arguments
    from test_insitu_device_products import device_configuration, resident_configuration
    args = arguments(tmp_path)
    args.statistics = False
    args.directory_budget_bytes = 256*1024**2
    args.runtime_timeout_seconds = 300
    settings = device_configuration('pinned', statistics=False, profile='velocity_slice') if pipeline == 'compatible' else \
        resident_configuration(pipeline, statistics=False, profile='velocity_slice')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'flat_slice', 2, enabled=False,
        grid='32,32,32', insitu_config=settings, no_field_io=True, postprocess_transport='pinned')
    for step in (1, 2):
        for rank in (0, 1):
            record = json.loads((case/'outdat/render'/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            item = record['products']['velocity_slice']
            status = item['image']
            assert (status['status'] if isinstance(status, dict) else status) == 'published'
            if pipeline == 'compatible':
                assert item['flat_color_lighting']
            else:
                assert item['lighting_enabled'] is False
                assert record['geometry_host_bytes'] == 0


@pytest.mark.parametrize('bounds,streamlines', [
    ([20., 90., 0., 5., 20., 70.], False),
    ([20., 90., 0., 8., 39., 51.], True),
])
def test_oblique_camera_fits_physical_window_and_annotation_margin(bounds, streamlines):
    # Load the pure camera calculation without initializing Catalyst/ParaView.
    pipeline = Path(__file__).resolve().parents[2]/'scripts/insitu/device_render_pipeline.py'
    tree = ast.parse(pipeline.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'm12_oblique_camera')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(pipeline), 'exec'), namespace)
    size = [3840, 2160]
    camera = namespace['m12_oblique_camera'](bounds, size, streamlines=streamlines)
    eye, focal = np.asarray(camera['position']), np.asarray(camera['focal_point'])
    direction = eye-focal
    direction /= np.linalg.norm(direction)
    assert direction[0] < 0. and direction[2] > 0.
    assert np.degrees(np.arcsin(direction[1])) == pytest.approx(30. if streamlines else 18.)
    assert np.degrees(np.arctan2(direction[2], -direction[0])) == pytest.approx(40.)
    np.testing.assert_array_equal(camera['view_up'], [0., 1., 0.])
    right = np.cross([0., 1., 0.], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    corners = np.array(list(itertools.product(*[bounds[2*d:2*d+2] for d in range(3)])))
    assert np.max(np.abs((corners-focal)@up)) < .95*camera['parallel_scale']
    assert np.max(np.abs((corners-focal)@right)) < .95*camera['parallel_scale']*size[0]/size[1]
    if streamlines:
        assert bounds[3] > 5.
        setup = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'setup_m12_view')
        branch = next(node for node in ast.walk(setup) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "base == 'instantaneous_streamlines'"
                      and any(isinstance(statement, ast.Assign) and
                              any(isinstance(target, ast.Name) and target.id == 'bounds'
                                  for target in statement.targets) for statement in node.body))
        assignment = next(statement for statement in branch.body if isinstance(statement, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'bounds'
                                  for target in statement.targets))
        assert ast.literal_eval(assignment.value) == bounds
        # Check projected seed separation on an undisturbed parallel-flow fixture.
        z = [(k+.5)*90./64 for k in range(64) if bounds[4] < (k+.5)*90./64 < bounds[5]]
        assert len(z) == 8
        seeds = np.array([[54., y, zk] for y in (.5, 1., 2., 5.) for zk in z])
        assert len(seeds) == 32
        tangent = np.array([right[0], up[0]])
        normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
        projected = np.column_stack([(seeds-focal)@right, (seeds-focal)@up])
        offsets = np.sort(projected@normal)
        gap_pixels = np.diff(offsets)*size[1]/(2.*camera['parallel_scale'])
        assert gap_pixels.min() > 2.*3.  # More than two three-pixel line widths.


def test_hd_schedule_has_no_forced_initial_or_final_frames():
    config = insitu_configuration()
    assert 'initial_frame=f, final_frame=f' in config
    assert "products='m12_diagnostics',streamline_seeds='bl-layered256'" in config
    assert "rendering_pipeline='standard-device'" in config
    assert 'product_steps=250,100,250' in config
    assert 'product_steps=1000000000,1000000000,1000000000' in insitu_configuration(False)


def test_first_output_window_covers_all_approved_images():
    assert len(expected_images(100)) == 9
    assert expected_images(100) == expected_images(200)
    assert set(expected_images(250)) == {'q_surface', 'q_surface.local', 'instantaneous_streamlines'}
    assert sum(len(expected_images(step)) for step in (100, 200, 250)) == 21
    assert expected_images(100, weak=True) == expected_images(100) | {'spanwise_slice_weak': (5120, 1536)}
    assert expected_images(250, weak=True) == expected_images(250)


def test_probe_map_uses_nearest_physical_node_without_interpolation(tmp_path):
    (tmp_path/'datin').mkdir()
    x = np.linspace(0., 1100., 65)
    y = np.linspace(0., 150., 33)
    with h5py.File(tmp_path/'datin/grid.2d', 'w') as grid:
        grid['x'] = np.tile(x, (33, 1))
        grid['y'] = np.tile(y[:, None], (1, 65))
    probes = prepare_probe_map(tmp_path, (64, 32, 24))
    assert len(probes) == 72
    assert len({tuple(p['target']) for p in probes}) == 72
    for p in probes:
        assert p['indices'] == [int(np.abs(axis-target).argmin()) for axis, target in
                                zip((x, y, np.linspace(0., 90., 25)), p['target'])]
        assert p['indices'][2] < 24
    assert (tmp_path/'datin/insitu_probes.dat').read_text().splitlines()[0] == 'ASTR_M12_PROBES_V1 72'


@pytest.mark.parametrize('status,sha', [('running', 'same'), ('failed', 'same'), ('completed', 'different')])
def test_finalization_rejects_incomplete_or_unmatched_runs(tmp_path, status, sha):
    record = dict(status='completed', shape=[476, 130, 96], steps=250, dt=.02,
                  topology='2,1,1', executable_sha256='same')
    for label, receipt in (('images_off', record),
                           ('images_on', dict(record, status=status, executable_sha256=sha))):
        directory = tmp_path/label
        directory.mkdir()
        (directory/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='Only matched, completed coverage runs'):
        finish_comparison(tmp_path)
    assert not (tmp_path/'comparison.json').exists()
