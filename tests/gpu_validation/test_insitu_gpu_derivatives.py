"""Bounded real EGL rendering and restart identity for optional GPU diagnostics."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from run_output_restart_validation import run_case, compare_fields
from test_output_insitu_restart import ROOT, arguments, configuration, check_render

FINAL = 'outdat/new/checkpoints/step000000000004'


def candidate_config(backend):
    return configuration(statistics=False, interval=2).replace(
        'render=.true.,', f"render=.true., derivative_backend='{backend}',")


@pytest.fixture(scope='module',params=['x','y','z'])
def reference(request,tmp_path_factory):
    args = arguments(tmp_path_factory.mktemp(f'gpu_derivatives_render32_{request.param}'),request.param)
    args.statistics = False
    cases = {}
    for backend in ('cpu', 'gpu'):
        case, _ = run_case(args, ROOT, 'gpu', 2, backend, 4, grid='32,32,32', checkpoint_interval=1,
                           insitu_config=candidate_config(backend))
        check_render(case, 2, [2, 4], statistics=False)
        cases[backend] = case
    compare_fields(cases['cpu']/FINAL/'state.h5', cases['gpu']/FINAL/'state.h5')
    return args, cases


def test_gpu_derivatives_render_real_products(reference, record_property):
    args, cases = reference
    from test_insitu_products import compare_geometry
    fields=('u v w du_dx dv_dx dw_dx du_dy dv_dy dw_dy '
            'du_dz dv_dz dw_dz Q_rs divergence omega_x omega_y omega_z').split()
    differences = {}
    for picture in sorted((cases['cpu']/'outdat/render').glob('*.jpeg')):
        with Image.open(picture) as a, Image.open(cases['gpu']/'outdat/render'/picture.name) as b:
            aa, bb = np.asarray(a.convert('RGB')), np.asarray(b.convert('RGB'))
            assert aa.shape == bb.shape
            differences[picture.name] = int(np.max(abs(aa.astype(int)-bb.astype(int))))
        compare_geometry(picture.with_suffix('.pvtp'),
                         (cases['gpu']/'outdat/render'/picture.name).with_suffix('.pvtp'),fields)
    assert len(differences) == 8
    # Pixel differences are reported, not substituted for the numerical Q gate.
    record_property('cpu_gpu_jpeg_max_channel_difference', json.dumps(differences))
    record_property('topology_axis',args.axis)


def test_gpu_derivative_backend_switch_requires_override(reference, tmp_path):
    args, cases = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    source = cases['cpu']/'outdat/new/checkpoints/step000000000003'
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    run_case(args, ROOT, 'gpu', 2, 'reject_switch', 4, grid='32,32,32', checkpoint_interval=1,
             restore=source, insitu_config=candidate_config('gpu'),
             reject='native render configuration differs; select explicit override')
    resumed, _ = run_case(args, ROOT, 'gpu', 2, 'override_switch', 4, grid='32,32,32', checkpoint_interval=1,
                          restore=source, override=True, insitu_config=candidate_config('gpu'))
    compare_fields(cases['gpu']/FINAL/'state.h5', resumed/FINAL/'state.h5')
    check_render(resumed, 2, [4], statistics=False)
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


def test_gpu_derivatives_exact_render_continuation(reference, tmp_path):
    args, cases = reference
    args = SimpleNamespace(**vars(args)); args.output = tmp_path
    source = cases['gpu']/'outdat/new/checkpoints/step000000000003'
    resumed, _ = run_case(args, ROOT, 'gpu', 2, 'same_backend_resume', 4, grid='32,32,32', checkpoint_interval=1,
                          restore=source, insitu_config=candidate_config('gpu'))
    compare_fields(cases['gpu']/FINAL/'state.h5', resumed/FINAL/'state.h5')
    assert (cases['gpu']/FINAL/'insitu_control.bin').read_bytes() == (
        resumed/FINAL/'insitu_control.bin').read_bytes()
    check_render(resumed, 2, [4], statistics=False)
    for path in (resumed/'outdat/render').rglob('*'):
        if path.suffix in ('.jpeg', '.vtp'):
            counterpart = cases['gpu']/'outdat/render'/path.relative_to(resumed/'outdat/render')
            assert path.read_bytes() == counterpart.read_bytes(), path
