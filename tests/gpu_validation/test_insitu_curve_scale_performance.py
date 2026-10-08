"""Timing report controls do not change solver or integration settings."""
import pytest

from run_insitu_curve_scale_performance import (BACKENDS, METRICS, backend_key,
                                              group_bytes, production_configuration, round_order, summarize)


def records():
    return [dict(backend=backend_key(*backend), round=round_id,
                 timing={metric: dict(seconds=10.+index+round_id*.1) for metric in METRICS},
                 step4_output_seconds=2.+index+round_id*.1)
            for index, backend in enumerate(BACKENDS) for round_id in range(-1, 5)]


def test_order_visits_each_backend_once():
    for round_id in range(-1, 5):
        assert len(round_order(round_id)) == 4 and set(round_order(round_id)) == set(BACKENDS)
    for position in range(4):
        assert {round_order(round_id)[position] for round_id in range(4)} == set(BACKENDS)
    pairs = [(first, second) for round_id in range(4)
             for first, second in zip(round_order(round_id), round_order(round_id)[1:])]
    assert len(set(pairs)) == 12


def test_summary_excludes_warmup_and_selects_complete_window():
    rows = records()
    for row in rows:
        if row['round'] == -1:
            row['timing']['completed_window']['seconds'] = 1000.
    result = summarize(rows, 5)
    assert result['selected'] == backend_key(*BACKENDS[0])
    assert result['backends'][result['selected']]['completed_window']['median'] == 10.2
    assert result['nested_timings_are_not_additive']


def test_summary_rejects_missing_round():
    with pytest.raises(ValueError, match='Missing or repeated'):
        summarize(records()[:-1], 5)


def test_every_step_production_preserves_other_configuration():
    from test_insitu_curve_demo import scale_configuration
    for pipeline, mode in BACKENDS:
        assert production_configuration(pipeline, mode).replace('step_interval=1', 'step_interval=2') == \
            scale_configuration(pipeline, mode)


def test_disk_budget_counts_receipts_once(tmp_path):
    case = tmp_path/'curve_case'
    case.mkdir()
    (case/'data.bin').write_bytes(b'abcd')
    (tmp_path/'curve_receipt.xml').write_bytes(b'abc')
    (tmp_path/'curve_current').symlink_to(case, target_is_directory=True)
    (tmp_path/'unrelated.bin').write_bytes(b'0123456789')
    assert group_bytes(tmp_path) == 7


@pytest.mark.parametrize('temperature,mapping,grid,dt,cfl', [
    (0., 'y-wavy', '64,64,64', 2e-5, .5),
    (-1., 'y-wavy', '64,64,64', 2e-5, .5),
    (float('nan'), 'y-wavy', '64,64,64', 2e-5, .5),
    (float('inf'), 'y-wavy', '64,64,64', 2e-5, .5),
    (True, 'y-wavy', '64,64,64', 2e-5, .5),
    (1., 'periodic', '64,64,64', 2e-5, .5),
    (1., 'y-wavy', '32,32,32', 2e-5, .5),
    (1., 'y-wavy', '64,64,64', None, .5),
    (1., 'y-wavy', '64,64,64', 2e-5, float('nan')),
])
def test_wall_override_rejects_before_preparation(tmp_path, temperature, mapping, grid, dt, cfl):
    from run_output_restart_validation import run_case
    from test_insitu_device_curve_wall_fields import current_arguments
    from test_insitu_curve_derivatives import ROOT
    args = current_arguments(tmp_path, samples=False)
    args.scale_timestep, args.maximum_cfl = dt, cfl
    with pytest.raises(ValueError, match='Wall temperature override'):
        run_case(args, ROOT, 'gpu', 2, 'invalid', 2, grid=grid, tgv_mapping=mapping,
                 curve_wall_temperature=temperature)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('temperature', [None, 1.])
def test_wall_override_preserves_default_and_accepts_approved_fixture(tmp_path, monkeypatch, temperature):
    from run_output_restart_validation import run_case
    from test_insitu_device_curve_wall_fields import current_arguments
    from test_insitu_curve_derivatives import ROOT
    args = current_arguments(tmp_path, samples=False)
    args.scale_timestep, args.maximum_cfl = 2e-5, .5

    class Prepared(Exception):
        pass

    def boundary(path, value):
        wall = '273.15d0' if temperature is None else '1.00000000000000000e+00'
        assert path == tmp_path/'gpu_np2_valid/datin/input.tgv'
        assert value == f'1;1;41,{wall};41,{wall};1;1'
        raise Prepared

    monkeypatch.setattr('run_output_restart_validation.subprocess.run', lambda *args, **kwargs: None)
    for name in ('set_gridfile', 'set_runtime_flags', 'set_homogeneous'):
        monkeypatch.setattr('prepare_tgv_case.'+name, lambda *args: None)
    monkeypatch.setattr('prepare_tgv_case.set_bctype', boundary)
    with pytest.raises(Prepared):
        run_case(args, ROOT, 'gpu', 2, 'valid', 2, grid='64,64,64', tgv_mapping='y-wavy',
                 curve_wall_temperature=temperature)
