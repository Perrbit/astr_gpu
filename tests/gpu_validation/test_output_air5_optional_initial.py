"""Profile-only startup, exact continuation, and development-window retention."""
import pytest

import run_output_air5_restart_validation as restart
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_profile_only_exact_restart(tmp_path, monkeypatch, backend):
    original = restart.prepare

    def profile_only(*args, **kwargs):
        tau = original(*args, **kwargs)
        (args[0]/'datin/air5_hbl_initial_field.dat').unlink()
        return tau

    monkeypatch.setattr(restart, 'prepare', profile_only)
    args = arguments(tmp_path)
    args.mean_statistics = False
    continuous, _ = restart.launch(args, backend, 2, 'continuous', 4, interval=2)
    first, _ = restart.launch(args, backend, 2, 'first', 2, interval=2)
    seed = first/'outdat/new/checkpoints/step000000000002'
    resumed, _ = restart.launch(args, backend, 2, 'resumed', 4, interval=2, restore=seed)
    final = 'outdat/new/checkpoints/step000000000004'
    compare_fields(continuous/final/'state.h5', resumed/final/'state.h5')
    for case in (continuous, first, resumed):
        assert not (case/'outdat/new/resources/air5_hbl_initial_field.dat').exists()
        for name in ('air5_hbl_domain.dat', 'air5_hbl_profile.dat'):
            assert (case/'outdat/new/resources'/name).is_file()
    assert (seed/'air5_config.bin').read_bytes()[:8] == b'ASTRA503'


def test_initial_diagnostic_survives_retention(tmp_path, monkeypatch):
    import run_air5_local_development as window
    original = window.prepare_window

    def frequent_checkpoint(*args, **kwargs):
        runtime, record = original(*args, **kwargs)
        config = args[0]/'datin/input.output'
        config.write_text(config.read_text().replace('interval_steps=4', 'interval_steps=2'))
        return runtime, record

    monkeypatch.setattr(window, 'prepare_window', frequent_checkpoint)
    case = tmp_path/'case'
    row = window.run_case(case, arguments(tmp_path).executable,
                          'baseline', 4, True, tmp_path)
    assert row['status'] == 'bounded-development-run-passed'
    checkpoints = case/'outdat/new/checkpoints'
    assert sorted(p.name for p in checkpoints.glob('step*')) == [
        'step000000000002', 'step000000000004']
    initial = case/'initial_state.h5'
    assert initial.is_file() and initial.stat().st_nlink == 1


def test_development_heat_components_cpu_gpu(tmp_path, monkeypatch):
    import numpy as np
    import run_air5_local_development as window
    from run_air5_sbli_preflight import set_value
    from test_air5_flow_monitor import monitor_difference
    original = window.prepare_window
    cases = {}
    for backend in ('cpu','gpu'):
        def prepared(*args, **kwargs):
            runtime, record = original(*args, **kwargs)
            set_value(args[0]/'datin/input.air5_c4',
                'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
                f'f,t,t,f,f,f,f,t,{"t" if backend=="gpu" else "f"}')
            return runtime, record
        monkeypatch.setattr(window,'prepare_window',prepared)
        case = tmp_path/backend
        window.run_case(case,arguments(tmp_path).executable,'baseline',4,True,tmp_path)
        cases[backend] = case
    for name in ('air5_profiles.dat','air5_wall.dat'):
        cpu = np.loadtxt(cases['cpu']/'monitor'/name,ndmin=2)
        gpu = np.loadtxt(cases['gpu']/'monitor'/name,ndmin=2)
        if name == 'air5_wall.dat':
            assert cpu.shape[1] == gpu.shape[1] == 11
        assert monitor_difference(cpu,gpu,name) <= 2e-10
