"""Keep diagnostic downloads out of the separately captured M12 render path."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_m12_local_compare import run
from run_m12_device_observation import add_edges


@pytest.mark.parametrize('mode,topology,profiler_allowance,expected_host_budget', (
    ('images', '1,1,1', False, 4),
    ('geometry', '1,1,1', False, 4),
    ('trace', '1,1,1', False, 4),
    ('trace', '2,1,1', True, 4),
    ('trace', '4,1,1', True, 8),
    ('images', '4,1,1', True, 4),
))
def test_observation_environment_is_separate(tmp_path, mode, topology, profiler_allowance,
                                           expected_host_budget):
    args = SimpleNamespace(source=tmp_path, output=tmp_path/'runs', executable=Path('/bin/true'),
                           mpiexec=Path('/bin/true'), nsys=Path('/bin/true'),
                           memcheck=False, diagnostics=False,
                           np4_profiler_host_budget_8gib=profiler_allowance)
    captured = {}

    def prepare(_, case, __):
        for name in ('datin', 'outdat', 'diagnostics'):
            (case/name).mkdir(parents=True, exist_ok=True)

    def monitor(command, case, env, log, report, **kwargs):
        captured.update(env=env, command=command, monitor_options=kwargs)
        log.write(b'The job is done!\n')
        return dict(host_rss_peak_bytes=0, devices={})

    gate = SimpleNamespace(maximum=.05, check=lambda **kwargs: None)
    inherited = dict(ASTR_VALIDATION_UNEXPECTED='1', ASTR_M12_RENDER_ISOLATION='1',
                     ASTR_INSITU_TEST_CURVE_TRACE_PREFIX='old', ASTR_INSITU_RESIDENT_AUDIT='1',
                     ASTR_VTK_PIXEL_AUDIT='1')
    with patch.dict('os.environ', inherited), patch('run_m12_local_compare.prepare', prepare), \
            patch('run_m12_local_compare.run_monitored', monitor), \
            patch('run_m12_local_compare.CflGate', return_value=gate):
        case = run(args, 'gpu', topology, '&insitu_run enabled=t /', observation=mode)
    env = captured['env']
    assert not any(k.startswith(('ASTR_VALIDATION_', 'ASTR_M12_')) for k in env)
    assert 'ASTR_GEOMETRY_DUMP' not in env and 'ASTR_INSITU_RESIDENT_AUDIT' not in env
    assert env['ASTR_INSITU_TEST_ORACLE_IO'] == '0'
    assert env['ASTR_GPU_SYNC_MODE'] == 'explicit'
    assert env['ASTR_WALL_BLOWING_MODE'] == 'legacy_random'
    assert captured['monitor_options']['host_extra_budget_bytes'] == expected_host_budget*1024**3
    assert json.loads((case/'gate.json').read_text())['rho_min'] is None
    if mode == 'trace':
        assert not any(k.endswith('_PREFIX') and k.startswith('ASTR_INSITU_TEST_') for k in env)
        assert env['ASTR_VTK_PIXEL_AUDIT'] == '1'
        assert '--trace=cuda,nvtx,mpi' in captured['command']
    elif mode == 'geometry':
        assert env['ASTR_INSITU_TEST_CURVE_TRACE_PREFIX'] == 'diagnostics/trace'
        assert env['ASTR_INSITU_TEST_PLANE_PREFIX'] == 'diagnostics/plane'
        assert 'ASTR_VTK_PIXEL_AUDIT' not in env
    else:
        assert 'ASTR_VTK_PIXEL_AUDIT' not in env
        assert not any(k.endswith('_PREFIX') and k.startswith('ASTR_INSITU_TEST_') for k in env)


def test_triangle_edge_winding_and_exterior_count():
    edges = {}
    add_edges(edges, (0, 1, 2))
    add_edges(edges, (2, 1, 3))
    assert edges[(1, 2)] == (2, 0)
    assert sum(count == 1 for count, _ in edges.values()) == 4


def test_profiler_budget_option_cannot_relax_plain_run(tmp_path):
    output = tmp_path/'not-created'
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('run_m12_device_observation.py')),
        '--source', str(tmp_path), '--output', str(output), '--executable', '/bin/true',
        '--mpiexec', '/bin/true', '--library', str(tmp_path), '--mode', 'timing',
        '--np4-profiler-host-budget-8gib'], capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'only allowed with --mode trace' in result.stderr
    assert not output.exists()
