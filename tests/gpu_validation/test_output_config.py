"""Exercise the new Fortran parser without activating production output."""
import os
from pathlib import Path
import subprocess

import pytest

PROBE = os.environ.get('ASTR_OUTPUT_CONFIG_PROBE')
COLLECTIVE = os.environ.get('ASTR_OUTPUT_COLLECTIVE_PROBE')
MPIEXEC = os.environ.get('ASTR_OUTPUT_MPIEXEC')
pytestmark = pytest.mark.skipif(not PROBE, reason='Set ASTR_OUTPUT_CONFIG_PROBE from the root CMake build')

DISABLED = '&output\n/\n&checkpoint\n/\n&volume\n/\n&slices\n/\n'
VALID = '''&output
  host_budget_bytes=134217728, device_budget_bytes=67108864
/
&checkpoint
  enabled=t, mode='steps', interval_steps=100, keep=2
/
&volume
  enabled=t, mode='time', interval_time=0.01, vorticity=t
/
&slices
  enabled=t, interval_steps=5, i_indices=0,8,16, j_indices=12,24, k_indices=32
/
'''


def run(tmp_path, content):
    path = tmp_path / 'input.output'
    if content is not None:
        path.write_bytes(content.encode())
    return subprocess.run([str(Path(PROBE).resolve()), str(path)],
                          capture_output=True, text=True, timeout=10)


@pytest.mark.parametrize('content', [
    DISABLED, VALID, VALID.replace('\n', '\r\n'),
    '! comment\n' + VALID + '! trailing comment\n',
    VALID.replace('keep=2', 'keep=1'),
    VALID.replace('interval_steps=100', 'interval_steps=3000000000'),
    VALID.replace('interval_steps=100', "mode='time', interval_time=0.025"),
    VALID.replace('vorticity=t', 'velocity_gradient=t,qcriterion=t'),
    VALID.replace('host_budget_bytes=134217728',
                  "restore_directory='checkpoints/batch',restart_output='override',host_budget_bytes=134217728"),
])
def test_valid(tmp_path, content):
    result = run(tmp_path, content)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS: output configuration' in result.stdout
    assert 'checkpoint_final=T' in result.stdout
    if '3000000000' in content:
        assert 'checkpoint_interval_steps=3000000000' in result.stdout


@pytest.mark.parametrize('name,selected', [
    ('input.output.tgv.example', [0]*14),
    ('input.output.tgv.derived.example', list(range(1, 15))),
])
def test_documented_tgv_examples(tmp_path, name, selected):
    root = Path(__file__).resolve().parents[2]
    result = run(tmp_path, (root / 'scripts/output' / name).read_text())
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'slice_counts=1 1 1' in result.stdout
    for product in ('volume', 'slices'):
        line = next(line for line in result.stdout.splitlines() if line.startswith(product+'_derived_indices='))
        assert list(map(int, line.split('=', 1)[1].split())) == selected


@pytest.mark.parametrize('content', [
    None, '', '&old_restart\n/\n', VALID.replace('&output', '&unknown'),
    VALID.replace('&output', '&OUTPUT_WRONG'),
    DISABLED.replace('&volume\n/\n', ''),
    VALID + '&output\n/\n', VALID + 'unexpected',
    VALID.replace('keep=2', 'keep=0'), VALID.replace('keep=2', 'keep=3'),
    VALID.replace('&output\n', '&output\nformat_version=2,\n'),
    VALID.replace('interval_steps=100', 'interval_steps=0'),
    VALID.replace('interval_steps=100', 'interval_steps=-1'),
    VALID.replace('interval_steps=100', 'interval_steps=100,interval_time=0.1'),
    VALID.replace("mode='steps'", "mode='unknown'"),
    VALID.replace('interval_time=0.01', 'interval_time=0'),
    VALID.replace('interval_time=0.01', 'interval_time=-1'),
    VALID.replace('interval_time=0.01', 'interval_time=NaN'),
    VALID.replace('host_budget_bytes=134217728', 'host_budget_bytes=1'),
    VALID.replace('device_budget_bytes=67108864', 'device_budget_bytes=-1'),
    VALID.replace('device_budget_bytes=67108864', 'buffer_bytes=0'),
    VALID.replace('device_budget_bytes=67108864', "directory=''"),
    VALID.replace('device_budget_bytes=67108864', "directory='" + 'x' * 1100 + "'"),
    VALID.replace('device_budget_bytes=67108864', "restart_output='override'"),
    VALID.replace('vorticity=t', "fields='all'"),
    VALID.replace('i_indices=0,8,16, j_indices=12,24, k_indices=32', ''),
    VALID.replace('i_indices=0,8,16', 'i_indices=-2,8,16'),
    VALID.replace('interval_steps=100', 'misspelled_interval=100'),
    VALID.replace('i_indices=0,8,16', 'i_indices(257)=1'),
    VALID.replace('/\n&volume', '/ garbage\n&volume'),
])
def test_rejected_without_changing_active_options(tmp_path, content):
    result = run(tmp_path, content)
    assert result.returncode != 0
    assert 'REJECT unchanged:' in result.stdout, result.stdout + result.stderr


def test_slices_deduplicated(tmp_path):
    result = run(tmp_path, VALID.replace('i_indices=0,8,16', 'i_indices=0,8,0,8,16'))
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'slice_counts=3 2 1' in result.stdout


@pytest.mark.parametrize('flags', range(8))
@pytest.mark.parametrize('enabled', [True, False])
def test_derived_switches_have_canonical_layout(tmp_path, flags, enabled):
    controls = ','.join(f'{name}={"t" if flags & (1 << bit) else "f"}'
                        for bit, name in enumerate(('velocity_gradient', 'qcriterion', 'vorticity')))
    content = VALID.replace('vorticity=t', controls).replace('&slices\n', '&slices\n  ' + controls + ',\n')
    if not enabled:
        content = content.replace('enabled=t', 'enabled=f')
    result = run(tmp_path, content)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = []
    if enabled:
        if flags & 1:
            expected.extend(range(1, 10))
        if flags & 2:
            expected.extend((10, 11))
        if flags & 4:
            expected.extend((12, 13, 14))
    expected += [0]*(14-len(expected))
    for product in ('volume', 'slices'):
        line = next(line for line in result.stdout.splitlines() if line.startswith(product+'_derived_indices='))
        assert list(map(int, line.split('=', 1)[1].split())) == expected


@pytest.mark.parametrize('before,after', [('0,8,16', '0,8,17'), ('12,24', '12,25'), ('k_indices=32', 'k_indices=33')])
def test_global_grid_bounds(tmp_path, before, after):
    result = run(tmp_path, VALID.replace(before, after))
    assert result.returncode != 0
    assert 'REJECT grid:' in result.stdout, result.stdout + result.stderr


@pytest.mark.skipif(not COLLECTIVE or not MPIEXEC, reason='Set output collective probe and MPI launcher')
@pytest.mark.parametrize('content,mode,accepted', [
    (DISABLED, '', True), (VALID, '', True),
    (VALID.replace('interval_steps=100', 'interval_steps=3000000000'), '', True),
    (VALID.replace('vorticity=t', 'initial_frame=t,final_frame=t,velocity_gradient=t,qcriterion=t'), '', True),
    (VALID.replace('device_budget_bytes=67108864',
                   "device_budget_bytes=9000000000,restore_directory='ckpt/saved',restart_output='override'"), '', True),
    (None, '', False), (VALID.replace('keep=2', 'keep=3'), '', False),
    (VALID, 'mismatch', False),
])
def test_collective_configuration(tmp_path, content, mode, accepted):
    path = tmp_path / 'input.output'
    if content is not None:
        path.write_bytes(content.encode())
    result = subprocess.run([MPIEXEC, '--mca', 'coll_hcoll_enable', '0', '-np', '2',
                             str(Path(COLLECTIVE).resolve()), str(path), mode],
                            capture_output=True, text=True, timeout=30)
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == accepted, output
    marker = 'PASS collective rank ' if accepted else 'REJECT unchanged rank '
    assert output.count(marker) == 2, output
