"""Native independent clocks and publication history; no Python scheduler copy."""
import os
from pathlib import Path
import struct
import subprocess

import pytest

from test_insitu_run_config import VALID

PROBE = os.environ.get('ASTR_INSITU_PRODUCT_SCHEDULE_PROBE')
CONFIG = VALID.replace('step_interval=2', """product_ids='q_surface.image','q_surface.geometry',
 'velocity_slice.image','mean_favre_streamlines.image',
 product_modes='steps','time','time','steps',product_steps=2,0,0,5,
 product_times=0,0.003,0.004,0""").replace('final_frame=t','final_frame=f')


def run(config, mode, state, path):
    source = path / ('input.' + mode)
    source.write_text(config)
    return subprocess.run([PROBE, str(source), mode, str(state)], capture_output=True, text=True, timeout=10)


@pytest.mark.skipif(not PROBE, reason='Set ASTR_INSITU_PRODUCT_SCHEDULE_PROBE')
def test_independent_clocks_exact_restart_and_missing_identity(tmp_path):
    state = tmp_path / 'state.bin'
    continuous = run(CONFIG, 'continuous', state, tmp_path)
    assert continuous.returncode == 0, continuous.stdout + continuous.stderr
    resumed = run(CONFIG, 'resumed', state, tmp_path)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    events = [line.strip() for line in continuous.stdout.splitlines()]
    assert events == [
        '2 q_surface.image', '3 q_surface.geometry', '4 q_surface.image', '4 velocity_slice.image',
        '5 mean_favre_streamlines.image', '6 q_surface.geometry', '6 q_surface.image',
        '8 q_surface.image', '8 velocity_slice.image', '9 q_surface.geometry',
        '10 mean_favre_streamlines.image', '10 q_surface.image',
        '12 q_surface.geometry', '12 q_surface.image', '12 velocity_slice.image']
    assert resumed.stdout.splitlines() == [line for line in continuous.stdout.splitlines() if int(line.split()[0]) > 5]
    assert state.with_suffix('.bin.continuous').read_bytes() == state.with_suffix('.bin.resumed').read_bytes()
    payload = state.read_bytes()
    assert payload[:8] == b'ASTRPF01' and len(payload) < 4096
    # At step five the latest q image attempt at four is missing (ENOSPC), but success remains at two.
    image = payload.index(b'q_surface.image')
    assert struct.unpack_from('<7q', payload, image + 96) == (4, 2, 4, 28, 1, 28, 1)
    final = state.with_suffix('.bin.continuous').read_bytes()
    image = final.index(b'q_surface.image')
    assert struct.unpack_from('<7q', final, image + 96) == (12, 12, 4, 0, 0, 28, 1)


@pytest.mark.skipif(not PROBE, reason='Set ASTR_INSITU_PRODUCT_SCHEDULE_PROBE')
def test_clock_identity_and_next_target_corruption_rejected(tmp_path):
    state = tmp_path / 'state.bin'
    assert run(CONFIG, 'continuous', state, tmp_path).returncode == 0
    assert run(CONFIG.replace('product_steps=2,0,0,5','product_steps=3,0,0,5'),
               'resumed', state, tmp_path).returncode != 0
    payload = bytearray(state.read_bytes())
    # The first canonical product's derived next target is serialized and checked independently.
    struct.pack_into('<q', payload, 32 + 176, 999)
    state.write_bytes(payload)
    result = run(CONFIG, 'resumed', state, tmp_path)
    assert result.returncode != 0 and 'FAIL restore' in result.stdout
