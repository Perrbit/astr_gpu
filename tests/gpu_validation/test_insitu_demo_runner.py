"""Failure receipts must never certify an incomplete demonstration."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2]/'scripts/insitu/run_tgv256_demo.py'
    spec = importlib.util.spec_from_file_location('demo_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('corrupt', [False, True])
def test_incomplete_receipt_is_failed(runner, monkeypatch, tmp_path, corrupt):
    case = tmp_path/'demo'
    monkeypatch.setattr(sys, 'argv', ['runner', '--output', str(case), '--steps', '100'])

    def fail(args):
        out = case/'outdat'
        out.mkdir(parents=True)
        (out/'demo_rank0.json').write_text('{' if corrupt else json.dumps({'frames': [[1, .0001]]}))
        raise RuntimeError('injected render failure')

    monkeypatch.setattr(runner, 'execute', fail)
    with pytest.raises(RuntimeError, match='injected render failure'):
        runner.main()
    receipt = json.loads((case/'summary.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['completed_frames_by_rank'] == {'0': None if corrupt else 1, '1': 0}
    assert bool(receipt['receipt_errors']) == corrupt
    assert not list(case.glob('*.mp4'))


def test_existing_directory_is_untouched(runner, monkeypatch, tmp_path):
    case = tmp_path/'existing'
    case.mkdir()
    (case/'summary.json').write_text('original')
    monkeypatch.setattr(sys, 'argv', ['runner', '--output', str(case), '--steps', '2'])
    with pytest.raises(FileExistsError):
        runner.main()
    assert (case/'summary.json').read_text() == 'original'


def test_video_requires_all_decoded_frames(runner):
    with pytest.raises(AssertionError, match='decoded frame count'):
        runner.validate_video_stream(iter([{'fps': 20., 'size': (2, 2), 'duration': .1},
                                          bytes(range(12))]), 2)


def test_video_checks_rate_and_endpoint_payloads(runner):
    first, last = bytes(range(12)), bytes(reversed(range(12)))
    frames = [{'fps': 20., 'size': (2, 2), 'duration': .1}, first, last]
    metadata, endpoints = runner.validate_video_stream(iter(frames), 2)
    assert metadata['frame_count'] == 2
    assert endpoints == (first, last)
    frames[0] = dict(frames[0], fps=10.)
    with pytest.raises(AssertionError, match='frame rate'):
        runner.validate_video_stream(iter(frames), 2)
