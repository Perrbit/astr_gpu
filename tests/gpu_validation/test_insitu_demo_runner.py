"""Failure receipts must never certify an incomplete demonstration."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess

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
        for product in ('q_surface','instantaneous_streamlines'):
            out=case/'outdat/output/render_products'/product/'version000000000001/segment-test'
            out.mkdir(parents=True)
            (out/'frame.step000000000001.rank00000000.json').write_text('{' if corrupt else json.dumps(
                dict(step=1,product=dict(image='published'))))
        raise RuntimeError('injected render failure')

    monkeypatch.setattr(runner, 'execute', fail)
    with pytest.raises(RuntimeError, match='injected render failure'):
        runner.main()
    receipt = json.loads((case/'summary.json').read_text())
    assert receipt['status'] == 'failed'
    assert receipt['completed_frames_by_rank'] == {'0': None if corrupt else 1, '1': 0}
    assert bool(receipt['receipt_errors']) == corrupt
    assert not list(case.glob('*.mp4'))


def test_a_single_successful_product_is_not_a_complete_frame(runner,monkeypatch,tmp_path):
    case=tmp_path/'partial'
    monkeypatch.setattr(sys,'argv',['runner','--output',str(case),'--steps','2'])
    def fail(args):
        out=case/'outdat/output/render_products/q_surface/version000000000001/segment-test'
        out.mkdir(parents=True)
        (out/'frame.step000000000001.rank00000000.json').write_text(json.dumps(
            dict(step=1,product=dict(image='published'))))
        raise RuntimeError('second product failed')
    monkeypatch.setattr(runner,'execute',fail)
    with pytest.raises(RuntimeError): runner.main()
    assert json.loads((case/'summary.json').read_text())['completed_frames_by_rank']=={'0':0,'1':0}


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


@pytest.mark.skipif(not os.environ.get('ASTR_CATALOG_VIDEO_CASE'),reason='Select existing two-step native images')
def test_real_video_uses_complete_steps_not_the_initial_frame(runner,tmp_path):
    import imageio_ffmpeg
    source=Path(os.environ['ASTR_CATALOG_VIDEO_CASE'])
    for product in runner.benchmark.PRODUCTS:
        sequence=next((source/'outdat/output/render_products'/product).glob('version*/segment*'))
        assert (sequence/'frame.step000000000000.jpeg').is_file()
        movie=tmp_path/f'{product}.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-n','-framerate','20','-start_number','1',
            '-i',str(sequence/'frame.step%012d.jpeg'),'-frames:v','2','-c:v','libx264','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],check=True,timeout=30,
            capture_output=True)
        result=runner.inspect_video(imageio_ffmpeg,movie,sequence,product,2)
        assert result['frame_count']==2 and result['fps']==20. and result['duration']==.1
        (tmp_path/f'{product}.json').write_text(json.dumps(result,indent=2)+'\n')
