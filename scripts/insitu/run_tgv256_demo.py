"""Prepare/run the approved two-GPU render-only demonstration and encode movies."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
MPI = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
PV = Path('/home/dell/workspace/astr_dependencies/install/paraview-6.1.1-hpcx-gcc13')
VIDEO_MODULES = '/home/dell/workspace/astr_dependencies/python-video'


def validate_video_stream(reader, steps):
    metadata = next(reader)
    assert metadata['fps'] == 20., 'Unexpected video frame rate'
    assert abs(metadata['duration']-steps/20.) <= .005, 'Unexpected video duration'
    first = last = None
    count = 0
    width, height = metadata['size']
    for frame in reader:
        assert len(frame) == width*height*3, 'Invalid decoded RGB payload'
        if first is None:
            first = frame
        last = frame
        count += 1
    assert count == steps, f'Unexpected decoded frame count: {count}, expected {steps}'
    return dict(frame_count=count, fps=metadata['fps'], duration=metadata['duration'],
                size=list(metadata['size'])), (first, last)


def inspect_video(imageio_ffmpeg, path, out, product, steps):
    import numpy as np
    from PIL import Image
    metadata, endpoints = validate_video_stream(imageio_ffmpeg.read_frames(str(path)), steps)
    differences = []
    for step, payload in zip((1, steps), endpoints):
        original = np.asarray(Image.open(out/f'{product}.step{step:08d}.jpeg').convert('RGB'))
        decoded = np.frombuffer(payload, dtype=np.uint8).reshape(metadata['size'][1], metadata['size'][0], 3)
        assert original.shape == decoded.shape, 'Video changed image dimensions'
        assert decoded.min() != decoded.max(), 'Blank video endpoint'
        difference = np.abs(original.astype(np.int16)-decoded.astype(np.int16))
        differences.append(dict(step=step, mean_absolute_rgb_difference=float(difference.mean()),
                                maximum_rgb_difference=int(difference.max())))
    metadata['endpoints'] = differences
    return metadata


def execute(args):
    case = args.output.resolve()
    if case.exists():
        raise FileExistsError(case)
    subprocess.run([sys.executable, str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
                    '--src-case', str(ROOT/'examples/Taylor_Green_Vortex'), '--dst-case', str(case),
                    '--use-gpu', 't', '--grid', '256,256,256', '--maxstep', str(args.steps-1),
                    '--feqchkpt', '1000', '--deltat', '1.d-4', '--lfilter', 't',
                    '--diffterm', 't', '--scheme', '643e'], check=True)
    (case/'datin/input.output').write_text(
        "&output\n directory='outdat/output', buffer_bytes=4096,\n"
        'host_budget_bytes=67108864, device_budget_bytes=67108864\n/\n'
        '&checkpoint\n enabled=f\n/\n&volume\n enabled=f\n/\n&slices\n enabled=f\n/\n')
    (case/'insitu.nml').write_text(
        "&insitu_run enabled=t, statistics=f, render=t, output_directory='outdat',\n"
        "schedule_mode='steps', step_interval=1, initial_frame=f, final_frame=f,\n"
        'host_budget_bytes=17179869184, device_budget_bytes=6442450944,\n'
        'device_reserve_bytes=2147483648,\n'
        f"implementation_path='{PV/'lib/catalyst'}',\n"
        f"pipeline_file='{ROOT/'scripts/insitu/tgv256_demo.py'}'\n/\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith('ASTR_')}
    for key in ('DISPLAY', 'PYTHONPATH', 'CATALYST_IMPLEMENTATION_PREFER_ENV', 'VTK_EGL_DEVICE_INDEX'):
        env.pop(key, None)
    env.update(ASTR_INSITU_CONFIG=str(case/'insitu.nml'), ASTR_FORCE_MPI_TOPOLOGY='2,1,1',
               ASTR_GPU_SYNC_MODE='explicit', ASTR_GPU_HALO_TRANSPORT='pinned',
               ASTR_GPU_PRECISION_MODE='fp64', ASTR_GPU_BENCHMARK_NO_FIELD_IO='1',
               ASTR_GPU_RK_TIMING='1')
    command = [str(MPI), '--mca', 'coll_hcoll_enable', '0', '-np', '2',
               str(ROOT/'build_insitu_native_gpu/bin/astr'), 'run', 'datin/input.tgv']
    with (case/'run.log').open('w') as log:
        subprocess.run(command, cwd=case, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    out = case/'outdat'
    for rank in (0, 1):
        record = json.loads((out/f'demo_rank{rank}.json').read_text())
        assert record['finalized']
        assert [frame[0] for frame in record['frames']] == list(range(1, args.steps+1))
    assert not list(out.rglob('*.h5')) and not list(out.rglob('COMPLETE')) and not list(out.glob('restart*'))
    for product in ('q0_speed', 'streamlines_speed'):
        assert len(list(out.glob(f'{product}.*.jpeg'))) == args.steps
        assert len(list(out.glob(f'{product}.*.eps'))) == args.steps
    resources = []
    for rank in (0, 1):
        with (out/f'resources.rank{rank:08d}.csv').open() as stream:
            identity = stream.readline().strip()
            rows = list(csv.DictReader(stream))
        assert sum(row['stage'] == 'frame_after' for row in rows) == args.steps
        assert max(int(row['host_increment_bytes']) for row in rows) <= 16*1024**3
        assert max(int(row['device_increment_bytes']) for row in rows) <= 6*1024**3
        assert min(int(row['device_free_bytes']) for row in rows) >= 2*1024**3
        resources.append(dict(rank=rank, identity=identity,
                              host_increment_peak=max(int(row['host_increment_bytes']) for row in rows),
                              device_increment_peak=max(int(row['device_increment_bytes']) for row in rows)))
    videos = {}
    if args.encode:
        sys.path.insert(0, VIDEO_MODULES)
        import imageio_ffmpeg
        for product in ('q0_speed', 'streamlines_speed'):
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-n', '-framerate', '20',
                            '-start_number', '1', '-i', str(out/f'{product}.step%08d.jpeg'),
                            '-frames:v', str(args.steps), '-c:v', 'libx264', '-crf', '18',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(case/f'{product}.mp4')], check=True)
            videos[product] = inspect_video(imageio_ffmpeg, case/f'{product}.mp4', out, product, args.steps)
    (case/'summary.json').write_text(json.dumps(dict(status='passed-render-only-demo',
        steps=args.steps, grid=[256]*3, topology=[2, 1, 1], dt=1.e-4,
        physical_end=args.steps*1.e-4, q_threshold=0., color='speed',
        fps=20, videos_encoded=args.encode, video_checks=videos, resources=resources), indent=2)+'\n')
    print(f'PASS: {args.steps} complete steps; both frame sequences verified: {case}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, choices=(2, 100), required=True)
    p.add_argument('--encode', action='store_true')
    args = p.parse_args()
    case = args.output.resolve()
    if case.exists():
        raise FileExistsError(case)
    try:
        execute(args)
    except BaseException as error:
        if case.is_dir():
            completed = {}
            receipt_errors = {}
            for rank in (0, 1):
                path = case/'outdat'/f'demo_rank{rank}.json'
                try:
                    completed[str(rank)] = len(json.loads(path.read_text())['frames']) if path.exists() else 0
                except (OSError, ValueError, KeyError, TypeError) as receipt_error:
                    completed[str(rank)] = None
                    receipt_errors[str(rank)] = str(receipt_error)
            (case/'summary.json').write_text(json.dumps(dict(status='failed', error=str(error),
                requested_steps=args.steps, completed_frames_by_rank=completed,
                receipt_errors=receipt_errors), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
