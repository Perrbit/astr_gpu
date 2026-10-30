"""Prepare/run the approved two-GPU render-only demonstration and encode movies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_tgv256_visualization as benchmark

ROOT = Path(__file__).resolve().parents[2]
MPI = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')


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
        original = np.asarray(Image.open(out/f'frame.step{step:012d}.jpeg').convert('RGB'))
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
    if args.library is None:
        raise ValueError('Launching requires an explicit --library Catalyst implementation directory')
    benchmark.prepare(case,args.steps,True,args.library,args.transport,args.rendering_pipeline)
    command = [str(args.mpiexec), '--mca', 'coll_hcoll_enable', '0', '-np', '2',
               str(args.executable.resolve(strict=True)), 'run', 'datin/input.tgv']
    with (case/'run.log').open('w') as log:
        subprocess.run(command, cwd=case, env=benchmark.environment(case,args.transport),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    checks=benchmark.verify(case,args.steps,True,args.rendering_pipeline)
    sequences={name:next((case/'outdat/output/render_products'/name).glob('version*/segment*'))
        for name in benchmark.PRODUCTS}
    videos = {}
    if args.encode:
        import imageio_ffmpeg
        for product,source in (('q0_speed','q_surface'),('streamlines_speed','instantaneous_streamlines')):
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-n', '-framerate', '20',
                            '-start_number', '1', '-i', str(sequences[source]/'frame.step%012d.jpeg'),
                            '-frames:v', str(args.steps), '-c:v', 'libx264', '-crf', '18',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(case/f'{product}.mp4')], check=True)
            videos[product] = inspect_video(imageio_ffmpeg, case/f'{product}.mp4', sequences[source], product, args.steps)
    (case/'summary.json').write_text(json.dumps(dict(status='passed-render-only-demo',
        steps=args.steps, grid=[256]*3, topology=[2, 1, 1], dt=1.e-4,
        physical_end=args.steps*1.e-4, q_threshold=0., color='speed',
        fps=20, videos_encoded=args.encode, video_checks=videos, resources=checks['resources'],
        initial_state_frame=True,video_step_range=[1,args.steps],
        image_sequences={name:str(path.relative_to(case)) for name,path in sequences.items()},
        timing=benchmark.timing(case,args.steps)), indent=2)+'\n')
    print(f'PASS: {args.steps} complete steps; both frame sequences verified: {case}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, choices=(2, 100), required=True)
    p.add_argument('--encode', action='store_true')
    p.add_argument('--executable',type=Path,default=ROOT/'build_insitu_device_render/bin/astr')
    p.add_argument('--mpiexec',type=Path,default=MPI)
    p.add_argument('--library',type=Path,help='Explicit Catalyst ParaView implementation directory')
    p.add_argument('--transport',choices=('pinned','device-aware'),default='pinned')
    p.add_argument('--rendering-pipeline',choices=('standard-device','direct-device','compatible'),default='standard-device')
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
                try:
                    frames=[]
                    for product in benchmark.PRODUCTS:
                        successful=set()
                        for path in (case/'outdat/output/render_products'/product).glob(
                                f'version*/segment*/frame.*.rank{rank:08d}.json'):
                            record=json.loads(path.read_text())
                            if record['product']['image']=='published' and 1<=record['step']<=args.steps:
                                successful.add(record['step'])
                        frames.append(successful)
                    completed[str(rank)] = len(set.intersection(*frames))
                except (OSError, ValueError, KeyError, TypeError) as receipt_error:
                    completed[str(rank)] = None
                    receipt_errors[str(rank)] = str(receipt_error)
            (case/'summary.json').write_text(json.dumps(dict(status='failed', error=str(error),
                requested_steps=args.steps, completed_frames_by_rank=completed,
                receipt_errors=receipt_errors), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
