#!/usr/bin/env python3
"""Bounded M12 graphics/isolation gate, not a production or transfer benchmark."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from run_m12_local_compare import TOPOLOGIES, check_pair, run

ROOT = Path(__file__).resolve().parents[2]
PRODUCTS = ('q_surface', 'velocity_slice', 'instantaneous_streamlines')
STEPS = (5, 10)


def visible_geometry_pixels(pixels):
    # The fixed legend is at the right edge; it is not evidence of visible geometry.
    region = pixels[:, :int(pixels.shape[1]*.9)]
    return int(np.count_nonzero(np.any(region < 220, axis=-1)))


def configuration(library):
    pipeline = ROOT/'scripts/insitu/device_render_pipeline.py'
    return f"""&insitu_run
 enabled=t, statistics=f, render=t, initial_frame=f, final_frame=f,
 processing_backend='device', derivative_backend='gpu',
 rendering_pipeline='standard-device', postprocess_transport='pinned',
 products='boundary_layer', streamline_seeds='bl-layered64',
 slice_definition='plane', slice_origin=0.,0.,45., slice_normal=0.,0.,1.,
 output_directory='outdat/render', schedule_mode='steps', step_interval=5,
 host_budget_bytes=4294967296, device_budget_bytes=2147483648, device_reserve_bytes=1073741824,
 implementation_path='{library}', pipeline_file='{pipeline}'
/
"""


def check_images(case, ranks, audit=True):
    directory = case/'outdat/render'
    for step in STEPS:
        for rank in range(ranks):
            record = json.loads((directory/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            if record['rendering_pipeline'] != 'standard-device' or record['geometry_host_bytes'] != 0 or \
                    set(record['products']) != set(PRODUCTS):
                raise ValueError('render product/pipeline identity differs')
            if record['products']['instantaneous_streamlines']['seeding'] != dict(
                    layout='bl-layered64', seeds=64, direction_trajectories=128):
                raise ValueError('streamline seed identity differs')
            for product in PRODUCTS:
                item = record['products'][product]
                if item['physical_bounds'] != [0., 1100., 0., 150., 0., 90.] or \
                        item['color_field'] != 'speed' or item['color_range'] != [0., 1.1]:
                    raise ValueError('physical frame/color identity differs')
                expected_camera = dict(position=[550., 375. if product == 'q_surface' else 75., 900.],
                                       focal_point=[550., 75., 45.], view_up=[0., 1., 0.], parallel_scale=150.)
                if item['camera'] != expected_camera:
                    raise ValueError('D7 changes only the approved Q camera')
            if record['products']['velocity_slice']['plane'] != dict(origin=[0., 0., 45.], normal=[0., 0., 1.]):
                raise ValueError('physical slice identity differs')
        for product in PRODUCTS:
            if not any(json.loads((directory/f'mesh_step{step:08d}_rank{rank}.json').read_text())
                       ['products'][product]['local_cells'] > 0 for rank in range(ranks)):
                raise ValueError('globally empty geometry: '+product)
        for product in PRODUCTS:
            path = directory/f'{product}.step{step:08d}.jpeg'
            if not path.with_suffix('.eps').is_file():
                raise ValueError('missing EPS counterpart')
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
            if pixels.shape != (384, 1536, 3) or np.count_nonzero(np.any(pixels < 220, axis=-1)) < 500:
                raise ValueError('blank or incorrectly sized image: '+str(path))
            if visible_geometry_pixels(pixels) == 0:
                raise ValueError('legend-only image without visible geometry: '+str(path))
    expected = {f'{product}.step{step:08d}.jpeg' for step in STEPS for product in PRODUCTS}
    if {p.name for p in directory.glob('*.jpeg')} != expected or list(directory.glob('*.vtp')):
        raise ValueError('unexpected frames or host geometry files')
    if not audit:
        return {'products': list(PRODUCTS), 'steps': list(STEPS), 'image_size': [1536, 384],
                'transfer_scope': 'clean image gate; actual transfers checked separately'}
    log = (case/'run.log').read_text()
    isolation = re.findall(r'ASTR_M12_RENDER_ISOLATION rank=(\d+) step=(\d+) q_bits_rng_unchanged=1', log)
    if set(isolation) != {(str(rank), str(step)) for rank in range(ranks) for step in STEPS}:
        raise ValueError('missing render authority/RNG bit-isolation proof')
    audits = re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=(\d+) step=(\d+) product=(\S+) '
                        r'field_maxabs=(\S+) display_maxabs=(\S+) points=(\d+)', log)
    selected = [row for row in audits if row[2] in ('q_surface', 'instantaneous_streamlines')]
    if len(selected) != ranks*len(STEPS)*2 or any(not np.isfinite(float(row[3])) or
            float(row[3]) > 2e-10 or float(row[4]) != 0. for row in selected):
        raise ValueError('resident field/display audit failed')
    return {'products': list(PRODUCTS), 'steps': list(STEPS), 'image_size': [1536, 384],
            'numerical_audits': len(selected),
            'transfer_scope': 'instrumented numerical gate; validation field downloads are present'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--topologies', nargs='+', choices=TOPOLOGIES, default=('1,1,1',))
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--sanitizer', type=Path)
    args = parser.parse_args()
    if args.memcheck and args.sanitizer is None:
        parser.error('--memcheck requires --sanitizer')
    if args.memcheck:
        args.sanitizer = args.sanitizer.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    args.executable = args.executable.resolve(strict=True)
    args.library = args.library.resolve(strict=True)
    args.diagnostics = False
    for topology in args.topologies:
        ranks = int(np.prod([int(v) for v in topology.split(',')]))
        off = run(args, 'gpu', topology, case_label='off')
        baseline = json.loads((off/'resources.sampled.json').read_text())
        on = run(args, 'gpu', topology, insitu_config=configuration(args.library), case_label='on',
                 resource_baseline=baseline)
        receipt = check_pair(off, on, ranks, reference_backend='gpu')
        if receipt['q']['max_abs'] != 0. or any(receipt['statistics_max_abs'].values()):
            raise ValueError('graphics observer changes authoritative state or statistics')
        receipt['graphics'] = check_images(on, ranks)
        receipt['graphics']['memcheck'] = bool(args.memcheck)
        (args.output/f'graphics_{topology.replace(",", "x")}.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('PASS M12 render/isolation topology='+topology, flush=True)


if __name__ == '__main__':
    main()
