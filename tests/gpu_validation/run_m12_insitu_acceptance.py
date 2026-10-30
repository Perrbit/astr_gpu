#!/usr/bin/env python3
"""Bounded M12 graphics/isolation gate, not a production or transfer benchmark."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from run_m12_local_compare import TOPOLOGIES, check_pair, run
from catalog_image_inspection import inspect_images

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
from catalog_examples import m12_short_products


def visible_geometry_pixels(pixels):
    # The fixed legend is at the right edge; it is not evidence of visible geometry.
    region = pixels[:, :int(pixels.shape[1]*.9)]
    return int(np.count_nonzero(np.any(region < 220, axis=-1)))


def configuration(library, host_budget_bytes=4*1024**3, device_budget_bytes=2*1024**3,
                  device_reserve_bytes=1024**3):
    pipeline = ROOT/'scripts/insitu/device_render_pipeline.py'
    return f"""&insitu_run
 enabled=t, statistics=f, render=t,
 processing_backend='device', derivative_backend='gpu',
 rendering_pipeline='standard-device', postprocess_transport='pinned',
 products='boundary_layer', streamline_seeds='bl-layered64',
 slice_definition='plane', slice_origin=0.,0.,45., slice_normal=0.,0.,1.,
 output_directory='outdat/render',
 host_budget_bytes={host_budget_bytes}, device_budget_bytes={device_budget_bytes}, device_reserve_bytes={device_reserve_bytes},
 implementation_path='{library}', pipeline_file='{pipeline}'
/
"""


def check_images(case, ranks, audit=True):
    result=inspect_images(case,ranks,10)
    if not result['publication_complete']:
        raise ValueError('Native M12 publication is incomplete: '+str(result['warnings']))
    result['authority_check']='paired final checkpoint/statistics in caller; no legacy profile receipt assumed'
    return result


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
                 resource_baseline=baseline,product_catalog_records=m12_short_products())
        receipt = check_pair(off, on, ranks, reference_backend='gpu')
        if receipt['q']['max_abs'] != 0. or any(receipt['statistics_max_abs'].values()):
            raise ValueError('graphics observer changes authoritative state or statistics')
        receipt['graphics'] = check_images(on, ranks)
        receipt['graphics']['memcheck'] = bool(args.memcheck)
        (args.output/f'graphics_{topology.replace(",", "x")}.json').write_text(json.dumps(receipt, indent=2)+'\n')
        print('PASS M12 render/isolation topology='+topology, flush=True)


if __name__ == '__main__':
    main()
