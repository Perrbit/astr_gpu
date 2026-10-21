"""Native M12 resident products, observer invariance and same-kind restart."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from run_m12_local_compare import run
from run_m12_insitu_acceptance import configuration, visible_geometry_pixels, PRODUCTS
from run_output_restart_validation import compare_fields
from run_native_precision_restart_gate import check_state


FINAL = 'outdat/new/checkpoints/step000000000010'


def check_images(case, pipeline, steps, precision):
    output=case/'outdat/render'
    for step in steps:
        cells=dict.fromkeys(PRODUCTS,0)
        for rank in range(2):
            receipt=json.loads((output/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            if receipt['rendering_pipeline']!=pipeline or receipt['geometry_host_bytes']!=0 or \
                set(receipt['products'])!=set(PRODUCTS):
                raise AssertionError('M12 native resident product identity differs')
            for product in PRODUCTS:
                if receipt['products'][product]['color_range']!=[0.,1.1]:
                    raise AssertionError('M12 native color controls differ')
                cells[product]+=receipt['products'][product]['local_cells']
        if any(count<=0 for count in cells.values()):
            raise AssertionError('M12 globally empty native product')
        for product in PRODUCTS:
            path=output/f'{product}.step{step:08d}.jpeg'
            with Image.open(path) as source:
                pixels=np.asarray(source.convert('RGB'))
            if (pixels.shape!=(384,1536,3) or np.count_nonzero(np.any(pixels<220,axis=-1))<500 or
                visible_geometry_pixels(pixels)==0):
                raise AssertionError('M12 missing/blank native product')
    if list(output.glob('*.eps')) or list(output.glob('*.vtp')) or list(output.glob('sample.*.bin')):
        raise AssertionError('M12 strict JPEG path emitted extra products')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--precision',choices=('fp32','fp64'),required=True)
    parser.add_argument('--source',type=Path,default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--out',type=Path,required=True)
    opts=parser.parse_args()
    root=opts.out.resolve();root.mkdir(exist_ok=False)
    args=SimpleNamespace(source=opts.source.resolve(),output=root,executable=opts.exe.resolve(),
        mpiexec=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'),
        memcheck=False,diagnostics=False,native_precision=opts.precision,insitu_write_eps='0',
        halo_transport='pinned',runtime_timeout_seconds=600,directory_budget_bytes=256*1024**2,
        boundary='nscbc',deltat=.005)
    plain=run(args,'gpu','2,1,1',case_label='off',observation='plain',checkpoint_interval=5,exact_restart=True)
    report=dict(precision=opts.precision,grid_intervals=[64,32,24],topology=[2,1,1],
        deltat=.005,boundaries=[11,22,41,52,1,1],cases={})
    # The existing boundary_layer preset admits standard-device only.
    for pipeline in ('standard-device',):
        config=configuration(opts.library.resolve()).replace("rendering_pipeline='standard-device'",
                                                             f"rendering_pipeline='{pipeline}'")
        active=run(args,'gpu','2,1,1',case_label=pipeline,insitu_config=config,observation='images',
                   checkpoint_interval=5,exact_restart=True)
        check_images(active,pipeline,(5,10),opts.precision)
        compare_fields(plain/FINAL/'state.h5',active/FINAL/'state.h5')
        seed=run(args,'gpu','2,1,1',case_label=pipeline+'_seed',insitu_config=config,observation='images',
                 steps=5,checkpoint_interval=5,exact_restart=True)
        source=seed/'outdat/new/checkpoints/step000000000005'
        before={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}
        resumed=run(args,'gpu','2,1,1',case_label=pipeline+'_resumed',insitu_config=config,observation='images',
                    restore=source,checkpoint_interval=5,exact_restart=True)
        check_images(resumed,pipeline,(10,),opts.precision)
        compare_fields(active/FINAL/'state.h5',resumed/FINAL/'state.h5')
        if before!={p.name:(p.stat().st_size,p.stat().st_mtime_ns) for p in source.iterdir() if p.is_file()}:
            raise AssertionError('M12 source checkpoint changed')
        for path in (resumed/'outdat/render').glob('*.jpeg'):
            if path.read_bytes()!=(active/'outdat/render'/path.name).read_bytes():
                raise AssertionError('M12 native continuation image differs')
        report['cases'][pipeline]=check_state(resumed/FINAL/'state.h5',opts.precision)
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(pipeline+': native M12 products, invariance and exact restart passed',flush=True)
    report['status']='passed'
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
