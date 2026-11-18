"""Real native C++ Reynolds/Favre slice and streamline consumers."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments,product
from test_runtime_catalog_images import image_product, line_product, image_segments
from test_insitu_device_products import resident_configuration
from run_output_restart_validation import run_case
from run_cpp_insitu_gate import compare_hdf
from run_cpp_statistics_gate import configuration, compare_statistics
from run_cpp_tgv_gate import logged_steps
from run_native_precision_restart_gate import check_state


def definitions(statistics_source='velocity-time'):
    items = [image_product('reynolds-slice'),image_product('favre-slice'),
             line_product('favre-lines',source='favre')]
    if statistics_source=='selected-time':
        items[1]=image_product('favre-slice',field='temperature')
        items[1]['display']['color_range']=[.99,1.01]
        items[2]=image_product('reynolds-pressure-fluct',field='pressure')
        items[2]['fluctuation']=True
        items[2]['sampling']=dict(mode='grid_index',axis='j',index=4,index_ranges=dict(i=[0,32],k=[0,32]))
        items[2]['display'].update(camera=[np.pi,13.,np.pi],focal_point=[np.pi,np.pi/4,np.pi],view_up=[0.,0.,1.])
        items[2]['display']['color_range']=[-.05,.05]
    for i,item in enumerate(items):
        item['source'] = 'reynolds' if i==0 or (statistics_source=='selected-time' and i==2) else 'favre'
        item['statistics_source'] = statistics_source
    return items


def inspect_images(case, ranks, expected_steps, records=None, first_step=1, count=4):
    expected={p['id']:p['source'] for p in (records if records is not None else definitions())}
    directories = {p.parent.parent.name:p for p in image_segments(case)}
    if set(directories) != set(expected):
        raise AssertionError(('mean image identities',directories))
    status = json.loads((case/'outdat/new/products.status.json').read_text())
    event_times = {int(row[0]):row[2] for row in logged_steps(case,count=count,first_step=first_step)}
    if len(status['products'])!=3 or any(p['state']!='active' or p['last_output_step']!=3 for p in status['products']):
        raise AssertionError(status)
    for name,directory in directories.items():
        if [p.name for p in sorted(directory.glob('*.jpeg'))] != [f'frame.step{step:012d}.jpeg' for step in expected_steps]:
            raise AssertionError(('covered mean output clock',name))
        receipts = [json.loads(p.read_text()) for p in directory.glob('frame.*.rank*.json')]
        if len(receipts)!=len(expected_steps)*ranks:
            raise AssertionError('mean image receipt count')
        for receipt in receipts:
            if receipt['field_d2h_bytes']!=0 or receipt['geometry_host_bytes']!=0:
                raise AssertionError('mean image read back field/geometry')
            if receipt['source']!=expected[name]:
                raise AssertionError('mean image silently used another source')
            if receipt['step'] not in expected_steps or receipt['time']!=event_times[receipt['step']]:
                raise AssertionError('mean image complete-step phase')
        for step in expected_steps:
            if sum(r['product']['local_cells'] for r in receipts if r['step']==step)<=0:
                raise AssertionError('empty mean image geometry')
            pixels = np.asarray(Image.open(directory/f'frame.step{step:012d}.jpeg').convert('RGB'))
            if pixels.shape!=(600,800,3) or pixels.std()<1:
                raise AssertionError('blank/dimensionally incorrect mean image')
        if list(directory.glob('*.eps')):
            raise AssertionError('EPS was disabled')
    return directories


def pixel_differences(actual,reference,steps):
    differences = {}
    for name,directory in actual.items():
        values=[]
        for step in steps:
            image=f'frame.step{step:012d}.jpeg'
            x=np.asarray(Image.open(directory/image).convert('RGB'),dtype=float)
            y=np.asarray(Image.open(reference[name]/image).convert('RGB'),dtype=float)
            values.append(float(np.mean(abs(x-y))))
        differences[name]=values
    return differences


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ranks',type=int,default=1)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--mapping',choices=['periodic'])
    parser.add_argument('--pipeline',choices=['standard-device','direct-device'],default='standard-device')
    parser.add_argument('--transport',choices=['pinned','device-aware'],default='pinned')
    parser.add_argument('--statistics-source',choices=['velocity-time','selected-time'],default='velocity-time')
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    options=parser.parse_args()
    output=options.output.resolve();output.mkdir(parents=True,exist_ok=False)
    args=arguments(output);args.executable=options.binary.resolve();args.axis=options.axis
    args.unrestricted=True;args.insitu_write_eps='0'
    args.native_precision=options.precision
    limit=None if options.precision=='fp32' else 2e-10
    config=resident_configuration(options.pipeline,options.transport,statistics=True,interval=2,
        profile='streamlines' if options.mapping else 'velocity_slice')
    unlimited=np.iinfo(np.int64).max//4096
    config=config.replace('host_budget_bytes=4294967296',f'host_budget_bytes={unlimited}')
    config=config.replace('device_budget_bytes=2147483648',f'device_budget_bytes={unlimited}')
    config=config.replace('device_reserve_bytes=1073741824','device_reserve_bytes=0')
    selected=options.statistics_source=='selected-time'
    records=definitions(options.statistics_source)
    if selected:
        profile='streamlines' if options.mapping else 'velocity_slice'
        config=config.replace(f"products='{profile}'","products='catalog_stats'")
        config=config.replace('statistics_window=0.0005,0.0115','statistics_window=0.0005,0.0035')
    quiet_config=(config.replace('render=.true.','render=.false.').replace(
        f"rendering_pipeline='{options.pipeline}'","rendering_pipeline='compatible'")
        if selected else configuration(True))
    quiet_records=[]
    if selected:
        for image in records:
            numeric=product(image['id'],[image['field']],image['source'])
            for key in ('sampling','statistics_source','schedule'):
                numeric[key]=image[key]
            if image.get('fluctuation'):
                numeric['fluctuation']=True
            quiet_records.append(numeric)
    if options.mapping and options.precision=='fp32':
        args.scale_timestep=2e-5;args.maximum_cfl=.5
        config=config.replace('statistics_window=0.0005,0.0115','statistics_window=0.00001,0.0002')
        config=config.replace('statistics_window=0.0005,0.0035','statistics_window=0.00001,0.00007')
        quiet_config=quiet_config.replace('statistics_window=.0005,.0115','statistics_window=.00001,.0002')
        quiet_config=quiet_config.replace('statistics_window=0.0005,0.0035','statistics_window=0.00001,0.00007')
    common=dict(grid='32,32,32',tgv_mapping=options.mapping,numerics=dict(filter_scheme='explicit10'),checkpoint_interval=2,
        gpu_precision_mode='' if options.precision=='fp32' else 'fp64')
    args.executable=options.reference.resolve()
    old,_=run_case(args,ROOT,'gpu',options.ranks,'old-gpu-mean-images',4,insitu_config=config,
        product_catalog_records=records,postprocess_transport=options.transport,**common)
    reference=inspect_images(old,options.ranks,(1,3),records)
    args.executable=options.binary.resolve()
    quiet,_=run_case(args,ROOT,'gpu',options.ranks,'cpp-statistics-only',4,
        insitu_config=quiet_config,
        product_catalog_records=quiet_records,**common)
    current,_=run_case(args,ROOT,'gpu',options.ranks,'cpp-mean-images',4,insitu_config=config,
        product_catalog_records=records,postprocess_transport=options.transport,**common)
    final='outdat/new/checkpoints/step000000000004'
    if 'backend=cuda' not in (current/'run.log').read_text():
        raise AssertionError('native C++ backend not selected')
    for name in ('state.h5','statistics.h5'):
        compare_hdf(current/final/name,quiet/final/name,exact=True)
    actual=inspect_images(current,options.ranks,(1,3),records)
    report=dict(ranks=options.ranks,axis=options.axis,mapping=options.mapping,pipeline=options.pipeline,
        precision=options.precision,difference_limit=limit,
        transport=options.transport,statistics_source=options.statistics_source,
        terminal_scaled=compare_hdf(current/final/'state.h5',old/final/'state.h5',limit=limit),
        physical_state=check_state(current/final/'state.h5',options.precision),
        authoritative_and_statistics_on_off_bits=True,field_and_geometry_readback_bytes=0,
        old_new_pixel_mean_absolute_difference=pixel_differences(actual,reference,(1,3)))
    if not selected:
        report['statistics']=compare_statistics(current/final/'statistics.h5',old/final/'statistics.h5',32,
            precision=options.precision)
    args.executable=options.binary.resolve()
    prefix,_=run_case(args,ROOT,'gpu',options.ranks,'cpp-prefix',2,insitu_config=config,
        product_catalog_records=records,postprocess_transport=options.transport,**common)
    resumed,_=run_case(args,ROOT,'gpu',options.ranks,'cpp-resumed',4,restore=prefix/'outdat/new/checkpoints/step000000000002',
        insitu_config=config,product_catalog_records=[],postprocess_transport=options.transport,**common)
    for name in ('state.h5','statistics.h5'):
        compare_hdf(current/final/name,resumed/final/name,exact=True)
    for name in ('control.bin','insitu_control.bin'):
        if (current/final/name).read_bytes()!=(resumed/final/name).read_bytes():
            raise AssertionError('mean image exact control differs: '+name)
    again=inspect_images(resumed,options.ranks,(3,),records,first_step=3,count=2)
    differences=pixel_differences(again,actual,(3,))
    if any(any(v!=0 for v in values) for values in differences.values()):
        raise AssertionError(('resumed mean image pixels differ',differences))
    report['same_backend_restart']=dict(state_bits=True,statistics_bits=True,saved_catalog_and_clock=True,pixels=differences)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
