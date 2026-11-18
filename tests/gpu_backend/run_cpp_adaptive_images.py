"""Existing adaptive event/window clocks with actual dynamic device image products."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments
from test_runtime_catalog_images import image_product,q_product,line_product,image_segments
from test_insitu_device_products import resident_configuration
from test_adaptive_output_products import RENDER_CONFIG
from run_output_restart_validation import run_case,archive_schedule_payload
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_cpp_restart_gate import exact_state
from run_cpp_adaptive_output_gate import monitor,compare_observation


def definitions():
    items=[q_product('event-q'),line_product('event-lines'),image_product('window-slice')]
    items[0]['schedule']=dict(mode='steps',interval=6,events=['decay'],dense_interval=2)
    items[1]['schedule']=dict(mode='time',interval=.008,events=['decay'],dense_interval=.002)
    items[2]['schedule']=dict(mode='steps',interval=6,windows=['burst'],dense_interval=2)
    return items


def expected_steps(case,record):
    rows,flags=monitor(case)
    last_step=None;last_time=None;was_dense=False;due=[]
    for row,flag in zip(rows,flags):
        step,time=int(row[0]),float(row[1])
        dense=flag[0][0]=='T' if 'events' in record['schedule'] else flag[1][0]==1
        clock=record['schedule'];interval=clock['dense_interval'] if dense else clock['interval']
        expired=(last_step is None or (step-last_step>=interval if clock['mode']=='steps' else time-last_time>=interval))
        if expired or dense and not was_dense:
            due.append(step);last_step=step;last_time=time
        was_dense=dense
    return due


def inspect(case,ranks,records,first=0):
    output={p.parent.parent.name:p for p in image_segments(case)}
    if set(output)!={r['id'] for r in records}:raise AssertionError('adaptive image identities')
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=3 or any(p['state']!='active' for p in status['products']):
        raise AssertionError(('adaptive image publication status',status))
    pictures={};clocks={int(r[0]):r[2] for r in logged_steps(case,count=12-first,first_step=first+1)}
    clocks[0]=0.
    for record in records:
        directory=output[record['id']]
        files=sorted(directory.glob('*.jpeg'))
        steps=[int(p.stem.split('step')[1]) for p in files]
        expected=[s for s in expected_steps(case,record) if s>first or first==0]
        if steps!=expected:raise AssertionError(('adaptive frame clock',record['id'],steps,expected))
        receipts=[json.loads(p.read_text()) for p in directory.glob('frame.*.rank*.json')]
        if len(receipts)!=len(steps)*ranks:raise AssertionError('adaptive image receipt count')
        for receipt in receipts:
            if receipt['field_d2h_bytes'] or receipt['geometry_host_bytes']:
                raise AssertionError('adaptive image downloaded flow or geometry')
            if receipt['step'] not in steps or receipt['time']!=clocks[receipt['step']]:
                raise AssertionError('adaptive image is not at the completed-step clock')
        for step in steps:
            if sum(r['product']['local_cells'] for r in receipts if r['step']==step)<=0:
                raise AssertionError('empty adaptive product geometry')
            path=directory/f'frame.step{step:012d}.jpeg';pixels=np.asarray(Image.open(path).convert('RGB'))
            if pixels.shape!=(600,800,3) or pixels.std()<1:raise AssertionError('adaptive image raster')
            pictures[record['id']+f'/{step}']=pixels
        if list(directory.glob('*.eps')):raise AssertionError('EPS was disabled')
    return output,pictures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','original','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=2)
    parser.add_argument('--pipeline',choices=('standard-device','direct-device'),default='standard-device')
    parser.add_argument('--reuse-original-case',type=Path,help='explicit completed original case after an analysis-only correction')
    parser.add_argument('--reuse-completed-images',action='store_true',help='reuse the completed full/prefix/tail after an analysis-only correction; run only the unstarted isolation cases')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=args.reuse_completed_images)
    setup=arguments(out);setup.executable=args.binary.resolve();setup.mpiexec=str(args.mpiexec.resolve())
    setup.unrestricted=True;setup.statistics=False;setup.insitu_write_eps='0';setup.axis='x'
    config=resident_configuration(args.pipeline,'pinned',statistics=False,interval=1,profile='velocity_slice')
    unlimited=np.iinfo(np.int64).max//4096
    config=config.replace('host_budget_bytes=4294967296',f'host_budget_bytes={unlimited}')
    config=config.replace('device_budget_bytes=2147483648',f'device_budget_bytes={unlimited}')
    config=config.replace('device_reserve_bytes=1073741824','device_reserve_bytes=0')
    common=dict(grid='32,32,32',tgv_reynolds=1,checkpoint_interval=5,checkpoint_keep=2,
        buffer_bytes=1048576,postprocess_transport='pinned',solver_halo_transport='pinned',insitu_timing=True)
    records=definitions()
    def run(label,binary,steps=12,restore=None,products=True,adaptive=True,reuse=False):
        setup.executable=binary.resolve()
        if reuse:
            case=out/f'gpu_np{args.ranks}_{label}'
            if 'The job is done!' not in (case/'run.log').read_text():raise AssertionError('reused image run did not finish')
            logged_steps(case,count=steps-(5 if restore else 0),first_step=6 if restore else 1)
            return case
        return run_case(setup,ROOT,'gpu',args.ranks,label,steps,restore=restore,insitu_config=config,
            product_catalog_records=records if products and restore is None else [],
            adaptive_config=RENDER_CONFIG if adaptive else None,**common)[0]
    original=args.reuse_original_case.resolve() if args.reuse_original_case else run('original',args.original)
    old_dirs,old_pixels=inspect(original,args.ranks,records)
    actual=run('cpp',args.binary,reuse=args.reuse_completed_images);actual_dirs,pixels=inspect(actual,args.ranks,records)
    final=Path('outdat/new/checkpoints/step000000000012')
    report=dict(ranks=args.ranks,pipeline=args.pipeline,binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        original_sha256=hashlib.sha256(args.original.read_bytes()).hexdigest(),
        original_case=str(original),
        monitor_scaled=compare_observation(actual,original),
        state_scaled=compare(checkpoint(actual/final/'state.h5'),checkpoint(original/final/'state.h5')),
        products={r['id']:expected_steps(actual,r) for r in records},pixel_differences={})
    if set(pixels)!=set(old_pixels):raise AssertionError('original/new adaptive image clock differs')
    for name,value in pixels.items():
        error=abs(value.astype(float)-old_pixels[name].astype(float))
        report['pixel_differences'][name]=dict(mean_abs=float(error.mean()),max_abs=float(error.max()))
    prefix=run('prefix',args.binary,steps=5,reuse=args.reuse_completed_images)
    tail=run('tail',args.binary,restore=prefix/'outdat/new/checkpoints/step000000000005',reuse=args.reuse_completed_images)
    exact_state(actual/final/'state.h5',tail/final/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        if (actual/final/name).read_bytes()!=(tail/final/name).read_bytes():raise AssertionError('adaptive image control continuation '+name)
    if archive_schedule_payload(actual/final/'archives.bin')!=archive_schedule_payload(tail/final/'archives.bin'):
        raise AssertionError('adaptive image schedule continuation')
    tail_dirs={p.parent.parent.name:p for p in image_segments(tail)}
    for record in records:
        name=record['id'];expected=[s for s in report['products'][name] if s>5]
        if not expected:
            if name in tail_dirs and list(tail_dirs[name].glob('*.jpeg')):
                raise AssertionError('an image was emitted with no restored event due')
            continue
        paths=sorted(tail_dirs[name].glob('*.jpeg'))
        if [int(p.stem.split('step')[1]) for p in paths]!=expected:raise AssertionError('restored adaptive image clock')
        for path in paths:
            if path.read_bytes()!=(actual_dirs[name]/path.name).read_bytes():raise AssertionError('restored adaptive image pixels')
    for label,products,adaptive in (('products-off',False,True),('monitor-and-products-off',False,False)):
        quiet=run(label,args.binary,products=products,adaptive=adaptive)
        exact_state(actual/final/'state.h5',quiet/final/'state.h5')
        if logged_steps(actual,count=12).tolist()!=logged_steps(quiet,count=12).tolist():
            raise AssertionError('adaptive observation changed time/dt')
    report.update(exact_state_controls_schedules_images=True,monitor_product_state_clock_isolation=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('adaptive dynamic device images: reference, clocks, restart and observation isolation checked')


if __name__=='__main__':main()
