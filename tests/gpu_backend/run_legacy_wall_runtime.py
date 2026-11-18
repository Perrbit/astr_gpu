"""Ordinary C++ ASTR legacy wall curves, device history and exact continuation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments
from test_runtime_catalog_walls import wall_frames,rows,wall_chart,chart_segments
from run_output_restart_validation import run_case,archive_schedule_payload
from run_native_precision_restart_gate import check_state
from prepare_legacy_wall_reference import definitions
from run_cpp_statistics_gate import configuration
from run_cpp_restart_gate import exact_state
from run_cpp_tgv_gate import checkpoint,compare,logged_steps


def curves(case,tail=False,images=False,nodes=33):
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=12+2*int(images) or any(p['state']!='active' for p in status['products']):
        raise AssertionError(('legacy wall products',status))
    values={}
    for record in definitions():
        files=wall_frames(case,record['id'])
        expected=([4] if tail else [0,2,4]) if record['source']=='instantaneous' else ([3] if tail else [1,3])
        if [int(p.stem.split('step')[1]) for p in files]!=expected:raise AssertionError('wall product completed-step identities')
        for path in files:
            table=rows(path);array=np.asarray([[float(v) for v in row.values()] for row in table])
            if list(table[0])!=['step','time','global_i','x','y','z','value'] or array.shape!=(nodes,7) or not np.isfinite(array).all():
                raise AssertionError('wall CSV schema/finiteness')
            values[record['id']+'/'+path.name]=array
    return values


def image_definitions():
    instant=wall_chart('cpp-wall-pressure','wall_pressure')
    mean=wall_chart('cpp-wall-mean-shear','wall_shear_x','upper')
    mean.update(source='reynolds',statistics_source='wall-time')
    instant['display']['y_range']=[71.,73.]
    mean['display']['y_range']=[-.09,.09]
    for record in (instant,mean):record['sampling']['bounds']=[-1.,2*np.pi+1.]*3
    return [instant,mean]


def images(case,ranks,tail=False):
    from PIL import Image
    pictures={}
    for record in image_definitions():
        found=chart_segments(case,record['id'])
        if len(found)!=1:raise AssertionError('one segment per wall chart')
        directory=found[0]
        expected=([4] if tail else [0,2,4]) if record['source']=='instantaneous' else ([3] if tail else [1,3])
        files=sorted(directory.glob('*.jpeg'))
        if [int(f.stem.split('step')[1]) for f in files]!=expected or list(directory.glob('*.eps')):
            raise AssertionError('wall chart output clock/formats')
        receipts=list(directory.glob('frame.*.rank*.json'))
        if len(receipts)!=len(expected) or any('rank00000000.json' not in p.name for p in receipts):
            raise AssertionError('wall chart rank-zero publication receipts')
        for receipt in receipts:
            record_json=json.loads(receipt.read_text());step=record_json['step']
            numeric=(('lower-wall_pressure-instantaneous' if record['source']=='instantaneous' else
                      'upper-wall_shear_x-reynolds')+f'/frame.step{step:012d}.csv')
            path=list((case/'outdat/new/wall_products'/numeric.split('/')[0]).glob('*/segment*/'+numeric.split('/')[1]))
            if len(path)!=1:raise AssertionError('one matching numerical curve per chart')
            values=np.asarray([float(row['value']) for row in rows(path[0])])
            dtype=np.float32 if record_json['source_bytes']==4 else np.float64
            if not np.array_equal(np.asarray(record_json['product']['actual_range'],dtype=dtype),
                np.asarray([values.min(),values.max()],dtype=dtype)) or record_json['product']['points']!=33:
                raise AssertionError('chart values differ from numerical wall curve')
            if record['source']=='reynolds' and record_json['statistics']['provider']!='wall-time':
                raise AssertionError('wall mean chart used a different accumulator')
        for path in files:
            pixels=np.asarray(Image.open(path).convert('RGB'))
            if pixels.shape!=(600,800,3) or pixels.std()<1:raise AssertionError('wall chart raster')
            pictures[record['id']+'/'+path.name]=pixels
    text=(case/'run.log').read_text()
    entries=re.findall(r'ASTR_RUNTIME_WALL_CHART rank=(\d+) id=\S+ step=(\d+) host_curve_points=(\d+) '
        r'host_curve_bytes=(\d+) three_dimensional_field_download_bytes=(\d+)',text)
    if len(entries)!=(2 if tail else 5)*ranks or any(int(e[4]) for e in entries):
        raise AssertionError('wall chart read back a three-dimensional field')
    if any(int(e[2])!=33 for e in entries if int(e[0])==0):raise AssertionError('wall chart has an incomplete curve')
    return pictures


def compare_curves(actual,reference,precision,exact=False):
    result={}
    for key,x in actual.items():
        y=reference[key]
        if not np.array_equal(x[:,:3],y[:,:3]):raise AssertionError(('wall clock/index',key))
        if exact:
            if x.tobytes()!=y.tobytes():raise AssertionError(('resumed wall CSV values',key))
        else:
            error=np.max(abs(x[:,3:]-y[:,3:]),axis=0)/np.maximum(1.,np.max(abs(y[:,3:]),axis=0))
            if precision=='fp64' and error.max()>2e-10:raise AssertionError(('wall product error',key,error))
            result[key]=error.tolist()
    return result


def wall_history(actual,reference,precision):
    result={}
    with h5py.File(actual) as a,h5py.File(reference) as b:
        x,y=a['wall_statistics'],b['wall_statistics']
        if x.keys()!=y.keys():raise AssertionError('wall history dataset identity')
        for key in x:
            av,bv=x[key][...],y[key][...]
            if av.dtype!=bv.dtype or av.shape!=bv.shape:raise AssertionError(('wall history shape/kind',key))
            if key=='rank_extras':
                cursor=0
                partitions=x['partitions'][...].reshape(-1,8)
                for rank,part in enumerate(partitions):
                    count=int(part[7])
                    if count:
                        first=av[cursor:cursor+count].reshape(34,-1).astype(float)
                        second=bv[cursor:cursor+count].reshape(34,-1).astype(float)
                        error=np.max(abs(first-second),axis=1)/np.maximum(1.,np.max(abs(second),axis=1))
                        if not np.isfinite(error).all() or precision=='fp64' and error.max()>2e-10:
                            raise AssertionError(('wall duplicate history',rank,error))
                        result[key+'_rank'+str(rank)]=error.tolist()
                    cursor+=count
                if cursor!=av.size:raise AssertionError('wall duplicate packing count')
            elif key.startswith('q'):
                error=float(np.max(abs(av.astype(float)-bv.astype(float)))/max(1.,float(np.max(abs(bv)))))
                if not np.isfinite(error) or precision=='fp64' and error>2e-10:raise AssertionError(('wall history',key,error))
                result[key]=error
            elif key=='metadata':
                # The existing protocol deliberately records CPU/GPU ownership in slot8.
                if not np.array_equal(np.delete(av,7),np.delete(bv,7)):raise AssertionError('wall sample/window/coverage metadata')
            elif av.tobytes()!=bv.tobytes():raise AssertionError(('wall history identity/extras',key))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--mpiexec',type=Path,required=True)
    p.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    p.add_argument('--ranks',type=int,choices=(1,2),default=1);p.add_argument('--axis',choices=('x','y','z'),default='x')
    p.add_argument('--images',action='store_true')
    p.add_argument('--transport',choices=('pinned','device-aware'),default='pinned',help='private postprocessing MPI transport')
    p.add_argument('--pipeline',choices=('standard-device','direct-device'),default='standard-device')
    p.add_argument('--implementation',type=Path,help='explicit ParaView Catalyst implementation for this native precision')
    p.add_argument('--automatic',action='store_true')
    p.add_argument('--reuse-completed-full',action='store_true',help='after an analysis-only correction, reuse completed references/full; run only unstarted prefix/tail/quiet')
    p.add_argument('--freeze-mpi-reference',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=a.reuse_completed_full)
    a.reference=a.reference.resolve()
    if a.images and a.implementation is None:raise ValueError('image tests require an explicit Catalyst implementation')
    if a.implementation is not None:a.implementation=a.implementation.resolve(strict=True)
    if a.automatic and not a.freeze_mpi_reference:raise ValueError('automatic clocks require explicitly freezing matching references')
    if a.transport!='pinned' and not a.images:raise ValueError('explicit device transport uses the existing device-rendering configuration')
    binary=out/'cpp_astr'
    if a.reuse_completed_full:
        if hashlib.sha256(binary.read_bytes()).digest()!=hashlib.sha256(a.binary.read_bytes()).digest():
            raise AssertionError('explicit reuse requires the same frozen executable')
    else:
        shutil.copy2(a.binary,binary);binary.chmod(0o555)
    args=arguments(out);args.executable=binary;args.mpiexec=str(a.mpiexec.resolve());args.unrestricted=True
    args.axis=a.axis;args.native_precision=a.precision;args.insitu_write_eps='0'
    args.scale_timestep=2e-5;args.maximum_cfl=.5
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    config=configuration(True).replace('statistics_window=.0005,.0115','statistics_window=.00001,.0002')
    config=config.replace('statistics=.true.,render=f',"statistics=.true.,render=f,products='channel_walls'")
    common=dict(grid='32,32,32',tgv_mapping='y-wavy',checkpoint_interval=2,insitu_config=config,
        gpu_precision_mode='' if a.precision=='fp32' else 'fp64',numerics=dict(filter_scheme='explicit10'),
        solver_halo_transport='pinned')
    def launch(backend,name,steps,**options):
        if a.automatic:
            options['cfl_variable_dt']=True
            options['run_config_text']=f"""&run_control max_steps={steps},reload_steps=1 /
&timestep mode='auto',cfl_target=.5,initial_dt=2e-5,max_dt=4e-5,growth_factor=1.1 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
"""
        return run_case(args,ROOT,backend,a.ranks,name,steps,**options)
    report=dict(precision=a.precision,ranks=a.ranks,axis=a.axis,transport=a.transport,images=a.images,pipeline=a.pipeline,
        automatic=a.automatic,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    references={}
    for impl in ('cpu','gpu'):
        if a.freeze_mpi_reference:
            args.executable=a.reference/'binaries'/a.precision
            if a.reuse_completed_full:
                reference=out/(impl+f'_np{a.ranks}_reference-'+impl)
                if 'The job is done!' not in (reference/'run.log').read_text():raise AssertionError('reused reference did not finish')
            else:
                reference,_=launch(impl,'reference-'+impl,4,product_catalog_records=definitions(),**common)
        else:
            if a.ranks!=1:raise ValueError('a new MPI reference must be explicitly frozen for the selected topology')
            reference=a.reference/(impl+'_np1_'+a.precision+('_csv' if impl=='gpu' else ''))
        references[impl]=reference.resolve()
    args.executable=binary
    records=definitions()
    if a.images:
        config=config.replace('render=f',f"render=t,derivative_backend='gpu',processing_backend='device',"
            f"postprocess_transport='{a.transport}',rendering_pipeline='{a.pipeline}',"
            f"implementation_path='{a.implementation}',pipeline_file='{ROOT/'scripts/insitu/device_render_pipeline.py'}'")
        common.update(insitu_config=config,postprocess_transport=a.transport)
        records+=image_definitions()
    if a.reuse_completed_full:
        full=out/f'gpu_np{a.ranks}_cpp-full'
        if 'The job is done!' not in (full/'run.log').read_text():raise AssertionError('reused main did not finish')
    else:
        full,_=launch('gpu','cpp-full',4,product_catalog_records=records,**common)
    report['physical']=check_state(full/'outdat/new/checkpoints/step000000000004/state.h5',a.precision)
    values=curves(full,images=a.images);report['curves']={};report['wall_history']={};report['flow']={}
    for impl,reference in references.items():
        report['curves'][impl]=compare_curves(values,curves(reference),a.precision)
        report['wall_history'][impl]=wall_history(full/'outdat/new/checkpoints/step000000000004/statistics.h5',
            reference/'outdat/new/checkpoints/step000000000004/statistics.h5',a.precision)
        report['flow'][impl]=compare(checkpoint(full/'outdat/new/checkpoints/step000000000004/state.h5'),
            checkpoint(reference/'outdat/new/checkpoints/step000000000004/state.h5'),None if a.precision=='fp32' else 2e-10)
    print('actual C++ wall curves and history agree with frozen references',flush=True)
    pictures=images(full,a.ranks) if a.images else {}
    prefix,_=launch('gpu','cpp-prefix',2,product_catalog_records=records,**common)
    resumed,_=launch('gpu','cpp-resumed',4,
        restore=prefix/'outdat/new/checkpoints/step000000000002',product_catalog_records=records,**common)
    cp=lambda case:case/'outdat/new/checkpoints/step000000000004'
    report['exact']={name:exact_state(cp(full)/name,cp(resumed)/name) for name in ('state.h5','statistics.h5')}
    for name in ('control.bin','insitu_control.bin'):
        if (cp(full)/name).read_bytes()!=(cp(resumed)/name).read_bytes():raise AssertionError('exact control differs: '+name)
    if archive_schedule_payload(cp(full)/'archives.bin')!=archive_schedule_payload(cp(resumed)/'archives.bin'):
        raise AssertionError('exact catalog scheduling differs')
    compare_curves(curves(resumed,True,images=a.images),values,a.precision,True)
    if a.images:
        again=images(resumed,a.ranks,True)
        if any(v.tobytes()!=pictures[k].tobytes() for k,v in again.items()):raise AssertionError('resumed chart pixels differ')
        report['image_frames']=len(pictures);report['resumed_image_frames']=len(again);report['resumed_pixels_exact']=True
    disabled=dict(common);disabled['insitu_config']=configuration(False)
    quiet,_=launch('gpu','cpp-no-statistics',4,**disabled)
    report['output_isolation']=exact_state(cp(full)/'state.h5',cp(quiet)/'state.h5')
    text=(full/'run.log').read_text()
    usage=re.findall(r'ASTR_INSITU_CPP_WALL_STATS rank=(\d+) staging_d2h_bytes=(\d+) staging_h2d_bytes=(\d+) '
        r'status_d2h_bytes=(\d+) field_download_bytes=(\d+)',text)
    if len(usage)!=5*a.ranks or any(int(r[4])!=0 for r in usage):raise AssertionError('wall-time sampling downloaded fields')
    if a.ranks==1 or a.transport=='device-aware':
        if any(int(r[1]) or int(r[2]) for r in usage):raise AssertionError('device/local wall-time has host MPI staging')
    report['wall_sampling_receipts']=[list(map(int,r)) for r in usage]
    report['steps']=logged_steps(full).tolist()
    report['references']={k:str(v) for k,v in references.items()}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('wall-time ordinary main: exact2+2, observation isolation and private transport complete')


if __name__=='__main__':main()
