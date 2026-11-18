"""Completed-step native line/probe histories, Welch spectra and exact restart."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
from PIL import Image
from scipy.signal import welch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments
from test_insitu_temporal_catalog import definition
from run_output_restart_validation import run_case
from run_cpp_insitu_gate import compare_hdf
from run_cpp_selected_gate import configuration
from run_cpp_tgv_gate import logged_steps
from test_insitu_device_products import resident_configuration

spec=importlib.util.spec_from_file_location('temporal_archive',ROOT/'scripts/insitu/temporal_archive.py')
archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)


def records(selected):
    line=definition()
    line['fields']=['density','velocity_x','pressure']
    line['sampling'].update(index=8,index_ranges=dict(i=[0,16],k=[4,4]))
    line['temporal']['block_samples']=4;line['schedule']['interval']=3
    probe=definition('probe_spectrum');probe['sampling']['indices']=[8,8,4]
    probe['temporal']=dict(segment_length=8,archive_samples=True,block_samples=4)
    probe['schedule']['interval']=3
    result=[line,probe]
    if selected:
        for source in ('reynolds','favre'):
            for fluctuation in (False,True):
                item=deepcopy(line);item.update(id=source+('-fluct' if fluctuation else '-mean'),
                    source=source,statistics_source='selected-time',fluctuation=fluctuation)
                item['fields']=['velocity_x','pressure'] if source=='reynolds' else ['velocity_x']
                result.append(item)
    return result


def history(cases,name):
    paths=[p for case in cases for p in (case/'outdat/new/temporal_products'/name).glob('*/segment*/block.*.csv')]
    groups=archive.read_histories(paths)
    if len(groups)!=1:
        raise AssertionError(('temporal history identity',name,groups.keys()))
    return next(iter(groups.values()))


def difference(x,y,exact=False):
    if x.shape!=y.shape:
        raise AssertionError(('temporal sample shapes',x.shape,y.shape))
    if exact:
        if x.dtype!=y.dtype or x.tobytes()!=y.tobytes():
            raise AssertionError('same-backend temporal values changed')
        return 0.
    x,y=x.astype(float),y.astype(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError('nonfinite temporal values')
    error=float(np.max(abs(x-y),initial=0)/max(1.,float(np.max(abs(y),initial=0))))
    if error>2e-10:
        raise AssertionError(('temporal scaled error',error))
    return error


def compare_history(x,y,exact=False):
    if x.dtype.names!=y.dtype.names:
        raise AssertionError('temporal column names differ')
    return max(difference(x[field],y[field],exact) for field in x.dtype.names)


def spectra(case):
    result={}
    for path in (case/'outdat/new/temporal_products/probe_spectrum').glob('*/segment*/spectrum.*.csv'):
        text=path.read_text();lines=[line for line in text.splitlines() if not line.startswith('#')]
        data=np.loadtxt(lines[1:],delimiter=',',ndmin=2) if len(lines)>1 else np.empty((0,len(lines[0].split(','))))
        result[path.name]=(text,data)
    return result


def inspect(case,items,automatic):
    status=json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=len(items) or any(
        p['state']!=('waiting-for-statistics' if automatic and p['id']=='probe_spectrum' else 'active')
        for p in status['products']):
        raise AssertionError(('temporal admission',status))
    rows={item['id']:history([case],item['id']) for item in items}
    line,probe=rows['time_space'],rows['probe_spectrum']
    if len(line)!=13*17 or len(probe)!=13:
        raise AssertionError('temporal sample cadence must not equal publication cadence')
    for name,table in rows.items():
        if np.any(table['gap_before']):
            raise AssertionError('unexpected gap in continuous temporal history')
        if not np.array_equal(np.unique(table['step']),np.arange(0 if name in ('time_space','probe_spectrum') else 1,13)):
            raise AssertionError('missing complete-step sample')
    clock=logged_steps(case,12)
    difference(probe['time'][1:],clock[:,2])
    if automatic and np.ptp(np.diff(probe['time']))<=1e-5:
        raise AssertionError('automatic fixture did not vary timestamps')
    with h5py.File(case/'outdat/new/checkpoints/step000000000012/state.h5') as state:
        rho,mx,my,mz,e=[state[f'q{c:04d}'][4,8,:] for c in range(1,6)]
    last=line[line['step']==12]
    errors=[difference(last['velocity_x'],mx/rho),difference(last['pressure'],.4*(e-.5*(mx*mx+my*my+mz*mz)/rho))]
    errors.append(difference(probe['pressure'][-1:],last['pressure'][8:9]))
    density=line['density'].reshape(13,17)
    for item in items[2:]:
        for step in range(1,13):
            times=probe['time'][:step+1];weights=np.zeros(step+1)
            weights[:-1]+=.5*np.diff(times);weights[1:]+=.5*np.diff(times)
            sample=rows[item['id']][rows[item['id']]['step']==step]
            mass=weights@density[:step+1]
            for field in item['fields']:
                primitive=line[field].reshape(13,17)
                mean=(weights@primitive[:step+1]/weights.sum() if item['source']=='reynolds'
                    else weights@(density[:step+1]*primitive[:step+1])/mass)
                expected=primitive[step]-mean if item['fluctuation'] else mean
                errors.append(difference(sample[field],expected))
    output=spectra(case)
    if len(output)!=5:
        raise AssertionError('spectrum publication cadence differs')
    if automatic:
        if any('# available=F;' not in text or 'valid_segments=0;' not in text or len(values) for text,values in output.values()):
            raise AssertionError('variable-step Welch spectrum must remain explicitly unavailable')
    else:
        text,values=output[sorted(output)[-1]]
        if '# available=T' not in text or 'valid_segments=2' not in text:
            raise AssertionError('fixed-step Welch coverage')
        window=.5*(1-np.cos(2*np.pi*np.arange(8)/8))
        frequency,power=welch(np.column_stack([probe['velocity_x'],probe['pressure']]),fs=1000.,
            window=window,nperseg=8,noverlap=4,detrend='constant',scaling='density',axis=0)
        errors.append(difference(values,np.column_stack([frequency,power])))
    packets=re.findall(r'ASTR_TEMPORAL_SAMPLE rank=\d+ id=(\S+) step=\d+ one_dimensional_d2h_bytes=(\d+)',
        (case/'run.log').read_text())
    transfers={name:sum(int(count) for field,count in packets if field==name) for name in rows}
    for item in items:
        name=item['id'];nodes=1 if item['kind']=='probe_spectrum' else 17
        if transfers[name]!=nodes*len(item['fields'])*8*(13 if item['source']=='instantaneous' else 12):
            raise AssertionError(('temporal selected payload bytes',name,transfers[name]))
    return rows,dict(independent_scaled=max(errors),one_dimensional_d2h_bytes=transfers)


def run_control(steps):
    return f"""&run_control max_steps={steps},reload_steps=1,stop_time=-1 /
&timestep mode='auto',cfl_target=.5,initial_dt=.001,max_dt=.002,growth_factor=1.1 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f,sample_clock='steps',sample_steps=1,write_steps=0 /
"""


def render_gate(opt,args,output):
    line=definition('time_space',True)
    line['sampling'].update(index=8,index_ranges=dict(i=[0,16],k=[4,4]))
    line['temporal']['block_samples']=4;line['schedule']['interval']=3
    line['display']['color_range']=[71.4,71.6]
    probe=definition('probe_spectrum',True);probe['sampling']['indices']=[8,8,4]
    probe['temporal']['segment_length']=8;probe['schedule']['interval']=3
    probe['display']['y_range']=[0.,4e-17]
    config=resident_configuration('standard-device',profile='catalog_stats').replace(
        'statistics_window=0.0005,0.0115','statistics_window=0.,1.')
    unlimited=np.iinfo(np.int64).max//4096
    config=config.replace('host_budget_bytes=4294967296',f'host_budget_bytes={unlimited}').replace(
        'device_budget_bytes=2147483648',f'device_budget_bytes={unlimited}').replace(
        'device_reserve_bytes=1073741824','device_reserve_bytes=0')
    common=dict(grid='16,16,16',insitu_config=config,checkpoint_interval=5,
        numerics=dict(filter_scheme='explicit10'),insitu_timing=True)
    def run(name,steps=12,restore=None,items=(line,probe)):
        return run_case(args,ROOT,'gpu',opt.ranks,name,steps,restore=restore,
            product_catalog_records=list(items),**common)[0]
    def pictures(case,tail=False):
        result={}
        directories={p.parent.parent.name:p for p in (case/'outdat/new/render_products').glob('*/*/segment*')}
        if set(directories)!={'time_space','probe_spectrum'}:
            raise AssertionError(('temporal image products',directories))
        status=json.loads((case/'outdat/new/products.status.json').read_text())
        if any(p['state']!='active' for p in status['products']):
            raise AssertionError(status)
        for name,directory in directories.items():
            steps=([6,9,12] if tail else [0,3,6,9,12]) if name=='time_space' else [9,12]
            files=sorted(directory.glob('frame.step*.jpeg'))
            if [int(p.stem.split('step')[1]) for p in files]!=steps or list(directory.glob('*.eps')):
                raise AssertionError(('temporal image schedule/EPS',name,files))
            for path in files:
                receipt=json.loads(path.with_name(path.stem+'.rank00000000.json').read_text())
                if receipt['chart_backend']!='existing-host-matplotlib' or receipt['nodes']>17 or receipt['rows']>4:
                    raise AssertionError(('temporal chart used a volume channel',receipt))
                pixels=np.asarray(Image.open(path).convert('RGB'))
                if pixels.shape!=(600,800,3) or np.ptp(pixels)<100:
                    raise AssertionError(('temporal chart raster',path))
                result[(name,path.name)]=pixels
        return result
    actual=run('cpp-temporal-images');images=pictures(actual)
    final='outdat/new/checkpoints/step000000000012'
    off=run('cpp-no-products',items=())
    compare_hdf(actual/final/'state.h5',off/final/'state.h5',exact=True)
    args.executable=opt.gpu_reference.resolve();old=run('old-gpu-temporal-images');reference=pictures(old)
    errors={name+'/'+path:float(np.mean(abs(values.astype(float)-reference[name,path])))
        for (name,path),values in images.items()}
    args.executable=opt.binary.resolve();prefix=run('cpp-image-prefix',5)
    resumed=run('cpp-image-tail',restore=prefix/'outdat/new/checkpoints/step000000000005',items=())
    for filename in ('state.h5','statistics.h5'):
        compare_hdf(actual/final/filename,resumed/final/filename,exact=True)
    for key,values in pictures(resumed,tail=True).items():
        if not np.array_equal(values,images[key]):
            raise AssertionError(('temporal image changed after continuation',key))
    report=dict(ranks=opt.ranks,axis=opt.axis,chart_backend='existing-host-matplotlib',
        old_new_pixel_mean_absolute_difference=errors,on_off_state_bits=True,
        exact_restart=dict(state_bits=True,statistics_bits=True,pixels_equal=True))
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser()
    for name in ('binary','cpu-reference','gpu-reference','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,default=2)
    parser.add_argument('--axis',choices=['x','y','z'],default='x')
    parser.add_argument('--automatic',action='store_true')
    parser.add_argument('--selected',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--images',action='store_true')
    opt=parser.parse_args();output=opt.output.resolve();output.mkdir(parents=True,exist_ok=False)
    args=arguments(output);args.executable=opt.binary.resolve();args.axis=opt.axis
    args.unrestricted=True;args.insitu_write_eps='0'
    if opt.images:
        render_gate(opt,args,output);return
    items=records(opt.selected)
    config=configuration().replace('statistics_window=.0005,.0115','statistics_window=0.,1.')
    common=dict(grid='16,16,16',insitu_config=config,product_catalog_records=items,
        checkpoint_interval=5,numerics=dict(filter_scheme='explicit10'),insitu_timing=True)
    def launch(name,steps=12,restore=None,**extra):
        kwargs=dict(common,run_config_text=run_control(steps) if opt.automatic else None,**extra)
        return run_case(args,ROOT,'gpu',opt.ranks,name,steps,restore=restore,**kwargs)[0]
    current=launch('cpp-temporal',memcheck=opt.memcheck)
    actual,report=inspect(current,items,opt.automatic)
    report.update(topology=opt.axis,ranks=opt.ranks,automatic=opt.automatic,selected=opt.selected,references={})
    final='outdat/new/checkpoints/step000000000012'
    quiet=run_case(args,ROOT,'gpu',opt.ranks,'cpp-off',12,**dict(common,product_catalog_records=[],
        run_config_text=run_control(12) if opt.automatic else None))[0]
    compare_hdf(current/final/'state.h5',quiet/final/'state.h5',exact=True)
    if not np.array_equal(logged_steps(current,12),logged_steps(quiet,12)):
        raise AssertionError('temporal observation changed timestep control')
    report['on_off_state_and_clock_bits']=True
    for name,binary,backend in [('cpu',opt.cpu_reference,'cpu'),('old_gpu',opt.gpu_reference,'gpu')]:
        args.executable=binary.resolve()
        reference=run_case(args,ROOT,backend,opt.ranks,name,12,**dict(common,
            run_config_text=run_control(12) if opt.automatic else None))[0]
        expected={item['id']:history([reference],item['id']) for item in items}
        report['references'][name]=dict(state_scaled=compare_hdf(current/final/'state.h5',reference/final/'state.h5'),
            history_scaled={key:compare_history(actual[key],expected[key]) for key in actual})
    args.executable=opt.binary.resolve();prefix=launch('cpp-prefix',5)
    tail=launch('cpp-tail',restore=prefix/'outdat/new/checkpoints/step000000000005')
    for filename in ('state.h5','statistics.h5'):
        compare_hdf(current/final/filename,tail/final/filename,exact=True)
    for name in actual:
        compare_history(history([prefix,tail],name),actual[name],exact=True)
    for name,(text,values) in spectra(tail).items():
        reference_text,reference_values=spectra(current)[name]
        if text!=reference_text:
            raise AssertionError('resumed spectrum CSV/metadata differs')
        difference(values,reference_values,exact=True)
    report['exact_restart']=dict(state_bits=True,statistics_bits=True,history_values_bits=True,spectrum_csv_bytes=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
