"""Actual catalog add/remove/re-enable, sampled gaps and selected restart."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments,product,segments,frames
from run_output_restart_validation import run_case
from run_cpp_insitu_gate import compare_hdf
from run_cpp_selected_gate import configuration,definitions


def all_frames(case):
    result={}
    for segment in segments(case):
        name=segment.parent.parent.name
        table=result.setdefault(name,{})
        for path in frames(segment):
            step=int(path.stem[4:])
            if step in table:
                raise AssertionError(('repeated output identity',name,step))
            table[step]=path
    return result


def compare(current,reference,tail=False):
    a,b=all_frames(current),all_frames(reference)
    if set(a)!=set(b):
        raise AssertionError(('lifecycle product ids',set(a),set(b)))
    errors={}
    for name in a:
        expected={s for s in b[name] if s>9} if tail else set(b[name])
        if set(a[name])!=expected:
            raise AssertionError(('lifecycle frame steps',name,set(a[name]),expected))
        errors[name]=max(compare_hdf(path,b[name][s],exact=tail) for s,path in a[name].items())
    return errors


def independent_late_integral(case):
    table=all_frames(case)
    if set(table['late'])!={4,5,6,9,10,11,12}:
        raise AssertionError(('late product activation/coverage steps',set(table['late'])))
    raw=table['raw'];late=table['late'][12]
    with h5py.File(raw[0].parent/'geometry.h5') as whole,h5py.File(late.parent/'geometry.h5') as part:
        if not np.array_equal(whole['coordinates'][3:11,2:12],part['coordinates'][...]):
            raise AssertionError('late ROI is not the requested physical subplane')
    w=np.zeros(13)
    # The absent interval (6,9) has no samples or time weight.
    for lo,hi in ((3,6),(9,12)):
        for s in range(lo+1,hi+1):
            t0,t1=(s-1)*.001,s*.001
            a,b=max(t0,.0005),min(t1,.0115)
            if b<=a:
                continue
            right=.5*((a-t0)/.001)+.5*((b-t0)/.001)
            w[s-1]+=(b-a)*(1-right);w[s]+=(b-a)*right
    error=0.
    with h5py.File(late) as data:
        measure=str(data.attrs['source_measure'])
        if 'sampling_segments=2' not in measure:
            raise AssertionError(('gapped history segment identity',measure))
        for field in ('pressure','temperature'):
            samples=[]
            for s in range(13):
                with h5py.File(raw[s]) as f:
                    samples.append(f[field][3:11,2:12])
            expected=np.tensordot(w,np.stack(samples),axes=1)/w.sum()
            observed=data[field][...]
            error=max(error,float(np.max(abs(observed-expected))/max(1.,float(np.max(abs(expected))))))
    if error>2e-10 or abs(w.sum()-.0055)>2e-18:
        raise AssertionError(('gapped integral',error,w.sum()))
    return dict(scaled=error,duration=float(w.sum()),source_measure=measure)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--cpu-reference',type=Path,required=True)
    parser.add_argument('--gpu-reference',type=Path,required=True)
    parser.add_argument('--replay-library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    opt=parser.parse_args()
    output=opt.output.resolve();output.mkdir(parents=True,exist_ok=False)
    args=arguments(output);args.executable=opt.binary.resolve();args.axis='y'
    args.unrestricted=True;args.insitu_write_eps='0'
    original=definitions(False)
    late=product('late',['pressure','temperature'],'reynolds')
    late['statistics_source']='selected-time';late['schedule']['interval']=1
    late['sampling']['index_ranges']=dict(i=[2,11],j=[3,10])
    added=[*original,late]
    renewed=deepcopy(late);renewed['enable_request']=2
    final_records=[*original,renewed]
    publications={4:added,8:original,10:final_records}
    common=dict(grid='16,16,16',checkpoint_interval=9,product_poll_steps=1,numerics=dict(filter_scheme='explicit10'))
    quiet,_=run_case(args,ROOT,'gpu',2,'cpp-off',12,insitu_config=configuration(False),
        product_catalog_records=[],**common)
    current,_=run_case(args,ROOT,'gpu',2,'cpp-lifecycle',12,insitu_config=configuration(),
        product_catalog_records=original,catalog_replay=(opt.replay_library,publications),**common)
    final='outdat/new/checkpoints/step000000000012'
    compare_hdf(current/final/'state.h5',quiet/final/'state.h5',exact=True)
    report=dict(topology='1,2,1',on_off_state_bits=True,gapped_integral=independent_late_integral(current),references={})
    for name,binary,backend in [('cpu',opt.cpu_reference,'cpu'),('old_gpu',opt.gpu_reference,'gpu')]:
        args.executable=binary.resolve()
        reference,_=run_case(args,ROOT,backend,2,name,12,insitu_config=configuration(),
            product_catalog_records=original,catalog_replay=(opt.replay_library,publications),**common)
        report['references'][name]=dict(state_scaled=compare_hdf(current/final/'state.h5',reference/final/'state.h5'),
                                      fields=compare(current,reference))
    args.executable=opt.binary.resolve()
    prefix,_=run_case(args,ROOT,'gpu',2,'cpp-prefix',9,insitu_config=configuration(),
        product_catalog_records=original,catalog_replay=(opt.replay_library,publications),**common)
    resumed,_=run_case(args,ROOT,'gpu',2,'cpp-resumed',12,insitu_config=configuration(),
        product_catalog_records=[],restore=prefix/'outdat/new/checkpoints/step000000000009',**common)
    for name in ('state.h5','statistics.h5'):
        compare_hdf(current/final/name,resumed/final/name,exact=True)
    report['same_backend_restart']=dict(state_bits=True,statistics_bits=True,fields=compare(resumed,current,tail=True))
    status=json.loads((resumed/'outdat/new/products.status.json').read_text())
    late_state=next(p for p in status['products'] if p['id']=='late')
    if late_state['state']!='active' or late_state['consumed_enable_request']!=2 or late_state['last_output_step']!=12:
        raise AssertionError(late_state)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
