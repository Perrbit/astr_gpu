"""Actual existing CPU/CUDA-Fortran wall curves and wall-time accumulators."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_index_slice_runtime import arguments
from test_runtime_catalog_walls import wall_product,wall_frames,rows
from run_output_restart_validation import run_case
from run_cpp_statistics_gate import configuration
from run_cpp_tgv_gate import logged_steps
from run_native_precision_restart_gate import check_state


def definitions():
    records=[]
    for side in ('lower','upper'):
        for field in ('wall_pressure','wall_shear_x','wall_heat_into_gas'):
            for source in ('instantaneous','reynolds'):
                item=wall_product(side+'-'+field+'-'+source,field,side,4)
                # A whole-wall ROI includes the rounded native endpoint, not just exact2pi.
                item['sampling']['bounds']=[-1.,2*np.pi+1.]*3
                item['schedule']['interval']=2;item['source']=source
                if source=='reynolds':item['statistics_source']='wall-time'
                records.append(item)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original64',type=Path,required=True);p.add_argument('--original32',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-cpu64',action='store_true',help='reuse completed CPU64 calculation after correcting analysis only')
    a=p.parse_args();out=a.output.resolve()
    if not a.reuse_cpu64:
        out.mkdir(parents=True,exist_ok=False);(out/'binaries').mkdir()
    identity={};records=definitions()
    for precision,source in (('fp64',a.original64),('fp32',a.original32)):
        target=out/'binaries'/precision
        if not a.reuse_cpu64:
            shutil.copy2(source,target);target.chmod(0o555)
        identity[precision]=dict(source=str(source.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    if not a.reuse_cpu64:(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    shutil.copy2(__file__,out/('analysis_runner.py' if a.reuse_cpu64 else 'runner.py'))
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    report={}
    for precision in ('fp64','fp32'):
        cases={};data={}
        for backend in ('cpu','gpu'):
            args=arguments(out);args.executable=out/'binaries'/precision
            args.unrestricted=True;args.native_precision=precision;args.insitu_write_eps='0'
            args.scale_timestep=2e-5;args.maximum_cfl=.5
            config=configuration(True).replace('statistics_window=.0005,.0115','statistics_window=.00001,.0002')
            config=config.replace('statistics=.true.,render=f',"statistics=.true.,render=f,products='channel_walls'")
            # CSV references do not request the device-rendering startup mode.
            # The existing GPU numerical wall producer still runs on GPU for use_gpu=t.
            if a.reuse_cpu64 and precision=='fp64' and backend=='cpu':
                case=out/'cpu_np1_fp64'
            else:
                case,_=run_case(args,ROOT,backend,1,precision+('_csv' if backend=='gpu' else ''),4,grid='32,32,32',tgv_mapping='y-wavy',
                    checkpoint_interval=2,insitu_config=config,product_catalog_records=records,
                    gpu_precision_mode='' if precision=='fp32' else 'fp64',
                    numerics=dict(filter_scheme='explicit10'),rhs_snapshot_step=3)
            physical=check_state(case/'outdat/new/checkpoints/step000000000004/state.h5',precision)
            status=json.loads((case/'outdat/new/products.status.json').read_text())
            if len(status['products'])!=12 or any(r['state']!='active' for r in status['products']):raise AssertionError(status)
            values={}
            for record in records:
                files=wall_frames(case,record['id']);expected=[0,2,4] if record['source']=='instantaneous' else [1,3]
                if [int(f.stem.split('step')[1]) for f in files]!=expected:raise AssertionError('wall frame step identities')
                for f in files:
                    table=rows(f);array=np.asarray([[float(v) for v in row.values()] for row in table])
                    if list(table[0])!=['step','time','global_i','x','y','z','value'] or array.shape!=(33,7) or not np.isfinite(array).all():
                        raise AssertionError('wall CSV shape/finiteness')
                    values[record['id']+'/'+f.name]=array
            cases[backend]=dict(directory=str(case),physical=physical,steps=logged_steps(case).tolist());data[backend]=values
            print(precision+' '+backend+':12 legacy wall products active',flush=True)
        if data['cpu'].keys()!=data['gpu'].keys():raise AssertionError('CPU/GPU wall identities')
        if any(not np.array_equal(data['cpu'][key][:,:3],data['gpu'][key][:,:3]) for key in data['cpu']):
            raise AssertionError('wall row step/time/global_i identities')
        differences={key:np.max(abs(data['cpu'][key][:,3:]-data['gpu'][key][:,3:]),axis=0)/np.maximum(1.,np.max(abs(data['cpu'][key][:,3:]),axis=0))
                     for key in data['cpu']}
        worst=max(float(v.max()) for v in differences.values())
        if precision=='fp64' and worst>2e-10:raise AssertionError(('original FP64 wall field difference',worst))
        report[precision]=dict(cases=cases,columns=['x','y','z','quantity'],
            CPU_GPU_scaled={key:v.tolist() for key,v in differences.items()},maximum_scaled=worst)
        (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print('both precision reference pairs complete')


if __name__=='__main__':main()
