"""Physical q/carry diagnostic for the original single-rank periodic fast branch."""
import argparse, hashlib, json, shutil, subprocess
from fractions import Fraction
from pathlib import Path
import numpy as np
from run_air5_flux_components import extract
from run_air5_properties_components import read
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiler',type=Path,required=True)
    p.add_argument('--properties-reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=['src/chemistry_compensation.F90','src_gpu/qswap_gpu.cuf','src_gpu/halo_exchange_gpu.cuf',
           'tests/gpu_backend/air5_periodic_compensation_reference.cuf','tests/gpu_backend/probe_air5_periodic_compensation.py']
    hashes={}
    for name in files:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        hashes[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    source=out/'source';q=(source/'src_gpu/qswap_gpu.cuf').read_text();c=(source/'src/chemistry_compensation.F90').read_text()
    (out/'frozen_periodic_kernels.inc').write_text('\n'.join(extract(q,'qswap_'+axis+'_kernel') for axis in 'xyz'))
    (out/'frozen_compensation_math.inc').write_text('\n'.join(extract(c,name) for name in ['compensated_mean','compensated_normalize']))
    body=extract((source/'src_gpu/halo_exchange_gpu.cuf').read_text(),'exchange_solution_halo_gpu')
    (out/'frozen_controller.inc').write_text(body)
    start=body.index('    if(isize==1 .and. jsize==1 .and. ksize==1')
    end=body.index('\n    endif',start)+len('\n    endif')
    fast=body[start:end];assert 'call qswap_local_x_gpu' in fast and 'call compensation_gpu_axis' not in fast and '\n      return' in fast
    status,properties=read(a.properties_reference.resolve()/'cpu_all.bin');base=properties[120,:11]
    assert np.all(status[120]==0) and np.all(base[5:10]>0);base.astype('<f8').tofile(out/'input.bin')
    cmd=[a.compiler.resolve(),'-O2','-cuda','-gpu=rdc,maxregcount:128','-Mpreprocess','-Mextend','-I'+str(out),
         source/'tests/gpu_backend/air5_periodic_compensation_reference.cuf','-o',out/'probe']
    with (out/'build.log').open('w') as f:subprocess.run(list(map(str,cmd)),cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (out/'run.log').open('w') as f:subprocess.run([out/'probe',out/'input.bin',out/'output.bin'],cwd=out,stdout=f,stderr=subprocess.STDOUT,check=True)
    values=np.fromfile(out/'output.bin','<f8').reshape(11,6)
    assert np.array_equal(values[:,:2],np.broadcast_to(base[:,None],(11,2)))
    assert np.array_equal(values[:,2],np.ldexp(base,-60)) and np.array_equal(values[:,3],-np.ldexp(base,-60))
    assert np.all(values[:,5]==0.) and np.array_equal(values[:,4],base)
    assert np.all(abs(values[:,2])<=.5*abs(np.spacing(base)))
    represented_left=Fraction.from_float(float(values[0,0]))-Fraction.from_float(float(values[0,2]))
    represented_right=Fraction.from_float(float(values[0,1]))-Fraction.from_float(float(values[0,3]))
    report=dict(source_sha256=hashes,command=list(map(str,cmd)),physical_q_unchanged=True,
                normalized_nonzero_carry_unchanged=True,reference_mean_carry_zero=True,
                rho_left_carry=float(values[0,2]),rho_right_carry=float(values[0,3]),
                exact_represented_rho_difference=str(represented_right-represented_left),
                fast_branch=fast,scope='three original periodic kernels plus controller audit; not full ordinary flow or a repaired implementation')
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','command','fast_branch']},indent=2))


if __name__=='__main__':main()
