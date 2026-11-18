"""Freeze actual CUDA Fortran MP admissibility/budget results and source identities."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--normal-range',action='store_true',help='FP32: retain the subnormal case in its separate diagnostic, not this cross-compiler gate')
    args=parser.parse_args()
    if args.normal_range and args.precision!='fp32':parser.error('--normal-range is specific to the approved FP32 comparison')
    build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    source=root/'tests/gpu_validation/mp_positivity_reference_probe.cuf'
    shutil.copy2(source,out/source.name)
    shutil.copy2(__file__,out/'preparation.py')
    shutil.copy2(build/'bin/astr',out/'original_astr')
    for relative in ['src_gpu/mp_positivity_gpu.cuf','src_gpu/solver_gpu.cuf']:
        shutil.copy2(root/relative,out/Path(relative).name)
    compile_command=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpif90',
        '-Mpreprocess','-Mextend','-cuda','-gpu=rdc','-O2','-I'+str(build/'src'),
        '-I/opt/hdf5-1.14.6/include','-c',str(out/source.name),'-o',str(out/'probe.o')]
    if args.precision=='fp32':compile_command.insert(1,'-DASTR_REAL32')
    with (out/'compile.log').open('w') as log:
        subprocess.run(compile_command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    command=[]
    for argument in shlex.split((build/'src/CMakeFiles/astr.dir/link.txt').read_text()):
        command.extend(shlex.split((build/'src'/argument[1:]).read_text()) if argument.startswith('@') else [argument])
    matches=[i for i,word in enumerate(command) if word.endswith('/astr.F90.o')]
    assert len(matches)==1
    command[matches[0]]=str(out/'probe.o');command[command.index('-o')+1]=str(out/'probe')
    with (out/'link.log').open('w') as log:
        subprocess.run(command,cwd=build/'src',stdout=log,stderr=subprocess.STDOUT,check=True)
    cases=[
        ('unchanged',[1,0,0,0,1],[0,0,0,0,0],1,1.),
        ('density_bound',[1,0,0,0,1],[-2,0,0,0,0],1,.5),
        ('energy_bound',[1,0,0,0,1],[0,0,0,0,-2],1,.5),
        ('momentum_bound',[1,0,0,0,1],[0,2,0,0,0],1,np.sqrt(.5)),
        ('negative_density',[-1,0,0,0,1],[0,0,0,0,0],0,0.),
        ('zero_density',[0,0,0,0,1],[0,0,0,0,0],0,0.),
        ('nan_input',[1,np.nan,0,0,1],[0,0,0,0,0],0,0.),
        ('infinite_input',[1,0,0,0,np.inf],[0,0,0,0,0],0,0.),
        ('squared_velocity_overflow',[1,1e200,0,0,1e300],[0,0,0,0,0],0,0.),
        ('large_density_finite_kinetic',[1e300,1e300,0,0,1e300],[0,0,0,0,0],1,1.),
        ('velocity_division_overflow',[1e-300,1e300,0,0,1],[0,0,0,0,0],0,0.),
        ('positive_subnormal_density',[1e-310,0,0,0,1e-310],[0,0,0,0,0],1,1.),
        ('kinetic_equality',[1,2,0,0,2],[0,0,0,0,0],0,0.),
        ('three_component_kinetic',[2,2,4,6,15],[0,0,0,0,0],1,1.),
    ]
    rng=np.random.default_rng(742643)
    for i in range(256):
        density=10.**rng.uniform(-120,120)
        velocity=rng.uniform(-4,4,3)
        internal=density*10.**rng.uniform(-4,2)
        base=np.r_[density,density*velocity,internal+.5*density*np.sum(velocity**2)]
        correction=rng.uniform(-3,3,5)*np.maximum(abs(base),density)
        cases.append((f'finite_{i:03d}',base,correction,1,None))
    for name,base,correction,_,_ in list(cases[14:]):
        scale=base[0]
        cases.append((name+'_rho1',base/scale,correction/scale,1,None))
    if args.precision=='fp32':
        # The normalized frozen family is representable in FP32; retain the
        # extreme-range semantics using values from that precision's range.
        cases=cases[:14]+cases[270:]
        replacements={
            8:([1,1e30,0,0,1e35],[0,0,0,0,0]),
            9:([1e30,1e30,0,0,1e30],[0,0,0,0,0]),
            10:([1e-30,1e30,0,0,1],[0,0,0,0,0]),
            11:([1e-40,0,0,0,1e-40],[0,0,0,0,0])}
        for i,(q,d) in replacements.items():
            name,_,_,valid,analytic=cases[i];cases[i]=(name,q,d,valid,analytic)
    excluded=[]
    if args.normal_range:
        excluded=[row[0] for row in cases if row[0]=='positive_subnormal_density']
        assert excluded==['positive_subnormal_density']
        cases=[row for row in cases if row[0]!='positive_subnormal_density']
    dtype=np.dtype('<f4' if args.precision=='fp32' else '<f8');width=dtype.itemsize
    base=np.array([row[1] for row in cases],dtype=dtype)
    correction=np.array([row[2] for row in cases],dtype=dtype)
    with (out/'input.bin').open('wb') as stream:
        np.array([len(cases),width],dtype='<i4').tofile(stream)
        base.tofile(stream);correction.tofile(stream)
    with (out/'run.log').open('w') as log:
        subprocess.run([str(out/'probe'),str(out/'input.bin'),str(out/'reference.bin')],cwd=out,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'reference.bin').open('rb') as stream:
        header=np.fromfile(stream,dtype='<i4',count=2)
        assert header.tolist()==[len(cases),width]
        valid=np.fromfile(stream,dtype='<i4',count=len(cases))
        theta=np.fromfile(stream,dtype=dtype,count=len(cases))
        limited_valid=np.fromfile(stream,dtype='<i4',count=len(cases))
        assert stream.read()==b''
    assert np.array_equal(valid,np.array([row[3] for row in cases]))
    assert np.isfinite(theta).all() and np.all((theta>=0)&(theta<=1))
    analytic={}
    for i,(name,_,_,admissible,root_value) in enumerate(cases):
        if root_value is None:continue
        error=abs(float(theta[i])-root_value)
        if args.precision=='fp64':assert error<=2e-10,(name,theta[i],root_value)
        analytic[name]=dict(valid=int(valid[i]),theta=float(theta[i]),analytic_boundary=float(root_value),abs_error=float(error))
    files=['original_astr','probe','mp_positivity_gpu.cuf','solver_gpu.cuf','input.bin','reference.bin']
    failures=np.flatnonzero((valid==1)&(limited_valid!=1))
    report=dict(precision=args.precision,real_bytes=width,cases=len(cases),excluded_semantic_diagnostics=excluded,
        admissible_count=int(valid.sum()),limited_admissible=int(np.count_nonzero((valid==1)&(theta<1))),
        returned_budget_inadmissible=[dict(index=int(i),name=cases[i][0],base=base[i].tolist(),
            correction=correction[i].tolist(),theta=float(theta[i])) for i in failures],
        analytic=analytic,compile=compile_command,link=command,
        sha256={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in files})
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    for p in out.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
    out.chmod(0o555)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
