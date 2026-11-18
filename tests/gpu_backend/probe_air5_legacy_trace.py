"""Read-only numerical diagnosis of existing CPU/GPU legacy-filter references."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from run_air5_flux_components import extract

ROOT=Path(__file__).resolve().parents[2]
FILES=['src/astr_precision.F90','src/constdef.F90','src/chemistry_air5_data.F90','src/chemistry_core.F90',
       'src/chemistry_properties.F90','src/chemistry_pure_vt_body.inc','src/chemistry_vibrational_difference_body.inc',
       'src_gpu/chemistry_core_gpu.cuf','src_gpu/chemistry_solver_gpu.cuf','src_gpu/solver_gpu.cuf',
       'src_gpu/chemistry_compensation_gpu.cuf','tests/gpu_backend/air5_legacy_filter_trace_probe.cuf',
       'tests/gpu_backend/probe_air5_legacy_trace.py']

def run(cmd,work,log):
    with (work/log).open('w') as f:subprocess.run(list(map(str,cmd)),cwd=work,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compiler',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--expect-repaired',action='store_true')
    p.add_argument('--repair-input',type=Path)
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);hashes={}
    for name in FILES:
        dst=out/'source'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst);hashes[name]=hashlib.sha256(dst.read_bytes()).hexdigest()
    source=out/'source';work=out/'build';work.mkdir()
    core=(source/'src_gpu/chemistry_solver_gpu.cuf').read_text()
    functions=['record_air5_status','air5_interior_ratio_gpu','air5_vibrational_excess_gpu','air5_translational_margin_gpu',
               'air5_state_is_admissible_gpu','air5_limit_filtered_state_kernel']
    (work/'frozen_legacy_functions.inc').write_text('\n'.join(extract(core,name) for name in functions)+'\n'+extract((source/'src_gpu/solver_gpu.cuf').read_text(),'filter10_value'))
    (work/'frozen_projection.inc').write_text(extract((source/'src_gpu/chemistry_compensation_gpu.cuf').read_text(),'compensated_filter_projection_gpu'))
    flags=[str(args.compiler),'-O2','-Mpreprocess','-Mextend','-cuda','-gpu=rdc,maxregcount:128','-D_CUDA','-I'+str(source/'src'),'-I'+str(work)]
    inputs=['src/astr_precision.F90','src/constdef.F90','src/chemistry_air5_data.F90','src/chemistry_core.F90',
            'src/chemistry_properties.F90','src_gpu/chemistry_core_gpu.cuf','tests/gpu_backend/air5_legacy_filter_trace_probe.cuf']
    commands=[];objects=[]
    for name in inputs:
        obj=work/(Path(name).stem+'.o');objects.append(obj);cmd=flags+['-c',source/name,'-o',obj]
        commands.append(list(map(str,cmd)));run(cmd,work,obj.stem+'.log')
    cmd=flags+objects+['-o',work/'probe'];commands.append(list(map(str,cmd)));run(cmd,work,'link.log')
    if args.repair_input is not None:
        shutil.copy2(args.repair_input,out/'repair_input.bin')
        cmd=[work/'probe',out/'repair_result.bin',out/'repair_input.bin']
        commands.append(list(map(str,cmd)));run(cmd,work,'repair.log')
    run([work/'probe',out/'result.bin'],work,'run.log')
    values=np.fromfile(out/'result.bin','<f8').reshape(4,5);expected=np.array([2.**(e-10) for e in [-30,-60,-100]])
    assert np.array_equal(values[:3,0],expected),'the original linear filter did not generate the predicted dyadic trace'
    assert np.array_equal(values[:3,2],expected) and np.all(values[:3,3]==1) and np.all(values[:3,4]==0)
    if args.expect_repaired:
        assert np.array_equal(values[:3,1],expected)
    else:
        assert np.count_nonzero(values[:3,1]==0.)==2
    limited=values[3];assert limited[4]==1 and 0.<limited[3]<1.
    assert limited[1]==limited[2] and limited[1]>0.
    exact=Fraction(193,256)*Fraction(15,65536)-Fraction(15,128)*Fraction(193,131072)
    assert exact==0
    report=dict(source_sha256=hashes,commands=commands,raw=values[:,0].tolist(),cpu=values[:,1].tolist(),old_gpu=values[:,2].tolist(),
                theta=values[:,3].tolist(),limited=values[:,4].tolist(),cpu_erased_positive_rows=np.where((values[:,0]>0)&(values[:,1]==0))[0].tolist(),
                cpu_copy_revision_checked=args.expect_repaired,limited_case_exact_cpu_gpu=True,cpp_ported=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','commands']},indent=2))

if __name__=='__main__':main()
