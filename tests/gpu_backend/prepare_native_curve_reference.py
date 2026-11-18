"""Freeze the existing FP5 native-FP32 CURVE fixtures before C++ comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from named_fixture import prepare_named_run
from run_native_precision_curve_gate import compare as compare_original
from run_cpp_tgv_gate import namelist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original','output','mpiexec'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mapping',choices=('periodic','y-wavy'),required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--uniform',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(args.original,root/'original_astr')
    topology=[1,1,1];topology[1 if args.mapping=='y-wavy' else 0]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_GEOMETRY_DUMP='geometry.h5',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_VALIDATION_RHS_PREFIX='state',
        ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_SECONDARY='3',ASTR_VALIDATION_GRADIENTS='1')
    for backend in ('cpu','gpu'):
        case=root/backend
        command=[sys.executable,str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
            '--src-case',str(ROOT/'examples/Taylor_Green_Vortex'),'--dst-case',str(case),
            '--use-gpu','t' if backend=='gpu' else 'f','--grid','32,32,32','--maxstep','3',
            '--feqchkpt','1','--feqlist','1','--lfilter','t','--diffterm','t','--lreadgrid','t',
            '--gridfile','datin/grid.curvilinear.h5','--omit-grid-resource','--scheme','643e',
            '--ninit','3' if args.uniform else '0','--restart','f','--deltat','2.d-5']
        if args.mapping=='y-wavy':command+=['--homogeneous','t,f,t','--bctype','1;1;41,273.15d0;41,273.15d0;1;1']
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/generate_curvilinear_tgv_grid.py'),
            '--output',str(case/'datin/grid.curvilinear.h5'),'--report',str(case/'grid_quality.txt'),
            '--grid','32,32,32','--amplitude','.15','--mapping',args.mapping],check=True)
        if args.uniform:
            wall=args.mapping=='y-wavy'
            subprocess.run([sys.executable,str(ROOT/'tests/gpu_validation/generate_uniform_flow_field.py'),
                '--output',str(case/'datin/flowini3d.h5'),'--grid','32,32,32',
                '--density',str(1/273.15 if wall else 1),'--temperature',str(273.15 if wall else 1),
                '--u1','0' if wall else '.7','--u2','0' if wall else '-.2','--u3','0' if wall else '.1'],check=True)
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))
        execution=dict(env);primary=prepare_named_run(case,'datin/input.tgv',execution)
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks),
                str(root/'original_astr'),'run',primary],cwd=case,env=execution,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
    report=dict(precision='fp32',mapping=args.mapping,uniform=args.uniform,topology=topology,
        grid=[32]*3,steps=4,dt=2e-5,filter='explicit10',
        binary_sha256=hashlib.sha256((root/'original_astr').read_bytes()).hexdigest(),
        comparison=compare_original(root/'cpu',root/'gpu','fp32',args.mapping,args.uniform))
    (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,root/'preparation.py')
    for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
    root.chmod(0o555)
    print(json.dumps({k:v for k,v in report.items() if k!='comparison'},indent=2))


if __name__=='__main__':main()
