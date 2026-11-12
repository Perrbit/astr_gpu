"""Run the shipped low-order KEEP examples at NP=2 y, not private generated fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from insitu_cfl_gate import CflGate
from run_keep_periodic_gate import compare_q

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=['wall','open','ldc','inflow','plate','y-farfield','incident','incident-compact'],default=['wall','open','ldc','inflow','plate'])
    parser.add_argument('--viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--inlet-mode',choices=['prof','intp'],default='prof')
    parser.add_argument('--blowing',choices=['off','modal','counter','legacy_random'],default='off')
    parser.add_argument('--extruded',action='store_true')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
        ASTR_FORCE_MPI_TOPOLOGY='1,2,1',ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_VALIDATION_RK_STATE='1',ASTR_VALIDATION_PROFILE_STAGE_STATE='1')
    report=dict(status='running',topology=[1,2,1],viscosity=args.viscosity,convection=args.convection,
                inlet_mode=args.inlet_mode,blowing=args.blowing,extruded=args.extruded,cases={})
    for kind in args.cases:
        states={};entry={}
        for backend in ('cpu','gpu'):
            case=out/(kind+'_'+backend)
            subprocess.run([sys.executable,str(ROOT/'examples/KEEP_Low_Order_Boundaries/prepare.py'),
                '--case',kind,'--backend',backend,'--viscosity',args.viscosity,'--convection',args.convection,
                '--inlet-mode',args.inlet_mode,'--blowing',args.blowing,'--output',str(case),
                *(['--extruded'] if args.extruded else [])],check=True)
            command=[str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np','2',
                     str(getattr(args,backend).resolve()),'run','datin/input.case']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            cfl=CflGate(case/'run.log',.5,1e-5,0,4);cfl.check(final=True)
            states[backend]=case/'outdat/new/checkpoints/step000000000004/state.h5'
            entry[backend]=dict(case=str(case),command=command,maximum_c_upper=cfl.maximum)
        entry['cpu_gpu']=compare_q(states['cpu'],states['gpu'])
        report['cases'][kind]=entry
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(kind,'maximum scaled field difference:',max(v['max_scaled'] for v in entry['cpu_gpu'].values()),flush=True)
    report['status']='passed_four_step_examples'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
