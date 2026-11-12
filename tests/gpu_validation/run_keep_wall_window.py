"""Unrepaired KEEP wall window with an explicitly selected existing filter."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from run_keep_wall_gate import ROOT,wall_state_budget
from run_output_restart_validation import run_case
from insitu_cfl_gate import CflGate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--initial-snapshot',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--dt',type=float,default=1e-3)
    parser.add_argument('--viscous-form',choices=['standard','me6_opti'],default='me6_opti')
    parser.add_argument('--scheme',choices=['643e','643c'],default='643e')
    parser.add_argument('--filter',choices=['off','explicit10','compact10'],default='off')
    parser.add_argument('--checkpoint-interval',type=int)
    opts=parser.parse_args()
    opts.output=opts.output.resolve();opts.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    args=SimpleNamespace(output=opts.output,executable=opts.gpu.resolve(),mpiexec=opts.mpiexec.resolve(),
        case='channel',mode='steps',restart_step=opts.steps,initial_dimension=3,initial_restart=False,
        legacy_statistics=False,statistics=False,filter_workspace='scalar',force='fixed',axis='y',
        no_samples=True,wall_samples=False,unrestricted=True)
    runtime=f'''&run_control max_steps={opts.steps} /
&timestep mode='fixed',fixed_dt={opts.dt:.17e} /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
'''
    numerics={'convection_form':'keep','conschm':opts.scheme,'viscous_form':opts.viscous_form,'difschm':'643e'}
    if opts.filter!='off':
        numerics['filter_scheme']=opts.filter
    interval=opts.steps if opts.checkpoint_interval is None else opts.checkpoint_interval
    case,_=run_case(args,ROOT,'gpu',2,'keep_wall_window',opts.steps,lfilter=opts.filter!='off',scheme=opts.scheme,
        numerics=numerics,
        initial_resource=opts.initial.resolve(),run_config_text=runtime,
        checkpoint_interval=interval,checkpoint_keep=2,rk_state_check=True,rk_stage_state_check=True)
    cfl=CflGate(case/'run.log',0.5,opts.dt,0,opts.steps)
    cfl.check(final=True)
    checked={rank:[] for rank in range(2)}
    for line in (case/'run.log').read_text().splitlines():
        if line.startswith('ASTR_VALIDATION_RK_STATE_PASS '):
            _,rank,step=line.split()
            checked[int(rank)].append(int(step))
    if any(steps!=list(range(opts.steps)) for steps in checked.values()):
        raise AssertionError('Missing per-rank conservative-state admission at complete RK steps')
    budget=wall_state_budget(case,opts.steps,opts.initial_snapshot.resolve())
    report={'status':'passed','case':str(case),'steps':opts.steps,'dt':opts.dt,
            'scheme':opts.scheme,'viscous_form':opts.viscous_form,'filter':opts.filter,'crashfix':False,
            'checkpoint_interval':interval,'ranks':2,'topology':[1,2,1],
            'admissibility':'existing read-only validators before KEEP RHS and after complete RK; no limiter or repair',
            'cfl':{'maximum':cfl.maximum,'limit':0.5,'check':'post-run completed-state log'},'stored_state_budget':budget,
            'scope':'same-path finite positive stability window; not production or statistical acceptance'}
    (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
