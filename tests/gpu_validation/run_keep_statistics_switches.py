"""Check source/target KEEP identity cases with retained native statistics checkpoints."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_keep_statistics_migration_probe import mean_metadata,formal_quantities
from run_output_restart_validation import run_case
from run_keep_periodic_gate import compare_q

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    unlimited=np.iinfo(np.int64).max//4096
    insitu=f"""&insitu_run
 enabled=t,statistics=t,render=f,statistics_window=0.d0,1.d0,
 output_directory='outdat',host_budget_bytes={unlimited},device_budget_bytes={unlimited},device_reserve_bytes=0
/
"""
    report=dict(status='running',scope='NP=1, source-free periodic 16-cubed TGV, FP64, dt=.001, no filter/repair',cases={})
    standards={}
    for label,form,end,fresh in [('c_to_e','keep',8,True),('keep_to_standard','standard',8,True),
                               ('standard_to_keep','keep',10,True),('standard_same','standard',10,False)]:
        runs={};entry={}
        for backend in ('cpu','gpu'):
            if label=='c_to_e':
                seed=args.source_root.resolve()/f'{backend}_np1_mean_migrated/outdat/new/checkpoints/step000000000006'
            elif label=='keep_to_standard':
                seed=args.source_root.resolve()/f'{backend}_np1_mean_source/outdat/new/checkpoints/step000000000006'
            else:
                seed=standards[backend]
            old=mean_metadata(seed/'statistics.h5','insitu')
            first_step=old['last_step'] if fresh else old['first_step']
            first_time=old['last_time'] if fresh else old['first_time']
            opts=SimpleNamespace(output=out,executable=getattr(args,backend).resolve(),mpiexec=args.mpiexec.resolve(),
                case='tgv',mode='steps',restart_step=4,initial_dimension=0,initial_restart=False,
                legacy_statistics=False,statistics=True,filter_workspace='scalar',force='feedback',
                axis='x',no_samples=True,wall_samples=False,unrestricted=True)
            runtime=f"""&run_control max_steps={end} /
&timestep mode='fixed',fixed_dt=1.d-3 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
"""
            case,_=run_case(opts,ROOT,backend,1,label,end,restore=seed,exact_restart=False,
                            restore_probe=True,lfilter=False,scheme='643e',run_config_text=runtime,
                            numerics=dict(convection_form=form,difschm='643e'),insitu_config=insitu)
            final=case/f'outdat/new/checkpoints/step{end:012d}'
            observed=mean_metadata(final/'statistics.h5','insitu')
            expected_duration=observed['last_time']-first_time
            if observed['first_step']!=first_step or observed['first_time']!=first_time or abs(observed['duration']-expected_duration)>2e-10:
                raise AssertionError(f'Method switch/reset scope differs: {label} {backend}: {observed}')
            log=(case/'run.log').read_text()
            if ('ASTR_KEEP_RESTART statistics_new_segment_step=' in log)!=fresh:
                raise AssertionError('KEEP statistics reset marker does not match the actual switch scope')
            entry[backend]=dict(source=str(seed),target=str(final),statistics=observed,
                restored_field=compare_q(seed/'state.h5',case/'outdat/restore_probe.h5'))
            if label=='keep_to_standard':
                standards[backend]=final
            runs[backend]=final
        entry['cpu_gpu_field']=compare_q(runs['cpu']/'state.h5',runs['gpu']/'state.h5')
        difference=float(np.max(abs(formal_quantities(runs['cpu']/'statistics.h5')-
                                   formal_quantities(runs['gpu']/'statistics.h5'))))
        if difference>2e-10:
            raise AssertionError(f'CPU/GPU statistics differ: {label}: {difference}')
        entry['cpu_gpu_normalized_statistics_max_abs']=difference
        report['cases'][label]=entry
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'first step',first_step,'duration',observed['duration'],'statistics difference',difference,flush=True)
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
