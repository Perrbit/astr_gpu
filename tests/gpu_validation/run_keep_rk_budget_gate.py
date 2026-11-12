"""Collect only the missing periodic KEEP RK/filter snapshots and check ON/OFF invariance."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analyze_keep_rk_budgets import analyze
from run_keep_periodic_gate import compare_q
from run_output_restart_validation import run_case,compare_fields

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('outdat/new/checkpoints/step000000000004/state.h5')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    report=dict(status='running',scope='periodic TGV; original RK3 and filters; no wall or open-boundary acceptance',cases={})
    matrix=[('explicit_compact10','643e','standard','compact10',None,'16,16,16'),
            ('compact_me6_explicit10_curve','643c','me6_opti','explicit10','periodic','32,32,32')]
    for label,scheme,viscous,filter_scheme,mapping,grid in matrix:
        entry={};runs={};budgets={}
        for backend in ('cpu','gpu'):
            opts=SimpleNamespace(output=out,executable=getattr(args,backend).resolve(),mpiexec=args.mpiexec.resolve(),
                case='tgv',mode='steps',restart_step=2,initial_dimension=0,initial_restart=False,
                legacy_statistics=False,statistics=False,filter_workspace='scalar',force='feedback',
                axis='x',no_samples=True,wall_samples=False,unrestricted=True)
            common=dict(lfilter=True,scheme=scheme,numerics=dict(convection_form='keep',
                        viscous_form=viscous,difschm='643e',filter_scheme=filter_scheme),
                        grid=grid,tgv_mapping=mapping)
            observed,_=run_case(opts,ROOT,backend,1,label+'_snapshots',4,rhs_snapshot_step=0,**common)
            plain,_=run_case(opts,ROOT,backend,1,label+'_plain',4,**common)
            compare_fields(observed/FINAL,plain/FINAL)
            budgets[backend]=analyze(observed,.001,True)
            entry[backend]=dict(snapshot_case=str(observed),plain_case=str(plain),
                                snapshot_on_off='all state datasets bit-identical',budget=budgets[backend])
            runs[backend]=observed
        entry['cpu_gpu_fields']=compare_q(runs['cpu']/FINAL,runs['gpu']/FINAL)
        entry['cpu_gpu_terms_max_scaled']={}
        for name in budgets['cpu']['terms']:
            a=np.array(budgets['cpu']['terms'][name]);b=np.array(budgets['gpu']['terms'][name])
            delta=float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
            entry['cpu_gpu_terms_max_scaled'][name]=delta
            if delta>2e-10:
                raise AssertionError(f'KEEP RK contribution differs: {label} {name}: {delta}')
        report['cases'][label]=entry
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'state transitions, term comparison and snapshot ON/OFF identical',flush=True)
    report['status']='passed_transition_and_balance_identities'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
