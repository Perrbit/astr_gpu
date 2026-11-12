"""Explicitly migrate approved KEEP nonreflecting checkpoints between E/C operators."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_keep_periodic_gate import compare_q
from run_output_restart_validation import run_case

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('outdat/new/checkpoints/step000000000004')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--explicit-source',type=Path,required=True)
    parser.add_argument('--compact-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    runtime="""&run_control max_steps=4 /
&timestep mode='fixed',fixed_dt=1.d-5 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
"""
    report=dict(status='running',scope='NP=2 y, nonreflecting KEEP, step2 to4, no filter/repair; no exact cross-method claim',cases={})
    for label,source,scheme in (('e_to_c',args.explicit_source,'643c'),('c_to_e',args.compact_source,'643e')):
        entry={};finals={}
        for backend in ('cpu','gpu'):
            seed=source.resolve()/f'{backend}_np2_keep_open_y/outdat/new/checkpoints/step000000000002'
            opts=SimpleNamespace(output=out,executable=getattr(args,backend).resolve(),mpiexec=args.mpiexec.resolve(),
                case='curve',mode='steps',restart_step=2,initial_dimension=3,initial_restart=False,
                legacy_statistics=False,statistics=False,filter_workspace='scalar',force='fixed',
                axis='y',no_samples=True,wall_samples=False,unrestricted=True)
            common=dict(restore=seed,lfilter=False,scheme=scheme,numerics=dict(convection_form='keep',
                        viscous_form='standard',difschm='643e',conschm=scheme),
                        boundary_options=dict(nscbc_farfield_mode='nonreflecting'),flatplate_warp=(0,0),
                        flatplate_bctype=(22,52),run_config_text=runtime,initial_resource=source.resolve()/'initial.h5',
                        rk_state_check=True,rk_stage_state_check=True)
            run_case(opts,ROOT,backend,2,label+'_exact',4,reject='compact operator identity mismatch',**common)
            case,_=run_case(opts,ROOT,backend,2,label+'_nonexact',4,exact_restart=False,restore_probe=True,**common)
            with h5py.File(seed/'state.h5') as before,h5py.File(case/'outdat/restore_probe.h5') as after:
                for component in range(1,6):
                    key=f'q{component:04d}'
                    if not np.array_equal(before[key][:],after[key][:]):
                        raise AssertionError(f'Nonexact method switch changed restored authoritative {key}')
            identity=(case/FINAL/'control.bin').read_bytes()
            target=b'/o24' if scheme=='643e' else b'/o34'
            if target not in identity or b'keep-top-nr-corner-conv-v1' not in identity or b'/wall-z0-v1' not in identity:
                raise AssertionError('Migrated KEEP operator/boundary identity is missing')
            compare_q(case/FINAL/'state.h5',case/FINAL/'state.h5')
            entry[backend]=dict(source=str(seed),target_case=str(case),exact='rejected different operator',
                                restored_q='five authoritative arrays bit-identical before advancement',target_scheme=scheme)
            finals[backend]=case/FINAL/'state.h5'
        entry['cpu_gpu_final']=compare_q(finals['cpu'],finals['gpu'])
        report['cases'][label]=entry
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(label,'exact rejection, nonexact state restoration and continuation compared',flush=True)
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
