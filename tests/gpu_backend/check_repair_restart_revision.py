"""Check exact rejection and non-exact continuation across the halo revision."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from prepare_mixed_viscous_reference import prepare_mixed_case
from run_cpp_tgv_gate import namelist, snapshot, compare, logged_steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','template','mpiexec','source','continuous','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.source.resolve()
    with (source/'control.bin').open('rb') as stream:
        stream.read(8);flags=np.fromfile(stream,np.int64,8)
    if int(flags[6])&24==24:raise AssertionError('this test needs an actual pre-repair control revision')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='4')
    for exact in (True,False):
        case=out/('exact' if exact else 'nonexact')
        prepare_mixed_case(case,args.template.resolve(),True,False,False,'none','tgv')
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(source),exact_restart=exact,checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        with (case/'run.log').open('w') as log:
            result=subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',
                str(args.binary.resolve()),'run'],cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if exact:
            if result.returncode==0 or 'numerical/controller contract mismatch' not in (case/'run.log').read_text():
                raise AssertionError('pre-repair exact restore did not reject its numerical identity')
        elif result.returncode!=0:raise AssertionError('explicit non-exact continuation failed')
    file='outdat/rhs.cpp_final.step00000004.rk01.rank00000000.bin'
    error=compare(snapshot(out/'nonexact'/file),snapshot(args.continuous.resolve()/file))
    if not np.array_equal(logged_steps(out/'nonexact',2,first_step=3),logged_steps(args.continuous.resolve())[-2:]):
        raise AssertionError('non-exact fixed-step continuation changed physical time/dt')
    report=dict(source=str(source),old_flag=int(flags[6]),exact_previous_revision_rejected=True,
        nonexact_physical_field_scaled=error,fixed_step_clock_equal=True)
    shutil.copyfile(__file__,out/'runner.py');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
