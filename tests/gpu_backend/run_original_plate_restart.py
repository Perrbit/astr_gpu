"""Exercise the approved static-profile output extension on its original backends."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import namelist
from run_cpp_insitu_gate import compare_hdf


def main():
    parser=argparse.ArgumentParser()
    for name in ('reference','reference_results','mpiexec','output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    final=Path('outdat/new/checkpoints/step000000000004')
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_COMPACT_TRANSPORT='pinned',ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_MP_POSITIVITY='off')
    report={}
    for label,binary in [('cpu','astr_cpu'),('old_gpu','astr_cuda_fortran')]:
        previous=args.reference_results.resolve()/label
        def run(name,steps,restore=None):
            case=root/name;case.mkdir()
            shutil.copytree(previous/'datin',case/'datin',copy_function=shutil.copyfile)
            (case/'outdat/new').mkdir(parents=True)
            (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
                timestep=dict(mode='fixed',fixed_dt=1e-6),logging=dict(list_steps=1),repair=dict(crashfix=False),
                statistics=dict(enabled=False))))
            (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
                restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=False,
                host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
                checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
                volume=dict(enabled=False),slices=dict(enabled=False))))
            with (case/'run.log').open('w') as log:
                subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n','1',
                    str((args.reference/'bin'/binary).resolve()),'run'],cwd=case,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
            return case
        prefix=run(label+'_prefix',2)
        tail=run(label+'_tail',4,prefix/'outdat/new/checkpoints/step000000000002')
        compare_hdf(previous/final/'state.h5',tail/final/'state.h5',exact=True)
        if (previous/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():
            raise AssertionError('original-backend restored control differs')
        full=np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)
        resumed=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
        if not np.array_equal(resumed,full[np.isin(full[:,0],resumed[:,0])]):
            raise AssertionError('original-backend restored scalar sequence differs')
        report[label]=dict(exact_state_and_control=True,exact_scalars=True,resumed_rows=len(resumed))
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
