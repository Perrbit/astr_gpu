"""CPU-reference channel fixture for the existing y+ sponge stage and restart."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/gpu_validation'))
from test_high_order_wall_runtime import prepare
from prepare_tgv_case import set_sponge
from named_fixture import prepare_named_run,namelist
from run_cpp_tgv_gate import compare,checkpoint,logged_steps,snapshot
from run_cpp_physical_gate import stages
from run_cpp_high_wall_gate import rhs_phase
from run_cpp_multi_tgv_gate import layouts
from run_cpp_insitu_gate import compare_hdf
from run_cpp_native_tgv import field as native_field,stage_report as native_stages


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','output','mpiexec'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--freeze-only',action='store_true');parser.add_argument('--reference',type=Path)
    parser.add_argument('--ranks',type=int,choices=[1,2],default=1)
    parser.add_argument('--transport',choices=['pinned','device'],default='pinned')
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    if not args.freeze_only and (args.reference is None or not (args.reference/'reference.json').is_file()):
        parser.error('a completed CPU sponge reference is required before the C++ run')
    def compare_data(a,b):
        return compare(a.astype(np.float64),b.astype(np.float64),limit=None if args.precision=='fp32' else 2e-10)
    def field(path):return native_field(path)[1] if args.precision=='fp32' else snapshot(path)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_HIGH_ORDER_WALL_MODE='off',
        ASTR_CHANNEL_FORCE_MODE='fixed',ASTR_CHANNEL_FORCE_FIXED='0',ASTR_GPU_MP_POSITIVITY='off',
        ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE='explicit' if args.freeze_only else 'dependency',
        ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0',ASTR_VALIDATION_RHS_STEP_END='4')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    def run(name,steps=4,writing=True,restore=None):
        case=root/name;gpu=not args.freeze_only
        prepare(case,'gpu' if gpu else 'cpu',steps,None,grid=(16,16,16),filtering=False)
        set_sponge(case/'datin/input.chl','0,0,0,4,0,0')
        execution=env|dict(ASTR_FORCE_MPI_TOPOLOGY=f'1,{args.ranks},1')
        prepare_named_run(case,'datin/input.chl',execution)
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),
            repair=dict(crashfix=False),statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            restore_directory=str(restore or ''),checkpoint_statistics=False,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=writing,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks),str(args.binary.resolve()),'run'],
                cwd=case,env=execution,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        if gpu and ('backend=cuda' not in (case/'run.log').read_text() or
                    'ASTR_GPU_CPP sponge rank=' not in (case/'run.log').read_text()):
            raise AssertionError('actual C++ sponge main not selected')
        return case
    current=run('cpu' if args.freeze_only else 'cpp');final=Path('outdat/new/checkpoints/step000000000004')
    report=dict(scope='private y-sponge channel numerical fixture, not a physical channel validation',
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),precision=args.precision,ranks=args.ranks,transport=args.transport)
    if args.freeze_only:
        if args.ranks!=1:raise AssertionError('CPU reference must be NP=1')
        if not np.isfinite(checkpoint(current/final/'state.h5')).all():raise AssertionError('nonfinite CPU reference')
        report['oracle']='original CPU only; old GPU channel main does not invoke this sponge'
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n');shutil.copyfile(__file__,root/'preparation.py')
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        root.chmod(0o555);print(json.dumps(report,indent=2));return
    if args.reference is None:parser.error('--reference required')
    reference=args.reference/'cpu'
    phase=(native_stages(current,reference,layouts(current)) if args.precision=='fp32' else stages(current,reference))
    report.update(stages=phase,terminal=compare_data(checkpoint(current/final/'state.h5'),checkpoint(reference/final/'state.h5')),
        scalars=compare_data(np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2),np.loadtxt(reference/'flowstate.dat',skiprows=1,ndmin=2)))
    if args.precision=='fp64':
        report.update(convection=rhs_phase(current,reference,'conv'),full_rhs=rhs_phase(current,reference,'full'))
    if not np.array_equal(logged_steps(current),logged_steps(reference)):raise AssertionError('sponge CPU/C++ clock')
    quiet=run('without-output',writing=False)
    for rank in layouts(current):
        filename=f'outdat/rhs.cpp_final.step00000004.rk01.rank{rank:08d}.bin'
        if field(current/filename).tobytes()!=field(quiet/filename).tobytes():raise AssertionError('sponge output changed q')
    if not np.array_equal(logged_steps(current),logged_steps(quiet)):raise AssertionError('sponge output changed clock')
    prefix=run('prefix',steps=2);tail=run('tail',restore=prefix/'outdat/new/checkpoints/step000000000002')
    compare_hdf(current/final/'state.h5',tail/final/'state.h5',exact=True)
    if (current/final/'control.bin').read_bytes()!=(tail/final/'control.bin').read_bytes():raise AssertionError('sponge restart control')
    whole=np.loadtxt(current/'flowstate.dat',skiprows=1,ndmin=2);part=np.loadtxt(tail/'flowstate.dat',skiprows=1,ndmin=2)
    if not np.array_equal(part,whole[np.isin(whole[:,0],part[:,0])]):raise AssertionError('sponge restarted scalar/driver bits')
    report.update(output_state_clock_bits=True,exact_state_control_scalars=True)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
