"""Existing Hermite7 eight-node legacy wall curves/history, ordinary programs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare_legacy_wall_reference import definitions
from run_cpp_tgv_gate import checkpoint,compare,logged_steps,namelist
from run_cpp_restart_gate import exact_state
from run_legacy_wall_runtime import curves,compare_curves,wall_history
from run_native_precision_restart_gate import check_state
from run_output_restart_validation import archive_schedule_payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','original','fixture','output','mpiexec'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ranks',type=int,choices=(1,2),default=1)
    p.add_argument('--axis',choices=('x','y','z'),default='y');p.add_argument('--memcheck',action='store_true')
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binaries={}
    for key,source in (('cpp',a.binary),('old',a.original)):
        binaries[key]=root/key;shutil.copy2(source,binaries[key]);binaries[key].chmod(0o555)
    topology=[1,1,1];topology['xyz'.index(a.axis)]=a.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',
        OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_FILTER_WORKSPACE='scalar',ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
        ASTR_INSITU_CONFIG='datin/input.insitu')
    budget=(2**63-1)//4096;final=Path('outdat/new/checkpoints/step000000000004')
    def run(name,binary,steps=4,restore=None,observe=True,instrument=False):
        case=root/name
        shutil.copytree(a.fixture/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755)
        (case/'outdat/new').mkdir(parents=True);(case/'outdat/render').mkdir()
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=steps),
            timestep=dict(mode='fixed',fixed_dt=1e-5),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',
            restore_directory=str(restore or ''),exact_restart=True,checkpoint_statistics=True,
            host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        (case/'datin/input.insitu').write_text(namelist(dict(insitu_run=dict(enabled=observe,statistics=observe,
            render=False,products='channel_walls',statistics_window=[.5e-5,3.5e-5],output_directory='outdat/render',
            host_budget_bytes=budget,device_budget_bytes=budget,device_reserve_bytes=0))))
        records=definitions() if observe and restore is None else []
        (case/'input.products.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in
            [dict(catalog_schema='astr-insitu-products-v1'),*records]))
        command=[str(a.mpiexec.resolve()),'--bind-to','none','-n',str(a.ranks)]
        if instrument:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
            '--tool','memcheck','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command +=[str(binary),'run']
        with (case/'run.log').open('x') as log:
            subprocess.run(command,cwd=case,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        return case
    old=run('original',binaries['old']);previous=curves(old,nodes=17)
    actual=run('current',binaries['cpp'],instrument=a.memcheck);values=curves(actual,nodes=17)
    report=dict(topology=topology,binary_sha256=hashlib.sha256(binaries['cpp'].read_bytes()).hexdigest(),
        original_sha256=hashlib.sha256(binaries['old'].read_bytes()).hexdigest(),
        wall_curves=compare_curves(values,previous,'fp64'),
        wall_history=wall_history(actual/final/'statistics.h5',old/final/'statistics.h5','fp64'),
        flow=compare(checkpoint(actual/final/'state.h5'),checkpoint(old/final/'state.h5')),
        physical=check_state(actual/final/'state.h5','fp64'))
    if logged_steps(actual).tolist()!=logged_steps(old).tolist():raise AssertionError('reference clocks')
    quiet=run('quiet',binaries['cpp'],observe=False)
    exact_state(actual/final/'state.h5',quiet/final/'state.h5')
    prefix=run('prefix',binaries['cpp'],steps=2)
    tail=run('tail',binaries['cpp'],restore=prefix/'outdat/new/checkpoints/step000000000002')
    for name in ('state.h5','statistics.h5'):exact_state(actual/final/name,tail/final/name)
    for name in ('control.bin','insitu_control.bin'):
        if (actual/final/name).read_bytes()!=(tail/final/name).read_bytes():raise AssertionError('restored control: '+name)
    if archive_schedule_payload(actual/final/'archives.bin')!=archive_schedule_payload(tail/final/'archives.bin'):
        raise AssertionError('restored product clock')
    compare_curves(curves(tail,tail=True,nodes=17),values,'fp64',exact=True)
    if a.memcheck:
        logs=list((actual).glob('memcheck.*.log'))
        if len(logs)!=a.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('high-wall consumer memcheck')
    report.update(observation_state_words=True,exact_state_history_controls_curves=True,memcheck=a.memcheck)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('eight-node wall consumers: field/history/reference, observation isolation and exact continuation checked')


if __name__=='__main__':main()
