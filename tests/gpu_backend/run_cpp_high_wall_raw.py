"""Compare existing completed-step wall packets without whole-flow sampling."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare_high_wall_consumers import read_wall
from run_cpp_tgv_gate import checkpoint,compare,logged_steps
from run_cpp_restart_gate import exact_state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','original','fixture','output','mpiexec'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=('x','y','z'),default='y')
    parser.add_argument('--transport',choices=('pinned','device-aware'),default='pinned')
    parser.add_argument('--memcheck',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binaries={}
    for key,source in (('cpp',args.binary),('original',args.original)):
        binaries[key]=root/key;shutil.copy2(source,binaries[key]);binaries[key].chmod(0o555)
    topology=[1,1,1];topology['xyz'.index(args.axis)]=args.ranks
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_INSITU_TEST_DEVICE_TRANSPORT=args.transport)
    if args.transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',UCX_MEMTYPE_CACHE='n')
    def run(name,binary,sample=True,instrument=False):
        case=root/name
        shutil.copytree(args.fixture/'gpu/datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        execution=dict(env)
        if sample:execution['ASTR_INSITU_SAMPLE_PREFIX']='outdat/sample'
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if instrument:
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--error-exitcode','99','--log-file',str(case/'memcheck.%p.log')]
        command +=[str(binary),'run']
        with (case/'run.log').open('x') as log:
            subprocess.run(command,cwd=case,env=execution,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        if binary==binaries['cpp'] and (case/'run.log').read_text().count('backend=cuda')!=args.ranks:
            raise AssertionError('new C++ wall packet backend was not selected')
        return case
    original=run('reference',binaries['original']);current=run('current',binaries['cpp'],instrument=args.memcheck)
    quiet=run('quiet',binaries['cpp'],sample=False)
    report=dict(topology=topology,transport=args.transport,
        binary_sha256=hashlib.sha256(binaries['cpp'].read_bytes()).hexdigest(),
        original_sha256=hashlib.sha256(binaries['original'].read_bytes()).hexdigest(),fields={})
    expected=[f'sample.wall.step{step:08d}.rank{rank:08d}.bin'
        for step in range(1,5) for rank in range(args.ranks)]
    for case in (original,current):
        if sorted(p.name for p in (case/'outdat').glob('sample.wall.step*.bin'))!=sorted(expected):
            raise AssertionError('wall packet rank/completed-step inventory')
    for name in expected:
        reference,actual=read_wall(original/'outdat'/name),read_wall(current/'outdat'/name)
        for index in (0,1,2,4):
            if reference[index].tobytes()!=actual[index].tobytes():
                raise AssertionError('wall packet metadata/coordinates/ownership: '+name)
        report['fields'][name]=compare(actual[3],reference[3])
    final=Path('outdat/new/checkpoints/step000000000004/state.h5')
    report['flow']=compare(checkpoint(current/final),checkpoint(original/final))
    exact_state(current/final,quiet/final)
    if logged_steps(current).tolist()!=logged_steps(original).tolist():raise AssertionError('wall packet clock')
    if args.memcheck:
        logs=list(current.glob('memcheck.*.log'))
        if len(logs)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
            raise AssertionError('wall packet memcheck')
    report.update(observation_state_words=True,memcheck=args.memcheck)
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('wall packet fields, coordinates, ownership, clocks and observation isolation checked')


if __name__=='__main__':main()
