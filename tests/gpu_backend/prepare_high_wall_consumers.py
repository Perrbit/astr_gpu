"""Freeze original eight-node wall consumers on existing Hermite7 inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_cpp_tgv_gate import checkpoint,compare,logged_steps


def read_wall(path):
    with path.open('rb') as f:
        if f.read(8)!=b'ASTRIW01':raise AssertionError('wall snapshot magic')
        header=np.fromfile(f,'<i4',9);clock=np.fromfile(f,'<f8',2)
        shape=tuple(header[3:6]);nodes=int(np.prod(shape))
        xyz=np.fromfile(f,'<f8',3*nodes).reshape((*shape,3),order='F')
        fields=np.fromfile(f,'<f8',4*nodes).reshape((*shape,4),order='F')
        owned=np.fromfile(f,'<i4',nodes).reshape(shape,order='F')
        if f.read(1) or header[0]!=1 or not np.isfinite(fields).all() or not np.isfinite(xyz).all():
            raise AssertionError('wall snapshot layout/finiteness')
        return header,clock,xyz,fields,owned


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','fixture','output','mpiexec'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=out/'frozen_original';shutil.copy2(a.binary,binary);binary.chmod(0o555)
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_FORCE_MPI_TOPOLOGY='1,1,1',ASTR_INSITU_SAMPLE_PREFIX='outdat/sample')
    reports={}
    for backend in ('cpu','gpu'):
        case=out/backend;shutil.copytree(a.fixture/backend/'datin',case/'datin',copy_function=shutil.copyfile)
        (case/'datin').chmod(0o755);(case/'outdat/new').mkdir(parents=True)
        execution=dict(env)
        if backend=='gpu':execution['ASTR_INSITU_TEST_DEVICE_TRANSPORT']='pinned'
        with (case/'run.log').open('x') as log:
            subprocess.run([str(a.mpiexec.resolve()),'-n','1',str(binary),'run'],cwd=case,env=execution,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        snapshots=sorted((case/'outdat').glob('sample.wall.step*.bin'))
        if [int(read_wall(p)[0][1]) for p in snapshots]!=[1,2,3,4]:raise AssertionError('four completed wall snapshots')
        final='outdat/new/checkpoints/step000000000004/state.h5'
        reports[backend]=dict(snapshot_steps=[1,2,3,4],clock=logged_steps(case).tolist(),
            existing_flow_reference=compare(checkpoint(case/final),checkpoint(a.fixture/backend/final)))
    differences={}
    for path in sorted((out/'cpu/outdat').glob('sample.wall.step*.bin')):
        first,second=read_wall(path),read_wall(out/'gpu/outdat'/path.name)
        for index in (0,1,2,4):
            if first[index].tobytes()!=second[index].tobytes():raise AssertionError('original wall metadata/geometry/ownership')
        differences[path.name]=compare(second[3],first[3])
    report=dict(scope='original CPU/CUDA-Fortran eight-node wall references, no new C++',fixture=str(a.fixture.resolve()),
        binary_source=str(a.binary.resolve()),binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        cases=reports,wall_differences=differences)
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,out/'runner.py')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
