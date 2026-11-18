"""Freeze actual original physical-midspan wall CSVs and independent q-based values."""
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
from prepare_hcw_laminar_sbli_case import prepare_case
from named_fixture import namelist
from test_hcw_insitu import configuration_2d
from test_runtime_catalog_sbli_walls import definition,frames,reference
from test_runtime_catalog_walls import rows
from run_cpp_tgv_gate import checkpoint,compare,logged_steps


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','old-gpu','implementation','mpiexec','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'binaries').mkdir()
    identity={}
    for name,source in (('cpu',a.cpu),('gpu',a.old_gpu)):
        target=out/'binaries'/name;shutil.copy2(source,target);target.chmod(0o555)
        identity[name]=dict(source=str(source.resolve()),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    field_names=('p_pinf','skin_friction','wall_heat_into_wall_normalized')
    records=[definition(name,name) for name in field_names]
    for item in records:item['schedule']['interval']=2
    cases={};report=dict(grid=[64,48,5],steps=4,scope='bounded original GPU midspan reference, not developed physical SBLI')
    for backend in ('cpu','gpu'):
        case=out/backend;cases[backend]=case
        manifest=prepare_case(case,use_gpu='t' if backend=='gpu' else 'f',im=64,jm=48,km=5)
        dt=manifest['run_plan']['dt_nondimensional']
        (case/'datin/input.run').write_text(namelist(dict(run_control=dict(max_steps=4),
            timestep=dict(mode='fixed',fixed_dt=dt),logging=dict(list_steps=1),repair=dict(crashfix=False),
            statistics=dict(enabled=False))))
        (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=True,
            checkpoint_statistics=False,host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
            checkpoint=dict(enabled=True,mode='steps',interval_steps=2,keep=2,initial_frame=False),
            volume=dict(enabled=False),slices=dict(enabled=False))))
        env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
        env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
            OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
            ASTR_FORCE_MPI_TOPOLOGY='1,1,1',ASTR_GPU_SYNC_MODE='explicit',ASTR_INSITU_WRITE_EPS='0')
        if backend=='gpu':
            config=configuration_2d().replace("implementation_path='"+str(__import__('test_hcw_insitu').LIBRARY)+"'",
                "implementation_path='"+str(a.implementation.resolve())+"'")
            config=config.replace('host_budget_bytes=4294967296','host_budget_bytes=-1').replace(
                'device_budget_bytes=2147483648','device_budget_bytes=-1').replace('device_reserve_bytes=1073741824','device_reserve_bytes=0')
            (case/'datin/input.insitu').write_text(config)
            (case/'input.products.jsonl').write_text('\n'.join(json.dumps(p) for p in
                [{'catalog_schema':'astr-insitu-products-v1'},*records])+'\n')
            env['ASTR_INSITU_CONFIG']='datin/input.insitu'
        with (case/'run.log').open('x') as log:
            subprocess.run([str(a.mpiexec.resolve()),'-n','1',str(out/'binaries'/backend),'run','datin/input.case'],
                cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'The job is done!' not in (case/'run.log').read_text():raise AssertionError('reference main did not finish')
    final='outdat/new/checkpoints/step000000000004/state.h5'
    report['flow_scaled']=compare(checkpoint(cases['gpu']/final),checkpoint(cases['cpu']/final))
    if not np.array_equal(logged_steps(cases['gpu']),logged_steps(cases['cpu'])):raise AssertionError('reference clock mismatch')
    status=json.loads((cases['gpu']/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=3 or any(p['state']!='active' for p in status['products']):raise AssertionError(status)
    report['independent_wall']={}
    for step in (2,4):
        x,z,expected=reference(cases['gpu'],step)
        for c,field in enumerate(field_names):
            files=frames(cases['gpu'],field)
            if [int(p.stem.split('step')[1]) for p in files]!=[0,2,4]:raise AssertionError('midspan frame phase')
            path=next(p for p in files if p.name==f'frame.step{step:012d}.csv')
            data=np.asarray([[float(row[k]) for k in ('x','y','z','value')] for row in rows(path)])
            target=np.column_stack([x,np.zeros_like(x),np.full_like(x,z),expected[:,c]])
            report['independent_wall'][field+f'/step{step}']=compare(data,target)
    report['clock']=logged_steps(cases['gpu']).tolist()
    (out/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
    shutil.copy2(__file__,out/'runner.py')
    for path in out.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
    out.chmod(0o555);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
