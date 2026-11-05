"""Original normalized probe/wall kernels through explicit runtime products."""
import csv
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from prepare_m12_local_case import prepare
from run_output_restart_validation import compare_fields
from convert_legacy_input import convert_case,namelist,RealLiteral
from run_m12_hd_coverage import prepare_probe_map

ROOT = Path(__file__).resolve().parents[2]
EXE = os.environ.get('ASTR_RUNTIME_M12_EXE')
MPI = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
pytestmark = pytest.mark.skipif(not EXE, reason='Select the root M12-UDF executable')


def definitions():
    return [dict(id=kind,kind=kind,enable_request=1,source='instantaneous',
        sampling=dict(mode='provider'),output=dict(channel='data',format='csv'),
        schedule=dict(mode='steps',interval=1 if kind=='m12_probes' else 5))
        for kind in ('m12_probes','m12_wall')]


def case(root,name,steps,records,restore=None):
    dst=root/name
    prepare(ROOT.parent/'M12_C13',dst,'gpu',intervals=(64,32,24))
    prepare_probe_map(dst,(64,32,24))
    (dst/'outdat/new').mkdir()
    groups=convert_case((dst/'datin/input.dat').read_text(),model_environment={
        'ASTR_WALL_BLOWING_MODE':'legacy_random','ASTR_NSCBC_FARFIELD_MODE':'compatibility',
        'ASTR_GPU_PRECISION_MODE':'fp64','ASTR_GPU_MP_POSITIVITY':'off'})
    groups['boundary']['bctype']=[11,22,41,52,1,1]
    groups['boundary']['pout']=RealLiteral('1.d0')
    (dst/'datin/input.case').write_text(namelist(groups))
    (dst/'datin/input.run').write_text(f"""&run_control
 max_steps={steps}
/
&timestep
 mode='fixed',fixed_dt=0.02d0
/
&repair
 crashfix=t
/
""")
    (dst/'datin/input.insitu').write_text("""&insitu_run
 enabled=t,statistics=t,render=f,products='catalog_stats',
 statistics_window=0.d0,1.d0,
 processing_backend='host',derivative_backend='cpu',
 host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0,
 output_directory='outdat/render'
/
""")
    (dst/'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',exact_restart=t,
 host_budget_bytes=-1,device_budget_bytes=-1,buffer_bytes=1048576,device_reserve_bytes=0
/
&checkpoint
 enabled=t,mode='steps',interval_steps=5,keep=2,initial_frame=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    (dst/'input.products.jsonl').write_text('\n'.join(json.dumps(r) for r in
        [dict(catalog_schema='astr-insitu-products-v1'),*records])+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY='2,1,1',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_SYNC_MODE='explicit',ASTR_INSITU_CONFIG='datin/input.insitu',
        OMPI_MCA_opal_cuda_support='false',OMPI_MCA_coll_hcoll_enable='0',
        OMPI_MCA_coll_ucc_enable='0',OMPI_MCA_osc='pt2pt')
    with (dst/'run.log').open('w') as log:
        completed=subprocess.run([str(MPI),'--bind-to','none','--mca','pml','ob1',
            '--mca','btl','self,vader,tcp','-np','2',str(Path(EXE).resolve()),
            'run','datin/input.case'],cwd=dst,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert completed.returncode==0,(dst/'run.log').read_text()[-6000:]
    status=json.loads((dst/'outdat/new/products.status.json').read_text())
    assert all(p['state']=='active' for p in status['products']),status
    return dst


def checkpoint(dst,step):
    return dst/f'outdat/new/checkpoints/step{step:012d}/state.h5'


def probe_rows(dst):
    files=list((dst/'outdat/new/monitor_products').rglob('probes.csv'))
    assert len(files)==1,files
    with files[0].open() as stream:
        return list(csv.DictReader(stream))


def test_original_diagnostics_preserve_state_values_and_continuation(tmp_path):
    off=case(tmp_path,'off',10,[])
    full=case(tmp_path,'full',10,definitions())
    compare_fields(checkpoint(off,10),checkpoint(full,10))
    rows=probe_rows(full)
    final=[r for r in rows if int(r['step'])==10]
    assert len(final)==72
    indices=np.loadtxt(full/'datin/insitu_probes.dat',skiprows=1,usecols=(0,1,2),dtype=int)
    with h5py.File(checkpoint(full,10)) as h:
        fields=[h[f'q{i:04d}'] for i in range(1,6)]
        assert fields[0].shape==(25,33,65),fields[0].shape
        q=np.array([[float(field[k,j,i]) for field in fields] for i,j,k in indices])
    rho=q[:,0];v=q[:,1:4]/rho[:,None]
    energy=q[:,4]-.5*rho*np.sum(v*v,axis=1)
    pinf=1/(1.4*12**2)
    pp=energy/(1/(1.4-1))/pinf
    expected=np.column_stack((v,rho,pp,pp/rho))
    names=['u_Uinf','v_Uinf','w_Uinf','rho_rhoinf','p_pinf','T_Tinf']
    actual=np.array([[float(row[name]) for name in names] for row in final])
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-10)
    walls=list((full/'outdat/new/wall_products').rglob('wall.step000000000006.rank*.csv'))
    assert len(walls)==2,walls
    values=np.concatenate([np.loadtxt(f,delimiter=',',skiprows=1) for f in sorted(walls)])
    np.testing.assert_array_equal(values[:,0],np.arange(65))
    assert np.isfinite(values).all()

    prefix=case(tmp_path,'prefix',5,definitions())
    tail=case(tmp_path,'tail',10,[],restore=checkpoint(prefix,5).parent)
    compare_fields(checkpoint(full,10),checkpoint(tail,10))
    assert probe_rows(tail)==[r for r in rows if int(r['step'])>5]
    for source in walls:
        matches=list((tail/'outdat/new/wall_products').rglob(source.name))
        assert len(matches)==1
        assert source.read_bytes()==matches[0].read_bytes()
