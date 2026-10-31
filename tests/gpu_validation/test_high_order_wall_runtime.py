"""HB1-B actual ASTR channel: same-stage fields and density-evolved walls."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import h5py
import numpy as np
import pytest

from prepare_tgv_case import (set_grid,set_homogeneous,set_runtime_flags,set_scheme,
    set_ninit,set_restart,set_gridfile,set_bctype,set_flowtype)
from prepare_s1_flatplate_case import write_controller
from run_m12_mp7_nscbc_performance import runtime,mpi_command
from run_native_precision_tgv_gate import physical_state,rhs_state
from run_output_restart_validation import compare_fields
from compare_flowstate import read_flowstate

ROOT=Path(__file__).resolve().parents[2]
EXE=os.environ.get('ASTR_HB1B_EXE')
MPIEXEC='/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'
pytestmark=pytest.mark.skipif(not EXE,reason='Set ASTR_HB1B_EXE to the root-CMake FP64 CUDA executable')
DT=1e-5
GRID=(32,32,16)


def initial_fields(grid=GRID,manufactured=False,wall_type=41):
    x,y,z=np.meshgrid(np.linspace(0,2*np.pi,grid[0]+1),np.linspace(0,1,grid[1]+1),
        np.linspace(0,2*np.pi,grid[2]+1),indexing='ij')
    wave=np.cos(x)*np.cos(z)
    fields=dict(ro=1+.01*wave*(1+.02*np.cos(np.pi*y)),
        u1=.1*np.sin(np.pi*y)*(1+.03*wave),u2=.01*np.sin(2*np.pi*y)*wave,
        u3=.02*np.sin(np.pi*y)*np.sin(x)*np.sin(z),t=1+.02*np.sin(np.pi*y)**2*wave)
    if manufactured:
        fields=dict(ro=1+.01*np.cos(np.pi*y),u1=.1*np.sin(np.pi*y),
            u2=.01*np.sin(2*np.pi*y),u3=.02*np.sin(np.pi*y),t=1+.02*np.sin(np.pi*y)**2)
    if wall_type==421:
        fields['u1']=(.1+.02*np.cos(np.pi*y))*(1 if manufactured else 1+.03*wave)
        fields['u3']=.02*np.cos(2*np.pi*y) if manufactured else .02*np.cos(np.pi*y)*np.sin(x)*np.sin(z)
    return (x,y,z),fields


def defined_extras(state,normal_axis=2):
    """Select physical replicas and one-face halos, excluding physical exterior ghosts."""
    global_cells=state['identity'][3:6]-1
    offset=0
    for partition in state['partitions'][:].reshape(-1,8):
        origin,cells=partition[:3],partition[3:6]
        halo,count=map(int,partition[6:8])
        shape=tuple(cells+2*halo+1)
        indices=np.indices(shape)-halo
        owned=cells+(origin+cells==global_cells)
        owner=np.all((indices>=0)&(indices<owned[:,None,None,None]),axis=0)
        extra=~owner.ravel(order='F')
        face=np.count_nonzero((indices<0)|(indices>cells[:,None,None,None]),axis=0)<=1
        global_normal=indices[normal_axis-1]+origin[normal_axis-1]
        defined=face&(global_normal>=0)&(global_normal<=global_cells[normal_axis-1])
        width=int(np.count_nonzero(extra))
        assert count==11*width
        payload=state['rank_extras'][offset:offset+count].reshape(11,width)
        yield payload[:,defined.ravel(order='F')[extra]]
        offset+=count
    assert offset==state['rank_extras'].size


def prepare(case,backend,steps,restore,grid=GRID,manufactured=False,dt=DT,checkpoint_interval=1,wall_type=41,normal_axis=2,flow_type=None,initial_condition=None,filtering=False):
    datin=case/'datin';datin.mkdir(parents=True)
    (case/'outdat/new').mkdir(parents=True)
    (case/'diagnostics').mkdir()
    primary=datin/'input.chl'
    shutil.copy2(ROOT/'examples/Channel/datin/input.chl',primary)
    permutation={1:(1,2,0),2:(0,1,2),3:(2,0,1)}[normal_axis]
    target_grid=tuple(grid[p] for p in permutation)
    flags=['t']*3;flags[normal_axis-1]='f'
    set_grid(primary,','.join(map(str,target_grid)));set_homogeneous(primary,','.join(flags))
    set_runtime_flags(primary,'t' if backend=='gpu' else 'f','t' if filtering else 'f','t','t')
    set_scheme(primary,'643e');set_ninit(primary,3)
    if wall_type==421:
        set_bctype(primary,'1;1;421,0.d0;421,0.d0;1;1')
    elif wall_type!=41:
        set_bctype(primary,f'1,1,{wall_type},{wall_type},1,1')
    if normal_axis!=2:
        entries=['1']*6
        value='41,1.d0' if wall_type==41 else str(wall_type)
        entries[2*normal_axis-2:2*normal_axis]=[value,value]
        set_bctype(primary,';'.join(entries));set_flowtype(primary,'tgv')
    if flow_type is not None:set_flowtype(primary,flow_type)
    set_restart(primary,'f');set_gridfile(primary,'datin/grid.h5')
    write_controller(datin/'controller',steps-1,1,dt,feqlist=1)
    producer=initial_fields if initial_condition is None else initial_condition
    coordinates,fields=producer(grid,manufactured,wall_type)
    if normal_axis!=2:
        coordinates=tuple(coordinates[p].transpose(permutation) for p in permutation)
        rotated={name:fields[name].transpose(permutation) for name in ('ro','t')}
        for component,p in enumerate(permutation,1):
            rotated[f'u{component}']=fields[f'u{p+1}'].transpose(permutation)
        fields=rotated
    with h5py.File(datin/'grid.h5','w') as f:
        for name,value in zip('xyz',coordinates):
            f[name]=value.transpose(2,1,0)
    with h5py.File(datin/'flowini3d.h5','w') as f:
        for name,value in fields.items():
            f[name]=value.transpose(2,1,0)
    (datin/'input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',exact_restart=t,
 host_budget_bytes=134217728,device_budget_bytes=134217728,buffer_bytes=1048576,device_reserve_bytes=0
/
&checkpoint
 enabled=t,interval_steps={checkpoint_interval},keep=2,initial_frame=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")


@pytest.fixture(scope='module')
def launch(tmp_path_factory):
    root=tmp_path_factory.mktemp('hb1b_channel')
    completed={}
    def run(backend,topology=(1,1,1),steps=12,restore=None,mode='hermite7',transport='pinned',
            grid=GRID,manufactured=False,wall_samples=False,sanitizer=False,reject=None,dt=DT,checkpoint_interval=1,wall_type=41,
            normal_axis=2,gradients=False,flow_type=None,initial_condition=None,filtering=False,workspace='scalar',dump_validation=True,force=0.):
        key=(backend,topology,steps,str(restore),mode,transport,grid,manufactured,wall_samples,sanitizer,reject,dt,checkpoint_interval,wall_type,normal_axis,gradients,flow_type,initial_condition,filtering,workspace,dump_validation,force)
        if key in completed:
            return completed[key]
        wall_label='' if wall_type==41 else f'_bc{wall_type}'
        if normal_axis!=2:wall_label+=f'_axis{normal_axis}'
        if flow_type is not None:wall_label+='_'+flow_type
        if filtering:wall_label+='_f6loc1_'+workspace
        if not dump_validation:wall_label+='_no_validation'
        case=root/f'{backend}_{"x".join(map(str,topology))}_{steps}_{mode}{wall_label}_{len(completed)}'
        prepare(case,backend,steps,restore,grid,manufactured,dt,checkpoint_interval,wall_type,normal_axis,flow_type,initial_condition,filtering)
        env=runtime(','.join(map(str,topology)),transport)
        for name in tuple(env):
            if name.startswith('ASTR_INSITU_') or name.startswith('ASTR_WALL_BLOWING_'):
                env.pop(name,None)
        env.update(ASTR_HIGH_ORDER_WALL_MODE=mode,ASTR_GPU_MP_POSITIVITY='off',
            ASTR_CHANNEL_FORCE_MODE='fixed',ASTR_CHANNEL_FORCE_FIXED=str(force),
            ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',ASTR_VALIDATION_RHS_STEP='0',
            ASTR_PERFECT_GAS_PRANDTL='0.72',ASTR_GPU_FILTER_WORKSPACE=workspace)
        if not dump_validation:env.pop('ASTR_VALIDATION_RHS_PREFIX',None)
        if gradients:env['ASTR_VALIDATION_GRADIENTS']='1'
        if wall_samples:
            env['ASTR_INSITU_SAMPLE_PREFIX']='diagnostics/wall'
            if backend=='gpu' and wall_samples is True:
                env['ASTR_INSITU_TEST_DEVICE_TRANSPORT']='pinned'
        application=[str(Path(EXE).resolve()),'run','datin/input.chl']
        if sanitizer:
            application=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
                '--tool','memcheck','--error-exitcode','77','--log-file','memcheck.%p.log']+application
        command=mpi_command(MPIEXEC,int(np.prod(topology)),transport)+application
        with (case/'run.log').open('w') as log:
            result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        text=(case/'run.log').read_text()
        if reject:
            assert result.returncode!=0 and reject in text,text[-12000:]
            return case,None
        assert result.returncode==0 and 'The job is done!' in text,text[-12000:]
        if sanitizer:
            reports=list(case.glob('memcheck.*.log'))
            assert len(reports)==int(np.prod(topology))
            for report in reports:
                assert 'ERROR SUMMARY: 0 errors' in report.read_text(),report.read_text()
        if mode=='hermite7':
            if filtering:assert 'high-order wall filter: local-f6-v1' in text
            identity={41:'channel-wall-h7-v1',42:'channel-adiabatic-p6t7-v1',421:'channel-slip-p6t7-v1'}[wall_type]
            if normal_axis==2 and flow_type!='tgv':assert identity in text
            else:assert f'Cartesian Hermite7 wall axis={normal_axis}' in text
        checkpoints=case/'outdat/new/checkpoints'
        final=checkpoints/f'step{steps:012d}'
        assert (final/'COMPLETE').is_file()
        with h5py.File(final/'state.h5') as f:
            for n in range(1,12):
                assert np.isfinite(f[f'q{n:04d}'][:]).all()
            assert np.min(f['q0001'][:])>0 and np.min(f['q0011'][:])>0
            if mode=='hermite7':
                for j in (0,grid[1]):
                    for n in ((3,) if wall_type==421 else (2,3,4)):
                        assert np.max(np.abs(np.take(f[f'q{n:04d}'][:],j,axis=3-normal_axis)))==0
                    if wall_type==41:
                        assert np.max(np.abs(np.take(f['q0011'][:],j,axis=3-normal_axis)-1))<2e-14
        completed[key]=(case,final)
        return case,final
    return run


@pytest.mark.parametrize('backend',['cpu','gpu'])
@pytest.mark.parametrize('mode',['off','hermite7'])
def test_channel_first_stage_local_mechanical_work(launch,backend,mode,record_property):
    """Use actual first-stage RHS snapshots; the zero-force run is the spatial baseline."""
    plain,_=launch(backend,steps=1,force=0.,mode=mode)
    driven,_=launch(backend,steps=1,force=.003,mode=mode)
    suffix='step00000000.rk01.rank00000000.bin'
    state=physical_state(driven/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')
    with h5py.File(driven/'outdat/new/resources/geometry.h5') as f:
        jacobian=f['q0004'][:].transpose(2,1,0)
    if mode=='off':
        # Legacy CPU "full" precedes sources; compare the actual first Euler stage instead.
        delta=(physical_state(driven/'diagnostics'/f'state.post_update.{suffix}','fp64')-
               physical_state(plain/'diagnostics'/f'state.post_update.{suffix}','fp64'))*jacobian[...,None]/DT
    else:
        delta=rhs_state(driven/'diagnostics'/f'state.full.{suffix}','fp64')-rhs_state(
            plain/'diagnostics'/f'state.full.{suffix}','fp64')
    expected=np.zeros_like(delta)
    expected[...,1]=.003*jacobian
    expected[...,4]=.003*(state[...,1]/state[...,0])*jacobian
    # Wall momentum is constrained separately by boucon; use strict physical interior.
    error=float(np.max(np.abs((delta-expected)[:,1:-1,:,:])))
    record_property('source_maximum_absolute_error',error)
    assert error<=2e-10,(backend,error)


def test_local_work_device_memory_safety(launch):
    launch('gpu',steps=4,force=.003,filtering=True,sanitizer=True)


def test_channel_restart_rejects_previous_bulk_work_model(launch):
    _,part=launch('cpu',steps=1,force=.003)
    old=part.parent/'previous_bulk_work'
    shutil.copytree(part,old)
    control=old/'control.bin'
    payload=control.read_bytes()
    marker=b'-localwork-v1'
    assert payload.count(marker)==1
    # Preserve the fixed-width control record, changing only the source identity.
    identity=b'channel-wall-h7-v1-localwork-v1'
    old_identity=b'channel-wall-h7-v1'
    assert identity in payload
    control.write_bytes(payload.replace(identity,old_identity.ljust(len(identity),b' ')))
    launch('cpu',steps=2,restore=old,force=.003,reject='boundary/driver configuration mismatch')


def analytic_normal_rhs(y,wall_type=41):
    """Independent continuous perfect-gas CNS divergence; no time integration."""
    import sympy as sp
    s=sp.Symbol('y',real=True)
    rho=1+sp.Rational(1,100)*sp.cos(sp.pi*s)
    u=sp.sin(sp.pi*s)/10;v=sp.sin(2*sp.pi*s)/100;w=sp.sin(sp.pi*s)/50
    if wall_type==421:
        u=sp.Rational(1,10)+sp.cos(sp.pi*s)/50;w=sp.cos(2*sp.pi*s)/50
    t=1+sp.sin(sp.pi*s)**2/50
    const2=sp.Rational(126,1000);const5=sp.Rational(36,1000);const6=sp.Rational(5,2)
    p=rho*t/const2;e=const6*p+rho*(u*u+v*v+w*w)/2
    st=sp.Rational(11030,27315)
    mu=t*sp.sqrt(t)*(1+st)/(t+st)/3000
    tau=[mu*sp.diff(u,s),sp.Rational(4,3)*mu*sp.diff(v,s),mu*sp.diff(w,s)]
    conv=[rho*v,rho*u*v,rho*v*v+p,rho*w*v,(e+p)*v]
    visc=[sp.Integer(0),*tau,u*tau[0]+v*tau[1]+w*tau[2]+mu/sp.Rational(72,100)/const5*sp.diff(t,s)]
    result=[]
    for c in range(5):
        function=sp.lambdify(s,-sp.diff(conv[c],s)+sp.diff(visc[c],s),'numpy')
        result.append(np.broadcast_to(function(y),y.shape).copy())
    rhs=np.stack(result,axis=-1)
    # The static wall constrains momentum and E/rho; only density is advanced.
    if wall_type==421:
        rhs[[0,-1],2]=0
    else:
        rhs[[0,-1],1:4]=0
    if wall_type==41:
        rhs[[0,-1],4]=rhs[[0,-1],0]*float(const6/const2)
    return rhs


def test_actual_rhs_spatial_convergence(launch,record_property):
    results=[]
    for n in (32,64,128):
        row={'intervals':n}
        exact=analytic_normal_rhs(np.linspace(0,1,n+1))
        strip=np.r_[0:8,n-7:n+1]
        backend_rhs={}
        for backend in ('cpu','gpu'):
            case,_=launch(backend,steps=1,grid=(8,n,8),manufactured=True)
            suffix='step00000000.rk01.rank00000000.bin'
            weighted=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
            with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
                jacobian=f['q0004'][:].transpose(2,1,0)
            rhs=weighted/jacobian[...,None]
            backend_rhs[backend]=rhs
            row[backend]=np.max(np.abs(rhs[0,strip,0,:]-exact[strip]),axis=0).tolist()
        row['backend_scaled']=float(np.max(np.abs(backend_rhs['cpu']-backend_rhs['gpu'])/
            (1+np.abs(backend_rhs['cpu']))))
        assert row['backend_scaled']<=2e-10,row
        results.append(row)
    orders={backend:np.log2(np.array([r[backend] for r in results])[:-1]/
        np.array([r[backend] for r in results])[1:]).tolist() for backend in ('cpu','gpu')}
    (case.parent/'spatial_convergence.json').write_text(json.dumps(dict(samples=results,orders=orders),indent=2)+'\n')
    record_property('spatial_convergence',json.dumps(dict(samples=results,orders=orders)))
    for backend in ('cpu','gpu'):
        assert np.min(orders[backend])>=5.5,(backend,results,orders)


@pytest.mark.parametrize('topology,mode',[( (1,1,1),'hermite7'),((1,2,1),'hermite7'),((1,1,1),'off')])
def test_completed_wall_diagnostics(launch,topology,mode,record_property):
    check_completed_wall_diagnostics(launch,topology,mode,record_property)


def check_completed_wall_diagnostics(launch,topology,mode,record_property,wall_type=41):
    from test_high_order_boundary_closures import derivative_rows
    weights=np.array(derivative_rows(True)[0],float)
    if mode=='off':
        weights=np.array([-1.5,2,-.5])
    results={}
    for backend in ('cpu','gpu'):
        case,checkpoint=launch(backend,topology,steps=2,mode=mode,wall_type=wall_type,
            wall_samples=True if mode=='hermite7' else 'host-only')
        with h5py.File(checkpoint/'state.h5') as f:
            q=np.stack([f[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,6)],axis=-1)
        error=0
        for path in sorted((case/'diagnostics').glob('wall.wall.step00000002.rank*.bin')):
            with path.open('rb') as f:
                assert f.read(8)==b'ASTRIW01'
                header=np.fromfile(f,'<i4',9);clock=np.fromfile(f,'<f8',2)
                shape=tuple(header[3:6]);nodes=int(np.prod(shape))
                xyz=np.fromfile(f,'<f8',nodes*3).reshape((*shape,3),order='F')
                fields=np.fromfile(f,'<f8',nodes*4).reshape((*shape,4),order='F')
                owned=np.fromfile(f,'<i4',nodes).reshape(shape,order='F')
                assert not f.read(1)
            np.testing.assert_allclose(clock,[2*DT,DT],rtol=0,atol=1e-18)
            assert np.isfinite(xyz).all() and np.isfinite(fields).all()
            for i,k,w in np.ndindex(shape):
                side=int(fields[i,k,w,3]);j=0 if side==1 else GRID[1]
                gi=(i+header[6])%GRID[0];gk=(k+header[8])%GRID[2]
                line=q[gi,j+side*np.arange(len(weights)),gk,:]
                rho=line[:,0];vel=line[:,1:4]/rho[:,None]
                p=(line[:,4]-.5*rho*np.sum(vel**2,axis=1))/2.5
                t=p/rho*.126
                mu=t[0]*np.sqrt(t[0])*(1+110.3/273.15)/(t[0]+110.3/273.15)/3000
                du=weights@vel[:,0]*GRID[1];dt=weights@t*GRID[1]
                expected=np.array([p[0],mu*du,-mu/.72/.036*dt,side])
                error=max(error,float(np.max(abs(fields[i,k,w]-expected))))
            results[(backend,int(header[2]))]=fields
        assert error<=2e-10,(backend,error)
        assert len([k for k in results if k[0]==backend])==int(np.prod(topology))
        record_property(backend+'_wall_oracle_maxabs',error)
    for rank in range(int(np.prod(topology))):
        np.testing.assert_allclose(results['cpu',rank],results['gpu',rank],rtol=0,atol=2e-10)


@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1),(1,1,2),(2,2,1)])
def test_same_stage_fields_and_statistics(launch,topology,record_property):
    cpu,a=launch('cpu',topology);gpu,b=launch('gpu',topology)
    errors={}
    with h5py.File(a/'state.h5') as x,h5py.File(b/'state.h5') as y:
        for name in [f'q{i:04d}' for i in range(1,12)]:
            p,q=x[name][:],y[name][:]
            errors[name]=float(np.max(np.abs(p-q)/(1+np.abs(p))))
            assert errors[name]<=2e-10,(topology,name,errors[name])
        for rank,(p,q) in enumerate(zip(defined_extras(x),defined_extras(y),strict=True)):
            assert np.isfinite(p).all() and np.isfinite(q).all()
            errors[f'defined_extras_rank{rank}']=float(np.max(np.abs(p-q)/(1+np.abs(p))))
            assert errors[f'defined_extras_rank{rank}']<=2e-10,errors
    for rank in range(int(np.prod(topology))):
        suffix=f'step00000000.rk01.rank{rank:08d}.bin'
        for phase in ('conv','full'):
            p=rhs_state(cpu/'diagnostics'/f'state.{phase}.{suffix}','fp64')
            q=rhs_state(gpu/'diagnostics'/f'state.{phase}.{suffix}','fp64')
            errors[f'{phase}_rank{rank}']=float(np.max(np.abs(p-q)/(1+np.abs(p))))
            assert errors[f'{phase}_rank{rank}']<=2e-10,errors
    ha,sa=read_flowstate(cpu);hb,sb=read_flowstate(gpu)
    assert ha==hb and sa.shape==sb.shape
    assert np.max(np.abs(sa-sb)/(1+np.abs(sa)))<=2e-10
    (gpu.parent/f'comparison_{"x".join(map(str,topology))}.json').write_text(json.dumps(errors,indent=2)+'\n')
    record_property('maximum_scaled_field_difference',max(errors.values()))


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_density_obeys_first_rk_continuity(launch,backend):
    case,_=launch(backend)
    suffix='step00000000.rk01.rank00000000.bin'
    before=physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')
    after=physical_state(case/'diagnostics'/f'state.post_update.{suffix}','fp64')
    rhs=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')
    with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
        # q0004 is the stored Jacobian after the x/y/z coordinate components.
        jacobian=f['q0004'][:].transpose(2,1,0)
    for j in (0,GRID[1]):
        expected=before[:,j,:,0]+DT*rhs[:,j,:,0]/jacobian[:,j,:]
        assert np.max(np.abs(after[:,j,:,0]-expected))<5e-13
        assert np.max(np.abs(after[:,j,:,0]-before[:,j,:,0]))>1e-8


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_same_topology_exact_restart(launch,backend):
    _,whole=launch(backend,(2,1,1),12)
    _,part=launch(backend,(2,1,1),5)
    _,resumed=launch(backend,(2,1,1),12,restore=part)
    compare_fields(whole/'state.h5',resumed/'state.h5')
    assert (whole/'control.bin').read_bytes()==(resumed/'control.bin').read_bytes()


def test_restart_rejects_wall_algorithm_change(launch):
    _,part=launch('cpu',(2,1,1),5)
    launch('cpu',(2,1,1),12,restore=part,mode='off',reject='boundary/driver configuration mismatch')


def test_device_wall_memory_safety(launch):
    launch('gpu',(1,2,1),steps=2,wall_samples=True,sanitizer=True)


@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_wall_diagnostics_do_not_change_advancement(launch,backend):
    _,plain=launch(backend,steps=2)
    _,sampled=launch(backend,steps=2,wall_samples=True)
    compare_fields(plain/'state.h5',sampled/'state.h5')


def wall_inventory(case,checkpoint):
    """Physical quadrature diagnostics, not a claim of an exact FD conservation norm."""
    from test_high_order_explicit_wall_candidates import pressure_hermite_ghosts
    from test_high_order_boundary_closures import derivative_matrix
    from scipy.integrate import simpson,romb,newton_cotes
    with h5py.File(case/'datin/flowini3d.h5') as f:
        initial={name:f[name][:].transpose(2,1,0) for name in ('ro','u1','u2','u3','t')}
    with h5py.File(checkpoint/'state.h5') as f:
        q=np.stack([f[f'q{m:04d}'][:].transpose(2,1,0) for m in range(1,6)],axis=-1)
    def integral(field):
        return float(np.trapezoid(field[:-1,:,:-1],dx=1/GRID[1],axis=1).sum()*
            (2*np.pi/GRID[0])*(2*np.pi/GRID[2]))
    rho=initial['ro']
    energy=rho*initial['t']/.126*2.5+.5*rho*sum(initial[name]**2 for name in ('u1','u2','u3'))
    pressure=(q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0])/2.5
    weights,_=pressure_hermite_ghosts()
    minimum_ghost=np.inf
    for side,j in ((1,0),(-1,GRID[1])):
        line=pressure[:,j+side*np.arange(7),:]
        ghost=np.einsum('gj,ijk->igk',weights[:,1:],line[:,1:,:]-line[:,0:1,:])+line[:,0:1,:]
        assert np.isfinite(ghost).all() and np.min(ghost)>0
        minimum_ghost=min(minimum_ghost,float(np.min(ghost)))
    mass0=integral(rho);energy0=integral(energy)
    mass=integral(q[...,0]);total_energy=integral(q[...,4])
    # Construct from D alone, without fitting weights to the flow or mass drift.
    derivative=GRID[1]*derivative_matrix(GRID[1],True)
    boundary=np.zeros(GRID[1]+1);boundary[0]=-1;boundary[-1]=1
    norm=np.linalg.lstsq(derivative.T,boundary,rcond=None)[0]
    delta=np.mean((q[...,0]-rho)[:-1,:,:-1],axis=(0,2))*4*np.pi**2
    nc,_=newton_cotes(8)
    bmatrix=np.diag(boundary)
    quadratures=dict(trapezoidal=float(np.trapezoid(delta,dx=1/GRID[1])),
        simpson=float(simpson(delta,dx=1/GRID[1])),romberg=float(romb(delta,dx=1/GRID[1])),
        composite_newton_cotes_8=float(sum(nc@delta[j:j+9]/GRID[1] for j in range(0,GRID[1],8))))
    compatible=dict(construction='minimum Euclidean norm solution of D.T*w=[-1,0,...,1]; no flow data',
        minimum_weight=float(np.min(norm)),maximum_weight=float(np.max(norm)),sum_weights=float(norm.sum()),
        column_sum_residual=float(np.max(np.abs(derivative.T@norm-boundary))),
        mass_change=float(norm@delta),relative_mass_change=float(norm@delta/mass0),
        diagonal_sbp_defect=float(np.linalg.norm(norm[:,None]*derivative+derivative.T*norm[None,:]-bmatrix,ord=np.inf)),
        note='Column-sum mass diagnostic only, not a diagonal-SBP or nonlinear energy-stability proof.')
    return dict(quadrature='unique x/z periodic nodes; trapezoidal y; physical uniform coordinates',
        initial_mass=mass0,final_mass=mass,mass_change=mass-mass0,relative_mass_change=(mass-mass0)/mass0,
        initial_energy=energy0,final_energy=total_energy,energy_change=total_energy-energy0,
        minimum_private_pressure_ghost=minimum_ghost,
        mass_change_by_quadrature=quadratures,operator_compatible_mass=compatible,
        note='Fixed-temperature walls exchange heat; energy change alone is not a balance residual.')


def first_step_energy_balance(case,dt):
    """Account for the thermostat reaction separately from the boundary heat flux."""
    from test_high_order_boundary_closures import derivative_matrix
    dy=GRID[1]*derivative_matrix(GRID[1],True)
    boundary=np.zeros(GRID[1]+1);boundary[0]=-1;boundary[-1]=1
    weights=np.linalg.lstsq(dy.T,boundary,rcond=None)[0]
    def normal_derivative(field):
        return np.einsum('ij,xjz->xiz',dy,field)
    def periodic_derivative(field,axis):
        result=np.zeros_like(field)
        for offset,c in ((1,.75),(2,-.15),(3,1/60)):
            result+=c*(np.roll(field,-offset,axis=axis)-np.roll(field,offset,axis=axis))
        return result*GRID[axis]/(2*np.pi)
    def integral(field):
        return float(4*np.pi**2*(weights@np.mean(field,axis=(0,2))))
    stages=[]
    for stage in (1,2,3):
        suffix=f'step00000000.rk{stage:02d}.rank00000000.bin'
        q=physical_state(case/'diagnostics'/f'state.pre_rhs.{suffix}','fp64')[:-1,:,:-1,:]
        rhs=rhs_state(case/'diagnostics'/f'state.full.{suffix}','fp64')[:-1,:,:-1,:]
        with h5py.File(case/'outdat/new/resources/geometry.h5') as f:
            jacobian=f['q0004'][:].transpose(2,1,0)[:-1,:,:-1]
        rhs=rhs/jacobian[...,None]
        rho=q[...,0];vel=q[...,1:4]/rho[...,None]
        p=(q[...,4]-.5*rho*np.sum(vel**2,axis=-1))/2.5
        t=p/rho*.126
        mu=t*np.sqrt(t)*(1+110.3/273.15)/(t+110.3/273.15)/3000
        gradient=np.stack([np.stack([periodic_derivative(vel[...,c],0),normal_derivative(vel[...,c]),
            periodic_derivative(vel[...,c],2)],axis=-1) for c in range(3)],axis=-2)
        divergence=np.trace(gradient,axis1=-2,axis2=-1)
        tau_y=mu[...,None]*(gradient[...,1]+gradient[...,1,:])
        tau_y[...,1]-=(2/3)*mu*divergence
        conductive=mu/.72/.036*normal_derivative(t)
        flux=np.sum(vel*tau_y,axis=-1)+conductive-(q[...,4]+p)*vel[...,1]
        normal_rhs=normal_derivative(flux)
        boundary_flux=float(4*np.pi**2*np.mean(flux[:,-1,:]-flux[:,0,:]))
        reaction=float(4*np.pi**2*sum(weights[j]*np.mean(rhs[:,j,:,4]-normal_rhs[:,j,:]) for j in (0,GRID[1])))
        rate=integral(rhs[...,4])
        stages.append(dict(stage=stage,energy_rhs_rate=rate,boundary_flux_rate=boundary_flux,
            wall_constraint_reaction_rate=reaction,rate_reconciliation=rate-boundary_flux-reaction))
        if stage==1:
            initial=integral(q[...,4])
        if stage==3:
            end=physical_state(case/'diagnostics'/f'state.post_update.{suffix}','fp64')[:-1,:,:-1,4]
            final=integral(end)
    rkweights=np.array([1/6,1/6,2/3])
    integrated=lambda name:float(dt*(rkweights@np.array([s[name] for s in stages])))
    boundary_heat=integrated('boundary_flux_rate');reaction=integrated('wall_constraint_reaction_rate')
    return dict(window='first complete SSP-RK3 step only; later-stage snapshots were not saved',stages=stages,
        energy_change=final-initial,boundary_flux_integral=boundary_heat,wall_constraint_reaction_integral=reaction,
        balance_residual=final-initial-boundary_heat-reaction,
        note='Discrete accounting with the derivative-compatible quadrature, not a continuum energy-stability proof.')


@pytest.mark.skipif(os.environ.get('ASTR_HB1B_STABILITY')!='1',reason='Explicitly opt in to the approved 2000-step window')
def test_nonlinear_stability_window(launch,record_property):
    cases={};summary={}
    for backend in ('cpu','gpu'):
        case,checkpoint=launch(backend,steps=2000,dt=1e-3,checkpoint_interval=2000)
        values=[float(v) for v in re.findall(r'current CFL:\s*([0-9.Ee+-]+)',(case/'run.log').read_text())]
        assert len(values)==2000 and np.isfinite(values).all() and max(values)<=.2,(backend,len(values),max(values))
        assert len(list((case/'outdat/new/checkpoints').glob('step*')))==1
        summary[backend]=dict(maximum_cfl=max(values),**wall_inventory(case,checkpoint),
            first_step_discrete_energy_balance=first_step_energy_balance(case,1e-3))
        cases[backend]=(case,checkpoint)
    errors={}
    with h5py.File(cases['cpu'][1]/'state.h5') as a,h5py.File(cases['gpu'][1]/'state.h5') as b:
        for name in [f'q{m:04d}' for m in range(1,12)]:
            x,y=a[name][:],b[name][:]
            errors[name]=float(np.max(np.abs(x-y)/(1+np.abs(x))))
        for rank,(x,y) in enumerate(zip(defined_extras(a),defined_extras(b),strict=True)):
            assert np.isfinite(x).all() and np.isfinite(y).all()
            errors[f'defined_extras_rank{rank}']=float(np.max(np.abs(x-y)/(1+np.abs(x))))
    summary['backend_errors']=errors
    (case.parent/'nonlinear_stability.json').write_text(json.dumps(summary,indent=2)+'\n')
    record_property('nonlinear_stability',json.dumps(summary))
    assert max(errors.values())<=2e-10,summary
