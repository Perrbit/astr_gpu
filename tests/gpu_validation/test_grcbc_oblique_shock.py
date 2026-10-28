"""Euler oblique-shock fixture and diagnostics; ASTR performs every time step."""
import json
import math
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from prepare_s1_flatplate_case import write_input,write_controller
from run_m12_mp7_nscbc_performance import runtime,mpi_command
from run_output_restart_validation import compare_fields
from test_sbli_joint_nscbc import compare_cpu_gpu,fields,MPIEXEC

EXE=os.environ.get('ASTR_SBLI_GRCBC_EXE')
solver=pytest.mark.skipif(not EXE,reason='Set ASTR_SBLI_GRCBC_EXE to a root-CMake CPU/GPU executable')
GAMMA=1.4
MACH=2.
BETA=math.radians(40.)
ENTRY=.25
DT=2.5e-4


def analytic_states():
    mn2=(MACH*math.sin(BETA))**2
    ratio=(GAMMA+1)*mn2/((GAMMA-1)*mn2+2)
    pressure=1/(GAMMA*MACH*MACH)
    post_pressure=pressure*(2*GAMMA*mn2-(GAMMA-1))/(GAMMA+1)
    u=1+(1/ratio-1)*math.sin(BETA)**2
    v=(1/ratio-1)*math.sin(BETA)*math.cos(BETA)
    # Primitive order: rho,u,v,w,T; pressure is recovered from rho*T/(gamma*Ma^2).
    pre=np.array([1.,1.,0.,0.,1.])
    post=np.array([ratio,u,v,0.,post_pressure/pressure/ratio])
    return pre,post


def prepare(case,backend,n,steps,initial='shock',restore=None,checkpoint_interval=None,
            background_v=0.,entry_x=ENTRY):
    datin=case/'datin'
    datin.mkdir(parents=True)
    z,y,x=np.meshgrid(np.linspace(0.,.125,9),np.linspace(0.,1.,n+1),np.linspace(0.,1.,n+1),indexing='ij')
    with h5py.File(datin/'grid.flatplate.h5','w') as file:
        for name,value in [('x',x),('y',y),('z',z)]:
            file[name]=value
    pre,post=analytic_states()
    pre[2]=background_v
    shocked=(x-entry_x)*math.sin(BETA)+(y-1)*math.cos(BETA)>0
    if initial in ('uniform','entropy'):
        shocked[:]=False
    elif initial!='shock':
        raise ValueError(initial)
    with h5py.File(datin/'flowini3d.h5','w') as file:
        if initial=='entropy':
            density=1.+.01*np.sin(2*np.pi*x)
            for name,value in [('ro',density),('u1',np.ones_like(x)),
                               ('u2',np.full_like(x,background_v)),('u3',np.zeros_like(x)),('t',1./density)]:
                file[name]=value
        else:
            for j,name in enumerate(('ro','u1','u2','u3','t')):
                file[name]=np.where(shocked,post[j],pre[j])
    with (datin/'inlet.prof').open('w') as file:
        file.write('Uniform Euler inflow density=provided\nrho u v temperature\n1 1 1 1\ny rho u v temperature\n')
        for _ in range(n+1):
            file.write(f'1.0000000000000000e+00 1.0000000000000000e+00 {background_v:.16e} 1.0000000000000000e+00\n')
    write_input(datin/'input.flatplate',n,n,8,'t' if backend=='gpu' else 'f',
        'f','f','743e','t',.001,1.e6,MACH,288.,1.,52,11,3,0,'prof',1,
        x_max_bctype=22,outlet_pressure=post[0]*post[4],lower_bctype=421,xslip=-1.)
    write_controller(datin/'controller',steps-1,10,DT,feqlist=max(1,min(100,steps//10)))
    (case/'outdat/new').mkdir(parents=True)
    checkpoint_interval=steps if checkpoint_interval is None else checkpoint_interval
    output_mode='override' if restore else 'saved'
    (datin/'input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{restore or ''}',restart_output='{output_mode}',exact_restart=t,
 host_budget_bytes=268435456,
 device_budget_bytes=268435456,buffer_bytes=4194304,device_reserve_bytes=0
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
    (case/'parameters.json').write_text(json.dumps(dict(mach=MACH,gamma=GAMMA,beta_deg=40.,
        entry_x=entry_x,background_v=background_v,intervals=[n,n,8],steps=steps,dt=DT,initial=initial,
        pre=pre.tolist(),post=post.tolist(),legacy_xslip=-1.,
        checkpoint_interval=checkpoint_interval,output_mode=output_mode,
        nominal_bottom_crossing=ENTRY+1/math.tan(BETA)),indent=2))


def launch(case,backend,n=64,steps=12,topology=(1,1,1),initial='shock',memcheck=False,restore=None,
           checkpoint_interval=None,background_v=0.,entry_x=ENTRY,field_source=None,diagnostics=False,
           sanitizer_tool='memcheck'):
    prepare(case,backend,n,steps,initial,restore,checkpoint_interval,background_v,entry_x)
    if field_source is not None:
        with h5py.File(field_source) as file:
            q=np.stack([file[f'q{i:04d}'][...] for i in range(1,6)],axis=-1)
        assert q.shape==(9,n+1,n+1,5)
        rho=q[...,0];velocity=q[...,1:4]/rho[...,None]
        temperature=.4*(q[...,4]-.5*rho*np.sum(velocity**2,axis=-1))*5.6/rho
        with h5py.File(case/'datin/flowini3d.h5','w') as file:
            for name,value in [('ro',rho),('u1',velocity[...,0]),('u2',velocity[...,1]),
                               ('u3',velocity[...,2]),('t',temperature)]:
                file[name]=value
    env=runtime(','.join(map(str,topology)))
    env.pop('ASTR_WALL_BLOWING_MODE',None)
    env.update(ASTR_GPU_MP_POSITIVITY='off',ASTR_GPU_MP_SPLIT_CACHE='off',
        ASTR_NSCBC_FARFIELD_MODE='sbli_grcbc',ASTR_NSCBC_FARFIELD_SHOCK_X=str(entry_x),
        ASTR_NSCBC_FARFIELD_SHOCK_ANGLE_DEG='40',ASTR_NSCBC_FARFIELD_RHO='1',
        ASTR_NSCBC_FARFIELD_U='1',ASTR_NSCBC_FARFIELD_V=str(background_v),ASTR_NSCBC_FARFIELD_W='0',
        ASTR_NSCBC_FARFIELD_T='1')
    if diagnostics:
        (case/'diagnostics').mkdir()
        env.update(ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',ASTR_VALIDATION_RHS_STEP='0')
    command=mpi_command(MPIEXEC,int(np.prod(topology)),'pinned')
    if memcheck:
        command+=['compute-sanitizer','--tool',sanitizer_tool,'--error-exitcode','99']
    command += [str(Path(EXE).resolve()),'run','datin/input.flatplate']
    with (case/'run.log').open('w') as log:
        result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    text=(case/'run.log').read_text()
    assert result.returncode==0 and 'The job is done!' in text,text[-10000:]
    if memcheck:
        assert text.count('ERROR SUMMARY: 0 errors')==int(np.prod(topology)),text[-10000:]
    fields(case,steps)
    return case


def diagnose(case,steps,report_path=None):
    state=fields(case,steps)
    rho=state['rho'].mean(axis=0)
    p=state['pressure'].mean(axis=0)*(GAMMA*MACH*MACH)
    vel=state['velocity'].mean(axis=0)
    temperature=state['temperature'].mean(axis=0)
    pre,post=analytic_states()
    n=rho.shape[1]-1
    line=np.linspace(0.,1.,n+1)
    x,y=np.meshgrid(line,line)
    h=1/n
    phi=(x-ENTRY)*math.sin(BETA)+(y-1)*math.cos(BETA)
    fraction=(p-1)/(post[0]*post[4]-1)
    x_cross=[]
    y_cross=[]
    ambiguous=0
    for j in np.flatnonzero((line>=.55)&(line<=.8)):
        indices=np.flatnonzero((fraction[j,:-1]<.5)&(fraction[j,1:]>=.5))
        indices=indices[(line[indices]>.3)&(line[indices+1]<.95)]
        if len(indices)!=1:
            ambiguous+=1
            continue
        i=int(indices[0])
        x_cross.append(float(line[i]+h*(.5-fraction[j,i])/(fraction[j,i+1]-fraction[j,i])))
        y_cross.append(float(line[j]))
    if len(x_cross)<3:
        raise ValueError('No single resolved oblique shock for angle measurement')
    slope,intercept=np.polyfit(y_cross,x_cross,1)
    if slope>=0:
        raise ValueError('Shock does not descend from the upper boundary')
    fitted_beta=math.degrees(math.atan(-1/slope))
    exact_x=ENTRY+(1-np.array(y_cross))/math.tan(BETA)
    primitive=np.stack([rho,vel[...,0],vel[...,1],vel[...,2],temperature,p],axis=-1)
    primitive_all=np.stack([state['rho'],state['velocity'][...,0],state['velocity'][...,1],
                            state['velocity'][...,2],state['temperature'],state['pressure']*5.6],axis=-1)
    plateau={}
    for name,target,side in [('pre',pre,phi<0),('post',post,phi>0)]:
        mask=side&(np.abs(phi)>6*h)&(x>.35)&(x<.85)&(y>.2)&(y<.9)
        if not np.any(mask):
            raise ValueError('Empty plateau sampling region')
        reference=np.r_[target,target[0]*target[4]]
        scale=np.array([target[0],1.,1.,1.,target[4],reference[-1]])
        error=np.abs(primitive_all[:,mask,:]-reference)/scale
        plateau[name]={
            'points':int(mask.sum()),'maximum_scaled_error':float(error.max()),
            'maximum_scaled_errors':dict(zip(('rho','u','v','w','T','pressure_ratio'),error.max(axis=(0,1)).tolist())),
            'mean_scaled_errors':dict(zip(('rho','u','v','w','T','pressure_ratio'),error.mean(axis=(0,1)).tolist()))}
    q_exact=np.where((phi>0)[...,None],post,pre)
    exact=np.stack([q_exact[...,0],q_exact[...,1],q_exact[...,2],q_exact[...,3],q_exact[...,4],
                    q_exact[...,0]*q_exact[...,4]],axis=-1)
    weights=np.ones_like(x)
    weights[:,[0,-1]]*=.5
    weights[[0,-1],:]*=.5
    interior_error=np.sum(np.abs(primitive-exact)*weights[...,None],axis=(0,1))*h*h
    summary={
        'n':n,'steps':steps,'time_nondimensional':steps*DT,
        'expected_beta_deg':40.,'fitted_beta_deg':float(fitted_beta),
        'angle_error_deg':float(abs(fitted_beta-40)),
        'fitted_entry_x':float(slope+intercept),'shock_position_mean_abs_error':float(np.mean(abs(x_cross-exact_x))),
        'ambiguous_rows':ambiguous,'sample_rows':len(x_cross),'plateau':plateau,
        'plateau_scope':'maximum over every spanwise node in the fixed sampling region',
        'primitive_l1_errors':dict(zip(('rho','u','v','w','T','pressure_ratio'),interior_error.tolist())),
        'spanwise_maximum_q_difference':float(np.max(np.abs(state['q']-state['q'][:1])))}
    top_reference=np.where((line>ENTRY)[:,None],np.r_[post,post[0]*post[4]],
                           np.r_[pre,pre[0]*pre[4]])
    top_scale=top_reference.copy()
    top_scale[:,1:4]=1.
    top_error=np.abs(primitive_all[:,-1]-top_reference)/top_scale
    front_crossing=(line<=ENTRY)&((line-ENTRY)*math.sin(BETA)+7*h*math.cos(BETA)>0)
    summary['top_boundary']={
        'scope':'all top nodes including the shock entry; reported, not an extra plateau acceptance gate',
        'maximum_scaled_errors':dict(zip(('rho','u','v','w','T','pressure_ratio'),
                                         top_error.max(axis=(0,1)).tolist())),
        'spanwise_maximum_q_difference':float(np.max(np.abs(state['q'][:,-1]-state['q'][:1,-1]))),
        'reference_sampling_front_crossing_x_indices':np.flatnonzero(front_crossing).tolist(),
        'front_crossing_treatment':'first exterior point, with all incoming same-stage sources retained'}
    destination=case/'oblique_diagnostics.json' if report_path is None else Path(report_path)
    destination.write_text(json.dumps(summary,indent=2))
    return summary


def check_physical_gate(summary):
    assert summary['ambiguous_rows']==0,summary
    assert summary['angle_error_deg']<=1.,summary
    assert max(side['maximum_scaled_error'] for side in summary['plateau'].values())<=.01,summary
    return {'accepted':True,'angle_limit_deg':1.,'maximum_plateau_error_limit':.01,
            'velocity_scale':'Uinf','thermodynamic_scales':'respective upstream/downstream reference'}


def diagnose_spanwise_replay(source,destination,steps=128):
    """Scale an observed perturbation for native ASTR diagnosis, not acceptance."""
    destination.mkdir(parents=True)
    with h5py.File(source) as file:
        original=np.stack([file[f'q{i:04d}'][...] for i in range(1,6)],axis=-1)
    mean=original[:-1].mean(axis=0,keepdims=True)
    n=original.shape[1]-1
    reports=[]
    for amplitude in (1e-4,1e-6):
        initial=mean+amplitude*(original-mean)
        path=destination/f'perturbation_{amplitude:g}.h5'
        with h5py.File(path,'w') as file:
            file.attrs['source']='diagnostic scaled perturbation, not an exact checkpoint'
            for component in range(5):
                file[f'q{component+1:04d}']=initial[...,component]
        case=launch(destination/f'gpu_{amplitude:g}','gpu',n=n,steps=steps,
                    topology=(2,1,1),field_source=path)
        final=fields(case,steps)['q']
        row=dict(perturbation_scale=amplitude,steps=steps,duration=steps*DT)
        for label,q in [('initial',initial),('final',final)]:
            fluctuation=q[:-1]-q[:-1].mean(axis=0,keepdims=True)
            row[label+'_spanwise_linf']=float(np.abs(fluctuation).max())
            row[label+'_top_three_l2']=float(np.linalg.norm(fluctuation[:,-3:]))
        row['top_three_amplification']=row['final_top_three_l2']/row['initial_top_three_l2']
        reports.append(row)
    (destination/'perturbation_response.json').write_text(json.dumps(reports,indent=2)+'\n')
    return reports


def test_analytic_jump_and_domain_placement():
    pre,post=analytic_states()
    normal=np.array([math.sin(BETA),math.cos(BETA),0.])
    tangent=np.array([math.cos(BETA),-math.sin(BETA),0.])
    fluxes=[]
    for primitive in (pre,post):
        rho=primitive[0];u=primitive[1:4];p=rho*primitive[4]/(GAMMA*MACH*MACH)
        energy=p/(GAMMA-1)+.5*rho*np.dot(u,u)
        un=normal@u
        fluxes.append(np.r_[rho*un,rho*u*un+p*normal,(energy+p)*un])
    np.testing.assert_allclose(fluxes[0],fluxes[1],rtol=0,atol=2e-14)
    assert tangent@pre[1:4]==pytest.approx(tangent@post[1:4],abs=2e-14)
    assert ENTRY+1/math.tan(BETA)>1
    assert post[1]>math.sqrt(post[4])/MACH


def test_diagnostics_against_sampled_analytic_field(tmp_path):
    case=tmp_path/'synthetic'
    prepare(case,'gpu',128,1000)
    with h5py.File(case/'datin/flowini3d.h5') as file:
        rho=file['ro'][...]
        vel=np.stack([file[f'u{i}'][...] for i in (1,2,3)],axis=-1)
        temperature=file['t'][...]
    p=rho*temperature/(GAMMA*MACH*MACH)
    q=np.concatenate([rho[...,None],rho[...,None]*vel,
        (p/(GAMMA-1)+.5*rho*np.sum(vel*vel,axis=-1))[...,None]],axis=-1)
    directory=case/'outdat/new/checkpoints/step000000001000'
    directory.mkdir(parents=True)
    with h5py.File(directory/'state.h5','w') as file:
        file.attrs['source']='synthetic analysis unit test, not an ASTR time integration'
        for j in range(5):
            file[f'q{j+1:04d}']=q[...,j]
        for j,value in enumerate((rho,vel[...,0],vel[...,1],vel[...,2],p,temperature),6):
            file[f'q{j:04d}']=value
    summary=diagnose(case,1000)
    assert summary['angle_error_deg']<.2
    assert summary['shock_position_mean_abs_error']<=.5/128
    assert max(side['maximum_scaled_error'] for side in summary['plateau'].values())<2e-14
    assert max(summary['top_boundary']['maximum_scaled_errors'].values())<2e-14
    check_physical_gate(summary)


@solver
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_grazing_entropy_keeps_tangential_transport(tmp_path,backend):
    positive=launch(tmp_path/'positive',backend,steps=2,initial='entropy',background_v=1.e-6,entry_x=2.)
    negative=launch(tmp_path/'negative',backend,steps=2,initial='entropy',background_v=-1.e-6,entry_x=2.)
    rho_positive=float(fields(positive,2)['rho'][:,-1,32].mean())
    rho_negative=float(fields(negative,2)['rho'][:,-1,32].mean())
    expected=1.+.01*math.sin(math.pi-2*math.pi*(2*DT))
    summary=dict(rho_positive=rho_positive,rho_negative=rho_negative,
                 expected_exact_entropy_advection=expected,normal_velocity_magnitude=1.e-6)
    (tmp_path/'grazing_entropy.json').write_text(json.dumps(summary,indent=2))
    assert abs(rho_positive-expected)<2.e-10,summary
    assert abs(rho_negative-rho_positive)<2.e-10,summary


@solver
@pytest.mark.parametrize('topology',[(1,1,1),(2,1,1),(1,2,1)])
def test_same_phase_cpu_gpu(tmp_path,topology):
    cpu=launch(tmp_path/'cpu','cpu',topology=topology)
    gpu=launch(tmp_path/'gpu','gpu',topology=topology)
    errors=compare_cpu_gpu(cpu,gpu)
    (tmp_path/'field_errors.json').write_text(json.dumps(errors,indent=2))


@solver
def test_two_rank_memcheck(tmp_path):
    launch(tmp_path/'gpu','gpu',steps=6,topology=(2,1,1),memcheck=True)


@solver
@pytest.mark.parametrize('backend',['cpu','gpu'])
def test_exact_state_continuation(tmp_path,backend):
    full=launch(tmp_path/'full',backend,topology=(2,1,1))
    part=launch(tmp_path/'part',backend,steps=5,topology=(2,1,1))
    source=part/'outdat/new/checkpoints/step000000000005'
    tail=launch(tmp_path/'tail',backend,topology=(2,1,1),restore=source)
    final=Path('outdat/new/checkpoints/step000000000012/state.h5')
    compare_fields(full/final,tail/final)
    assert b'sbli-grcbc-h1' in (full/final.parent/'control.bin').read_bytes()


@solver
@pytest.mark.parametrize('n',[64,128,192])
def test_seeded_shock_measurement(tmp_path,n):
    case=launch(tmp_path/'gpu','gpu',n=n,steps=4000,topology=(2,1,1))
    summary=diagnose(case,4000)
    if n==192:
        (case/'physical_gate.json').write_text(json.dumps(check_physical_gate(summary),indent=2))


@solver
def test_uniform_start_generation(tmp_path):
    case=launch(tmp_path/'gpu','gpu',n=192,steps=8000,topology=(2,1,1),initial='uniform')
    summary=diagnose(case,8000)
    (case/'physical_gate.json').write_text(json.dumps(check_physical_gate(summary),indent=2))
