"""Low-order KEEP/ME6 profile-boundary same-phase and restart comparisons."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py

from compare_q_validation_snapshots import read_q_snapshot
from run_keep_periodic_gate import compare_q
from run_output_restart_validation import run_case,compare_fields
from prepare_s1_flatplate_case import write_grid
from analyze_keep_nonreflecting_corner import check_corner
from run_adaptive_tgv_gate import STEP
from run_keep_simple_boundary_gate import frozen_rhs,scaled
from generate_dynamic_inflow_slices import FIELD_AMPLITUDES,temporal_amplitude

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('outdat/new/checkpoints/step000000000004')


def adaptive_clocks(case,first=1):
    text=(case/'run.log').read_text()
    rows=np.array([[int(match[1]),*map(float,match.groups()[1:])] for match in STEP.finditer(text)])
    if rows.shape!=(5-first,5) or not np.array_equal(rows[:,0],np.arange(first,5)):
        raise AssertionError('KEEP open automatic clock sequence differs')
    if not np.isfinite(rows).all() or not (rows[:,3:]>0).all():
        raise AssertionError('KEEP open automatic dt is nonfinite or nonpositive')
    if np.max(abs(rows[:,2]-(rows[:,1]+rows[:,3])))>32*np.finfo(float).eps:
        raise AssertionError('KEEP open complete-step time does not use its recorded dt')
    upper=np.array([float(line.split('=',1)[1].split()[-1]) for line in text.splitlines()
                    if line.startswith('ASTR_CFL directional/local_sum/upper_bound=')])
    if len(upper)!=len(rows) or not np.isfinite(upper).all() or not (upper>0).all():
        raise AssertionError('KEEP open automatic clock lacks detailed acoustic CFL records')
    expected=np.minimum(.5*rows[:,3]/upper,1.1*rows[:,3])
    if np.max(abs(expected-rows[:,4]))>64*np.finfo(float).eps:
        raise AssertionError('KEEP open next dt differs from existing CFL/growth rule')
    return dict(records=rows.tolist(),used_dt_c_upper=upper.tolist())


def write_common_initial(out,extruded=False,dimension=None):
    if dimension is None:
        dimension=2 if extruded else 3
    if dimension==0:
        return None
    geometry=out/'initial_grid.h5'
    lz=90. if extruded else .25
    write_grid(geometry,16,16,16,0,10,3,lz,0,0)
    with h5py.File(geometry) as state:
        x,y,z=[state[key][:].transpose(2,1,0) for key in ('x','y','z')]
    profile=1-np.exp(-(y/0.08)**2)
    ramp=1-np.exp(-x/10)
    phase=2*np.pi*z/lz
    temperature=1+1e-4*ramp*profile**2*np.cos(phase)
    fields={'ro':1/temperature,'t':temperature,'u1':profile*(1+1e-4*ramp*np.cos(phase)),
            'u2':1e-4*ramp*profile**2*np.sin(phase),'u3':1e-4*ramp*profile**2*np.cos(phase)}
    if dimension==2:
        fields={name:value[:,:,0] for name,value in fields.items() if name!='u3'}
    path=out/'initial.h5'
    with h5py.File(path,'x') as state:
        for name,value in fields.items():
            state[name]=value.T if dimension==2 else value.transpose(2,1,0)
    return path


def compare_open_rhs(cpu,gpu,out):
    suffix='.step00000000.rk01.rank00000000.bin'
    states=[read_q_snapshot(case/'outdat'/('rhs.pre_rhs'+suffix)) for case in (cpu,gpu)]
    if states[0].header!=states[1].header:
        raise AssertionError('KEEP open input layouts differ')
    im,jm,km,hm,nq=states[0].header
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    # Physical x/y faces and periodic z halo; no physical-exterior x/y data.
    selection=(slice(hm,hm+im+1),slice(hm,hm+jm+1),slice(None))
    q=[state.values.reshape((*shape,nq),order='F')[selection] for state in states]
    if any(not np.isfinite(value).all() for value in q):
        raise AssertionError('nonfinite KEEP open operator input')
    result={'q_max_scaled':float(np.max(abs(q[0]-q[1])/np.maximum(1,np.maximum(abs(q[0]),abs(q[1])))))}
    if result['q_max_scaled']>2e-10:
        raise AssertionError(f'KEEP open boundary states differ: {result}')
    primitives=[]
    for case in (cpu,gpu):
        with (case/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
            header=np.fromfile(stream,np.int32,4)
            values=np.fromfile(stream,np.float64)
        if tuple(header)!=(im,jm,km,hm):
            raise AssertionError('KEEP open primitive layout differs')
        primitives.append(values.reshape((*shape,6),order='F')[selection])
    if any(not np.isfinite(value).all() for value in primitives):
        raise AssertionError('nonfinite KEEP open primitive input')
    result['primitive_max_scaled']=float(np.max(abs(primitives[0]-primitives[1])/np.maximum(1,
        np.maximum(abs(primitives[0]),abs(primitives[1])))))
    if result['primitive_max_scaled']>2e-10:
        raise AssertionError(f'KEEP open primitive states differ: {result}')
    compare_fields(cpu/'outdat/new/resources/geometry.h5',gpu/'outdat/new/resources/geometry.h5')
    for phase in ('boundary','conv','full'):
        arrays=[]
        for case in (cpu,gpu):
            with (case/'outdat'/('rhs.'+phase+suffix)).open('rb') as stream:
                header=np.fromfile(stream,np.int32,4)
                values=np.fromfile(stream,np.float64)
            if tuple(header)!=(im,jm,km,nq):
                raise AssertionError('KEEP open RHS layout differs')
            arrays.append(values.reshape((im+1,jm+1,km+1,nq),order='F'))
        if phase=='boundary':
            arrays[0]=-arrays[0]  # CPU boundary stores positive divergence until rhscal.
        a,b=arrays
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError('nonfinite KEEP open RHS')
        scaled=abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))
        result[phase]={'max_scaled':float(scaled.max()),
                       'index':list(map(int,np.unravel_index(scaled.argmax(),scaled.shape)))}
        if phase=='boundary':
            result[phase]['outlet_top_max_scaled']=float(scaled[im,jm,:,:].max())
            result[phase]['outlet_top_max_abs']=float(abs(a[im,jm,:,:]-b[im,jm,:,:]).max())
        (out/'rhs_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
        if scaled.max()>2e-10:
            raise AssertionError(f'KEEP open {phase} RHS differs: {result[phase]}')
    return result


def profile_state_targets(case,top,inlet_mode='prof',blowing='off',inflow_mode='complete_state'):
    """Boundary response at the first native step, Ma=.3, Tw=1, physical x/y."""
    profile=np.loadtxt(case/'datin/inlet.prof',skiprows=4)
    result={}
    for stage in range(1,4):
        snapshot=read_q_snapshot(case/f'outdat/rhs.pre_rhs.step00000000.rk{stage:02d}.rank00000000.bin')
        im,jm,km,hm,nq=snapshot.header
        shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq)
        q=snapshot.values.reshape(shape,order='F')[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
        rho=q[...,0];velocity=q[...,1:4]/rho[...,None]
        pressure=.4*(q[...,4]-.5*np.sum(q[...,1:4]*velocity,axis=-1))
        temperature=(1.4*.3**2)*pressure/rho
        inlet_velocity=np.zeros((jm+1,km+1,3))
        inlet_velocity[...,0]=profile[:,1,None]
        inlet_velocity[...,1]=profile[:,2,None]
        inlet_rho=np.broadcast_to(profile[:,0,None],(jm+1,km+1)).copy()
        inlet_temperature=np.broadcast_to(profile[:,3,None],(jm+1,km+1)).copy()
        if inlet_mode=='intp':
            # First native step uses t=0 at each RK stage, exactly at source plane 0.
            wave=np.sin(np.pi*np.linspace(0,1,jm+1))[:,None]*np.cos(2*np.pi*np.linspace(0,1,km+1))[None,:]
            wave*=temporal_amplitude(0,0.,1e-5,'nonpolynomial')
            inlet_rho+=FIELD_AMPLITUDES['ro']*wave
            inlet_temperature+=FIELD_AMPLITUDES['t']*wave
            for component in range(3):
                inlet_velocity[...,component]+=FIELD_AMPLITUDES[f'u{component+1}']*wave
        if inflow_mode=='mach_pressure':
            inlet_rho=.126*((4*pressure[1]-pressure[2])/3)/inlet_temperature
        checks=dict(inlet_velocity=scaled([velocity[0,1:jm],inlet_velocity[1:jm]]),
            inlet_rho=scaled([rho[0,1:jm],inlet_rho[1:jm]]),
            inlet_temperature=scaled([temperature[0,1:jm],inlet_temperature[1:jm]]),
            wall_temperature=scaled([temperature[:,0],np.ones_like(temperature[:,0])]))
        if top!=52:
            checks['outlet_pressure']=scaled([pressure[im,1:jm],np.full_like(pressure[im,1:jm],1/(1.4*.3**2))])
        if blowing=='off':
            checks['wall_velocity']=scaled([velocity[:,0],np.zeros_like(velocity[:,0])])
        else:
            checks['wall_tangential_velocity']=scaled([velocity[:,0][...,[0,2]],np.zeros_like(velocity[:,0][...,[0,2]])])
            checks['wall_forcing']=wall_forcing_response(case,velocity[:,0,:,1],blowing)
        if top==50:
            primitive=np.concatenate([rho[...,None],velocity,pressure[...,None]],axis=-1)
            expected=(4*primitive[:,jm-1]-primitive[:,jm-2])/3
            checks['top_zero_normal_derivative']=scaled([primitive[:,jm],expected])
        result[str(stage)]=checks
    return result


def wall_forcing_response(case,observed,mode,envelope=(2.,5.,8.)):
    with h5py.File(case/'outdat/new/resources/geometry.h5') as grid:
        x=grid['q0001'][:,0,:].T;z=grid['q0003'][:,0,:].T
    lz=float(np.ptp(z))
    base=np.zeros_like(x)
    xa,xb,xc=envelope
    for a,b,offset in ((xa,xb,0.),(xb,xc,np.pi/2)):
        if mode=='modal' and offset!=0:
            continue
        mask=(x>=a)&(x<=b)
        if offset!=0:
            mask&=x>a
        theta=2*np.pi*(x-a)/(b-a)
        fx=4*np.sin(theta)*(1-np.cos(theta))/np.sqrt(27)
        if mode=='modal':
            phases=np.linspace(.05,.75,15)
            weights=.2*np.array([1.,.8])/(1-.8**2)
            gz=sum(weights[l-1]*np.sin(2*np.pi*l*((z-z.min())/lz+phases[l-1])) for l in (1,2))
            ht=sum(weights[m-1]*np.sin(2*np.pi*m*phases[m+9]) for m in (1,2))
            base[mask]=(.001*fx*gz*ht)[mask]
        else:
            base[mask]=(.001*fx*np.sin(4*np.pi*z/lz+offset))[mask]
    if mode=='counter':
        i,k=np.indices(base.shape);k=k%(base.shape[1]-1)
        h=(104729*(i+1)+130363*(k+1)+433494437)%2147483629
        expected=base*(1+.1*(2*h/2147483629-1))
        return scaled([observed,expected])
    if mode=='modal':
        return scaled([observed,base])
    active=abs(base)>64*np.finfo(float).eps*.001
    ratio=observed[active]/base[active]
    error=max(0.,float(np.max(.9-ratio)),float(np.max(ratio-1.1)),
              float(np.max(abs(observed[~active]))))
    if not np.isfinite(error) or error>2e-10:
        raise AssertionError(f'Legacy random wall does not obey its prescribed envelope: {error}')
    return error


def wall_inputs(out,mode):
    if mode=='off':
        return None,None
    folder=out/'wall_inputs';folder.mkdir()
    path=folder/'wallbs.dat'
    path.write_text('KEEP boundary fixture\namplitude beta xa xb xc\n'
                    f'1.e-3 2.0 2.0 5.0 8.0\nnmod_t nmod_z\n{2 if mode=="modal" else 0} 2\n')
    phase=None
    if mode=='modal':
        phase=folder/'wallbs_phase.dat'
        phase.write_text(' '.join(f'{v:.17e}' for v in np.linspace(.05,.75,15))+'\n')
    return path,phase


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['compatibility','incoming_only','nonreflecting'],default='compatibility')
    parser.add_argument('--outlet',type=int,choices=[21,22],default=22)
    parser.add_argument('--top',type=int,choices=[50,51,52],default=52)
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--inlet-mode',choices=['prof','intp'],default='prof')
    parser.add_argument('--inflow-mode',choices=['complete_state','mach_pressure'],default='complete_state')
    parser.add_argument('--blowing',choices=['off','modal','counter','legacy_random'],default='off')
    parser.add_argument('--extruded',action='store_true',help='Use the existing HBL UDF grid.2d producer; initializer defaults to ninit=2')
    parser.add_argument('--initial-dimension',type=int,choices=[0,2,3],help='Default is 2 for extrusion and 3 for read-grid fixtures')
    parser.add_argument('--axes',nargs='*',choices=['x','y','z'],default=['x','y','z'])
    parser.add_argument('--multi-axis',action='store_true')
    parser.add_argument('--warp',type=float,nargs=2,default=(0.,0.),metavar=('X','Y'))
    parser.add_argument('--scheme',choices=['643e','643c','543e','743e'],default='643e')
    parser.add_argument('--diffusion',choices=['643e','643c'],default='643e')
    parser.add_argument('--viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--filter',choices=['off','explicit10','compact10'],default='off')
    parser.add_argument('--filter-workspace',choices=['scalar','full'],default='scalar')
    parser.add_argument('--transport',choices=['pinned','device-aware'],default='pinned')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--timestep',choices=['fixed','auto'],default='fixed')
    opts=parser.parse_args()
    if (opts.outlet,opts.top)!=(22,52) and not (opts.outlet==21 and opts.top in (50,51,52)):
        parser.error('This gate covers the implemented 11/profile-outlet/41/top layouts')
    if opts.outlet==21 and opts.top==52 and opts.mode!='nonreflecting':
        parser.error('The newly connected pressure-outlet/top52 layout uses the existing nonreflecting mode')
    if opts.convection=='standard' and opts.viscosity!='me6_opti':
        parser.error('Original convection in this new-form gate requires ME6 viscosity')
    if opts.convection=='keep' and opts.scheme not in ('643e','643c'):
        parser.error('KEEP retains its sixth-order explicit or compact convection')
    nscbc=opts.top==52
    if not nscbc:
        opts.mode=None
    opts.output=opts.output.resolve();opts.output.mkdir(parents=True,exist_ok=False)
    initial_dimension=opts.initial_dimension if opts.initial_dimension is not None else (2 if opts.extruded else 3)
    initial=write_common_initial(opts.output,opts.extruded,initial_dimension)
    wallbs,wall_phase=wall_inputs(opts.output,opts.blowing)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    timestep="mode='fixed',fixed_dt=1.d-5" if opts.timestep=='fixed' else "mode='auto',cfl_target=.5d0,growth_factor=1.1d0"
    runtime=f"""&run_control max_steps=4 /
&timestep {timestep} /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled=f /
"""
    numerics={'convection_form':opts.convection,'viscous_form':opts.viscosity,'conschm':opts.scheme,'difschm':opts.diffusion}
    if opts.scheme in ('543e','743e'):
        numerics.update(recon_schem=5 if opts.scheme=='743e' else 3,lchardecomp=opts.scheme=='743e')
    if opts.filter!='off':
        numerics['filter_scheme']=opts.filter
    boundary={'nscbc_farfield_mode':opts.mode} if nscbc else {}
    boundary['profile_inflow_mode']=opts.inflow_mode
    if opts.blowing!='off':
        boundary['wall_blowing_mode']='legacy_random' if opts.blowing=='legacy_random' else 'current'
    filtered=opts.filter!='off'
    matrix=[(1,'x',None)]+[(2,axis,None) for axis in opts.axes]
    if opts.multi_axis:
        matrix.append((4,'xy',(2,2,1)))
    report={'status':'running','mode':opts.mode,'scheme':opts.scheme,'diffusion':opts.diffusion,'viscosity':opts.viscosity,
            'convection':opts.convection,'boundaries':[11,opts.outlet,41,opts.top,1,1],
            'inlet_mode':opts.inlet_mode,'inflow_mode':opts.inflow_mode,'blowing':opts.blowing,
            'extruded':opts.extruded,
            'warp':list(opts.warp),
            'initial_dimension':initial_dimension,
            'filter':opts.filter,'filter_workspace':opts.filter_workspace,
            'transport':opts.transport,'timestep':opts.timestep,
            'initial':str(initial) if initial is not None else None,'cases':{},'clocks':{}}
    singles={}
    for ranks,axis,topology in matrix:
        runs={}
        for backend in ('cpu','gpu'):
            args=SimpleNamespace(output=opts.output,executable=getattr(opts,backend).resolve(),
                mpiexec=opts.mpiexec.resolve(),case='dynamic' if opts.inlet_mode=='intp' else 'curve',mode='steps',restart_step=2,
                initial_dimension=initial_dimension,initial_restart=False,legacy_statistics=False,statistics=False,
                filter_workspace=opts.filter_workspace,force='fixed',axis=axis[0],no_samples=True,wall_samples=False,unrestricted=True,
                inflow_count=12)
            common=dict(lfilter=filtered,scheme=opts.scheme,numerics=numerics,boundary_options=boundary,
                        flatplate_warp=opts.warp,flatplate_bctype=(opts.outlet,opts.top),run_config_text=runtime,initial_resource=initial,
                        solver_halo_transport=opts.transport,wallbs_resource=wallbs,wall_phase_resource=wall_phase,
                        flatplate_extruded=opts.extruded)
            if opts.mode=='nonreflecting':
                common.update(rk_state_check=True,rk_stage_state_check=True)
            case,_=run_case(args,ROOT,backend,ranks,'keep_open_'+axis,4,topology=topology,
                rhs_snapshot_step=0 if ranks==1 else None,memcheck=opts.memcheck and backend=='gpu' and ranks==1,**common)
            control=(case/FINAL/'control.bin').read_bytes()
            if opts.extruded:
                resources=case/'outdat/new/resources'
                if not (resources/'grid.2d').is_file() or (resources/'grid.h5').exists():
                    raise AssertionError('Extruded restart must freeze the original grid.2d, not the generated 3D grid')
                with h5py.File(resources/'grid.2d') as plane,h5py.File(resources/'geometry.h5') as geometry:
                    for component,name in enumerate(('x','y'),start=1):
                        if not np.array_equal(geometry[f'q{component:04d}'][:],np.broadcast_to(plane[name][:],(17,17,17))):
                            raise AssertionError('Native HBL extrusion changed the supplied 2D coordinates')
                    if float(np.ptp(geometry['q0003'][:]))!=90.:
                        raise AssertionError('This fixture requires the existing HBL UDF spanwise length 90')
            if opts.convection=='keep':
                identity=(b'/o24' if opts.scheme=='643e' else b'/o34') if nscbc and opts.outlet==22 else (
                    b'/w24' if opts.scheme=='643e' else b'/w34')
                if identity not in control:
                    raise AssertionError('KEEP open numerical identity is missing')
            if opts.viscosity=='me6_opti' and b'/me6opti-g2w4-v1' not in control:
                raise AssertionError('ME6 fixed nodal boundary-band identity is missing')
            if opts.mode in ('compatibility','incoming_only') and b'bc22-backflow-v1/yz-d2-v1' not in control:
                raise AssertionError('KEEP open corner transport identity is missing')
            if opts.mode=='compatibility' and b'bc52-compat-grazing-v2' not in control:
                raise AssertionError('KEEP open grazing classification identity is missing')
            if opts.mode=='incoming_only' and b'bc52-incoming-grazing-v2' not in control:
                raise AssertionError('KEEP open incoming-only target identity is missing')
            if opts.mode=='nonreflecting':
                identity=b'keep-top-nr-corner-conv-v1' if opts.outlet==22 else b'y52-nr-geom-source-v1'
                if identity not in control:
                    raise AssertionError('Selected nonreflecting boundary identity is missing')
            if nscbc and opts.outlet==22 and opts.scheme in ('643e','643c') and b'/wall-z0-v1' not in control:
                raise AssertionError('KEEP outlet/lower-wall ownership identity is missing')
            if not any(opts.warp):
                with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
                    for component,coordinate_axis in enumerate((2,1,0),start=1):
                        coordinates=geometry[f'q{component:04d}'][:]
                        line=coordinates[0,0,:] if coordinate_axis==2 else (coordinates[0,:,0] if coordinate_axis==1 else coordinates[:,0,0])
                        shape=[1,1,1];shape[coordinate_axis]=len(line)
                        if not (np.diff(line)>0).all() or not np.array_equal(coordinates,np.broadcast_to(line.reshape(shape),coordinates.shape)):
                            raise AssertionError('Physical profile fixture coordinates are not the prescribed increasing tensor grid')
            if backend=='gpu' and opts.transport=='device-aware':
                active=[line for line in (case/'run.log').read_text().splitlines()
                        if 'ASTR_GPU_HALO_DEVICE_AWARE rank=' in line and 'active=T' in line]
                if len(active)!=ranks:
                    raise AssertionError('KEEP open device-aware solution halo is not active on every rank')
            runs[backend]=case
            if opts.timestep=='auto':
                report['clocks'][f'{backend}_np{ranks}_{axis}']=adaptive_clocks(case)
            if ranks==1:
                singles[backend]=case/FINAL/'state.h5'
                if opts.inlet_mode=='intp':
                    with h5py.File(case/FINAL/'inflow.h5') as cache:
                        cursor=int(cache['metadata'][2])
                    if cursor<4:
                        raise AssertionError('Dynamic inlet never advanced its initial four-plane window')
                    report.setdefault('inflow_cache_cursor',{})[backend]=cursor
            elif opts.blowing!='legacy_random':
                report['cases'][f'{backend}_np{ranks}_{axis}_vs_np1']=compare_q(singles[backend],case/FINAL/'state.h5')
            else:
                report['cases'][f'{backend}_np{ranks}_{axis}_vs_np1']='not a same-input comparison: prescribed rank-local random forcing differs'
            if opts.axes and ((ranks==2 and axis==opts.axes[0]) or (opts.blowing=='legacy_random' and ranks>1)):
                seed=case/'outdat/new/checkpoints/step000000000002'
                resumed,_=run_case(args,ROOT,backend,ranks,'keep_open_'+axis+'_resumed',4,restore=seed,topology=topology,**common)
                compare_fields(case/FINAL/'state.h5',resumed/FINAL/'state.h5')
                if opts.inlet_mode=='intp':
                    compare_fields(case/FINAL/'inflow.h5',resumed/FINAL/'inflow.h5')
                if opts.blowing!='off' and (resumed/'datin/wallbs.dat').exists():
                    raise AssertionError('Wall continuation retained an external forcing input instead of using frozen resources')
                if opts.extruded and any((resumed/'datin'/name).exists() for name in ('grid.2d','flowini2d.h5','inlet.prof')):
                    raise AssertionError('Extruded continuation retained external geometry/profile/initial-state inputs')
                if opts.timestep=='auto':
                    expected=report['clocks'][f'{backend}_np{ranks}_{axis}']['records'][2:]
                    if adaptive_clocks(resumed,first=3)['records']!=expected:
                        raise AssertionError('KEEP open exact continuation changed automatic clocks')
                for name in ('control.bin','archives.bin','insitu_control.bin'):
                    if (case/FINAL/name).read_bytes()!=(resumed/FINAL/name).read_bytes():
                        raise AssertionError(f'KEEP open exact restart differs: {backend} {name}')
                report['cases'][f'{backend}_np{ranks}_{axis}_exact']='state datasets, inflow cache when enabled, and control files identical'
                if opts.mode=='incoming_only':
                    changed=dict(common)
                    changed['boundary_options']={**boundary,'nscbc_farfield_u':1.01}
                    run_case(args,ROOT,backend,ranks,'keep_open_'+axis+'_changed_target',4,
                        restore=seed,topology=topology,reject='boundary/driver configuration mismatch',**changed)
                    report['cases'][f'{backend}_np2_{axis}_changed_target']='exact restore rejected'
        if ranks==1:
            report['rhs']=compare_open_rhs(runs['cpu'],runs['gpu'],opts.output)
            report['all_stage_rhs_and_cache']=frozen_rhs(runs['cpu'],runs['gpu'],[0,1],mach=.3)
            report['boundary_targets']={backend:profile_state_targets(case,opts.top,opts.inlet_mode,opts.blowing,opts.inflow_mode) for backend,case in runs.items()}
            if opts.mode=='nonreflecting' and opts.outlet==22:
                report['corner']={backend:check_corner(case,backend,opts.output) for backend,case in runs.items()}
            if opts.mode=='nonreflecting' and opts.outlet==21:
                from run_keep_y_farfield_gate import source_projection
                report['source_projection']={backend:source_projection(case,False) for backend,case in runs.items()}
        report['cases'][f'cpu_gpu_np{ranks}_{axis}']=compare_q(runs['cpu']/FINAL/'state.h5',runs['gpu']/FINAL/'state.h5')
        if opts.timestep=='auto':
            a=np.array(report['clocks'][f'cpu_np{ranks}_{axis}']['records'])[:,1:]
            b=np.array(report['clocks'][f'gpu_np{ranks}_{axis}']['records'])[:,1:]
            difference=float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
            if difference>2e-10:
                raise AssertionError(f'KEEP open CPU/GPU automatic clocks differ: {difference}')
            report['clocks'][f'cpu_gpu_np{ranks}_{axis}_max_scaled']=difference
        (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        label=opts.mode if nscbc else f'21/{opts.top}'
        print(f'{opts.convection} profile {label} NP={ranks} {axis}: compared',flush=True)
    report['status']='passed'
    (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
