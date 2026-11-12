"""Exercise existing simple/LDC boundary kernels with KEEP/ME6, not new BC formulas."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from named_fixture import namelist
from compare_q_validation_snapshots import read_q_snapshot
from run_keep_periodic_gate import compare_q
from run_output_restart_validation import compare_fields

FINAL = Path('outdat/new/checkpoints/step000000000004')
# Each existing family/direction is exercised; format/filter combinations are sampled.
MATRIX = [
    ('inflow', '643e', '643e', 'off', 'scalar'),
    ('nscbc', '643e', '643e', 'off', 'scalar'),
    ('ldc', '643e', '643e', 'off', 'scalar'),
    ('42y', '643e', '643e', 'off', 'scalar'),
    ('41x', '643e', '643e', 'off', 'scalar'),
    ('41z', '643c', '643e', 'compact10', 'scalar'),
    ('42x', '643c', '643c', 'explicit10', 'scalar'),
    ('411y', '643e', '643e', 'explicit10', 'full'),
    ('421y', '643c', '643e', 'off', 'scalar'),
    ('31y', '643e', '643e', 'off', 'scalar'),
    ('50x', '643e', '643c', 'off', 'scalar'),
    ('50y', '643c', '643e', 'compact10', 'scalar'),
    ('50z', '643e', '643e', 'explicit10', 'scalar'),
    ('60x', '643c', '643e', 'explicit10', 'full'),
    ('60y', '643e', '643e', 'off', 'scalar'),
    ('60z', '643c', '643c', 'compact10', 'scalar'),
]


def prepare(case, label, backend, scheme, diffusion, filtering, restore=None,
            convection='keep', viscosity='standard', stretched=False, wall_temperature=1.,exact_restart=True,
            steps=4,checkpoint=True,inflow_mode='complete_state',mach=.3,xslip=10.,intervals=16,upper_nonreflecting=False,
            wall_blowing_mode=None):
    case.mkdir(parents=True, exist_ok=False)
    datin = case / 'datin'
    datin.mkdir()
    (case / 'outdat/new').mkdir(parents=True)
    ldc = label=='ldc'
    open_x = label in ('inflow','nscbc')
    if ldc or open_x:
        kind,axis = (41 if ldc else (11 if label=='inflow' else 12)),0
    else:
        kind,axis = int(label[:-1]),'xyz'.index(label[-1])
    lengths = [1.,1.,1.] if ldc else [2*np.pi,np.pi,2*np.pi]
    lines = [np.linspace(0,length,intervals+1) for length in lengths]
    if stretched:
        for d in ([0,1] if ldc else [axis]):
            lines[d] = lengths[d]*np.expm1(1.5*np.linspace(0,1,17))/np.expm1(1.5)
    xyz = np.meshgrid(*lines,indexing='ij')
    s = xyz[axis] / [2*np.pi, np.pi, 2*np.pi][axis]
    tangents = [d for d in range(3) if d != axis]
    perturbation = np.cos(xyz[tangents[0]] * (2 if tangents[0] == 1 else 1))
    perturbation *= np.cos(xyz[tangents[1]] * (2 if tangents[1] == 1 else 1))
    profile = np.sin(np.pi*s)**2
    velocity = [.002*(d+1)*profile*(1+.01*perturbation) for d in range(3)]
    density = 1 + .001*np.cos(2*np.pi*s)*perturbation
    temperature = 1 + .001*profile*perturbation
    if kind in (50, 60, 421):
        for d in tangents:
            velocity[d] = .002*(d+1)*np.cos(np.pi*s)*(1+.01*perturbation)
        temperature = 1+.001*np.cos(2*np.pi*s)*perturbation
    if kind == 411:
        for d in tangents:
            velocity[d] = .002*(d+1)*np.cos(np.pi*s/2)**2*(1+.01*perturbation)
        temperature = 1+.001*np.cos(np.pi*s/2)**2*perturbation
    if kind == 31:
        density = 2-s
        temperature = (1.4*mach**2)*(1+1.5*s)/density
    if ldc:
        x,y,z = xyz
        envelope = np.sin(np.pi*x)**2*np.sin(np.pi*y)**2
        density = 1+.001*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z)
        temperature = 1+.001*envelope*np.cos(2*np.pi*z)
        velocity = [.01*(d+1)*envelope*np.cos(2*np.pi*z) for d in range(3)]
    if open_x:
        x,y,z = xyz
        temperature = 1+.0002*np.sin(x)*perturbation
        velocity = [1+.001*np.cos(x)*perturbation,.001*np.sin(2*y)*np.cos(z),.001*np.cos(2*y)*np.sin(z)]
    fields = dict(ro=density, t=temperature, **{f'u{d+1}':v for d,v in enumerate(velocity)})
    if restore is None:
        with h5py.File(datin/'flowini3d.h5', 'x') as state:
            for name, value in fields.items():
                state[name] = value.transpose(2,1,0)
        with h5py.File(datin/'grid.simple.h5', 'x') as grid:
            for name, value in zip('xyz', xyz):
                grid[name] = value.transpose(2,1,0)
    boundaries = [1]*6
    boundaries[2*axis:2*axis+2] = [kind]*2
    walls = [0.]*6
    walls[2*axis:2*axis+2] = [wall_temperature]*2
    periodic = [axis != d for d in range(3)]
    if ldc:
        boundaries = [41,41,41,0,1,1]
        walls = [wall_temperature]*4+[0.,0.]
        periodic = [False,False,True]
    if open_x:
        boundaries = [kind,21 if label=='inflow' else 22,1,1,1,1]
    groups = {
        'case_options': dict(flowtype='ldcavity' if ldc else ('tgv' if open_x else 'channel'),
                             ltimrpt=True, use_gpu=backend=='gpu', iomode='h'),
        'mesh': dict(ia=intervals, ja=intervals, ka=intervals, lihomo=periodic[0], ljhomo=periodic[1], lkhomo=periodic[2],
                     lreadgrid=True, gridfile='datin/grid.simple.h5', lfftk=False),
        'physics': dict(nondimen=True, diffterm=True, lcomb=False, ref_tem=273.15, reynolds=3000.,
                        mach=mach, num_species=0, turbmode='none'),
        'numerics': dict(convection_form=convection, viscous_form=viscosity, conschm=scheme,
                         difschm=diffusion, rkscheme='rk3', lfilter=filtering!='off',
                         filter_scheme=filtering if filtering!='off' else 'explicit10', alfa_filter=.49,
                         kcutoff=48, recon_schem=3, lchardecomp=False, bfacmpld=.3, shkcrt=.01),
        'boundary': dict(bctype=boundaries, twall=walls, xslip=xslip),
        'initialization': dict(limmbou=False, lrestart=False, ninit=3),
        'sponge': dict(spg_i0=0, spg_im=0, spg_j0=0, spg_jm=0, spg_k0=0, spg_km=0, spg_def='layer'),
        'precision_options': dict(gpu_precision_mode='fp64'),
        'transport_options': dict(channel_force_mode='fixed', channel_force_fixed=0.),
    }
    if open_x:
        groups['boundary']['pout']=1.
        if label=='inflow':
            groups['boundary']['turbinf']='prof'
            groups['boundary_options']=dict(profile_inflow_mode=inflow_mode)
    if upper_nonreflecting:
        groups['case_options']['flowtype']='tgv'
        groups['boundary']['bctype']=[1,1,41,52,1,1]
        groups['boundary_options']=dict(nscbc_farfield_mode='nonreflecting')
    if wall_blowing_mode is not None:
        groups.setdefault('boundary_options',{})['wall_blowing_mode']=wall_blowing_mode
    (datin/'input.case').write_text(namelist(groups))
    (datin/'input.run').write_text(namelist({
        'run_control': dict(max_steps=steps), 'timestep': dict(mode='fixed', fixed_dt=1e-5),
        'logging': dict(list_steps=1), 'repair': dict(crashfix=False), 'statistics': dict(enabled=False)}))
    (datin/'input.output').write_text(namelist({
        'output': dict(directory='outdat/new', restore_directory=str(restore or ''), exact_restart=exact_restart,
                       checkpoint_statistics=False, host_budget_bytes=-1, device_budget_bytes=-1,
                       buffer_bytes=67108864, device_reserve_bytes=0),
        'checkpoint': dict(enabled=checkpoint, mode='steps', interval_steps=2, keep=2, initial_frame=False),
        'volume': dict(enabled=False), 'slices': dict(enabled=False)}))
    return axis


def mixed_wall_targets(case,xslip,mach=.3):
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        x=geometry['q0001'][:].transpose(2,1,0)[:,0,:]
    slip=x<=xslip
    result=dict(slip_nodes=int(slip.sum()),noslip_nodes=int((~slip).sum()),stages={})
    for stage in range(1,4):
        snapshot=read_q_snapshot(case/f'outdat/rhs.pre_rhs.step00000000.rk{stage:02d}.rank00000000.bin')
        im,jm,km,hm,nq=snapshot.header
        q=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
        q=q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
        rho=q[...,0];velocity=q[...,1:4]/rho[...,None]
        pressure=.4*(q[...,4]-.5*np.sum(q[...,1:4]*velocity,axis=-1))
        temperature=1.4*mach**2*pressure/rho
        expected_velocity=(4*velocity[:,1]-velocity[:,2])/3
        expected_velocity[...,1]=0
        expected_velocity[~slip]=0
        expected_temperature=np.where(slip,(4*temperature[:,1]-temperature[:,2])/3,1.)
        errors=[scaled([velocity[:,0],expected_velocity]),scaled([temperature[:,0],expected_temperature]),
                scaled([pressure[:,0],(4*pressure[:,1]-pressure[:,2])/3]),
                scaled([velocity[:,jm],np.zeros_like(velocity[:,jm])]),
                scaled([temperature[:,jm],np.ones_like(temperature[:,jm])])]
        result['stages'][str(stage)]=max(errors)
    return result


def frozen_rhs(cpu, gpu, axis,mach=.3):
    result = {}
    for stage in range(1,4):
        suffix = f'.step00000000.rk{stage:02d}.rank00000000.bin'
        snapshots = [read_q_snapshot(p/'outdat'/('rhs.pre_rhs'+suffix)) for p in (cpu,gpu)]
        if snapshots[0].header != snapshots[1].header:
            raise AssertionError('frozen state layouts differ')
        im,jm,km,hm,nq = snapshots[0].header
        shape = (im+2*hm+1,jm+2*hm+1,km+2*hm+1)
        indices = np.indices(shape)-hm
        limits = np.array([im,jm,km])[:,None,None,None]
        outside = (indices<0)|(indices>limits)
        mask = (outside.sum(axis=0)<=1)&((indices>=-3)&(indices<=limits+3)).all(axis=0)
        for d in np.atleast_1d(axis):
            mask &= (indices[d]>=0)&(indices[d]<=limits[d])
        states = [s.values.reshape((*shape,nq),order='F')[mask] for s in snapshots]
        entries = {'q':scaled(states)}
        for path, state in zip((cpu,gpu),states):
            with (path/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
                header = np.fromfile(stream,np.int32,4)
                values = np.fromfile(stream,np.float64).reshape((*shape,6),order='F')[mask]
            assert tuple(header)==(im,jm,km,hm)
            rho = state[:,0]
            if not (rho>0).all():
                raise AssertionError(f'nonpositive frozen density at RK stage {stage}: {path}')
            vel = state[:,1:4]/rho[:,None]
            pressure = .4*(state[:,4]-.5*np.sum(state[:,1:4]*vel,axis=1))
            if not np.isfinite(pressure).all() or not (pressure>0).all():
                raise AssertionError(f'nonpositive/nonfinite frozen pressure at RK stage {stage}: {path}')
            expected = np.column_stack((rho,vel,pressure,(1.4*mach**2)*pressure/rho))
            entries[path.name+'_cache'] = scaled([values,expected])
        for phase in ('conv','full'):
            arrays = []
            for path in (cpu,gpu):
                with (path/'outdat'/('rhs.'+phase+suffix)).open('rb') as stream:
                    header = np.fromfile(stream,np.int32,4)
                    arrays.append(np.fromfile(stream,np.float64))
                assert tuple(header)==(im,jm,km,nq)
            entries[phase] = scaled(arrays)
        result[str(stage)] = entries
    return result


def ldc_wall_targets(case,mach=.3):
    result = {}
    for stage in range(1,4):
        path = case/f'outdat/rhs.pre_rhs.step00000000.rk{stage:02d}.rank00000000.bin'
        state = read_q_snapshot(path)
        im,jm,km,hm,nq = state.header
        q = state.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
        q = q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
        velocity = q[...,1:4]/q[...,0,None]
        pressure = .4*(q[...,4]-.5*np.sum(q[...,1:4]*velocity,axis=-1))
        temperature = (1.4*mach**2)*pressure/q[...,0]
        i,j,k = np.indices(q.shape[:3])
        wall = (i==0)|(i==im)|(j==0)|(j==jm)
        expected = np.zeros_like(velocity)
        expected[:,jm,:,0] = 1.
        result[str(stage)] = dict(velocity=scaled([velocity[wall],expected[wall]]),
            temperature=scaled([temperature[wall],np.ones_like(temperature[wall])]))
    return result


def inflow_targets(case,mode,mach):
    result = {}
    for stage in range(1,4):
        state = read_q_snapshot(case/f'outdat/rhs.pre_rhs.step00000000.rk{stage:02d}.rank00000000.bin')
        im,jm,km,hm,nq = state.header
        q = state.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
        q = q[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
        vel = q[...,1:4]/q[...,0,None]
        prs = .4*(q[...,4]-.5*np.sum(q[...,1:4]*vel,axis=-1))
        target_p = np.full((jm+1,km+1),1/(1.4*mach**2))
        if mode=='mach_pressure' and mach<1:
            target_p = (4*prs[1]-prs[2])/3
        target_rho = target_p*(1.4*mach**2)
        target_vel = np.zeros_like(vel[0]);target_vel[...,0]=1
        result[str(stage)] = dict(rho=scaled([q[0,...,0],target_rho]),
            velocity=scaled([vel[0],target_vel]),pressure=scaled([prs[0],target_p]),
            temperature=scaled([(1.4*mach**2)*prs[0]/q[0,...,0],np.ones_like(target_p)]))
    return result


def scaled(arrays):
    a,b = arrays
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError('nonfinite comparison input')
    error = float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
    if error>2e-10:
        raise AssertionError(f'maximum scaled difference {error} exceeds 2e-10')
    return error


def geometry_match(first, second):
    result = {}
    with h5py.File(first/'outdat/new/resources/geometry.h5') as a, \
         h5py.File(second/'outdat/new/resources/geometry.h5') as b:
        for component in range(1,14):
            key = f'q{component:04d}'
            x,y = a[key][:],b[key][:]
            result[key] = dict(max_scaled=scaled([x,y]),max_absolute=float(np.max(abs(x-y))))
    return result


def nodal_band_match(case, reference, axis):
    """The first-stage state is shared; compare the nodal viscous contribution only."""
    suffix = '.step00000000.rk01.rank00000000.bin'
    initial = []
    for path in (case,reference):
        state = read_q_snapshot(path/'outdat'/('rhs.pre_rhs'+suffix))
        im,jm,km,hm,nq = state.header
        values = state.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
        initial.append(values[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1])
    if not np.array_equal(*initial):
        raise AssertionError('nodal band reference is not the identical first-stage state')
    sources = []
    for path in (case,reference):
        phases = {}
        for phase in ('conv','full'):
            with (path/'outdat'/('rhs.'+phase+suffix)).open('rb') as stream:
                im,jm,km,nq = np.fromfile(stream,np.int32,4)
                phases[phase] = np.fromfile(stream,np.float64).reshape((im+1,jm+1,km+1,nq),order='F')
        sources.append(phases['full']-phases['conv'])
    positions = np.indices((im+1,jm+1,km+1))
    mask = np.zeros((im+1,jm+1,km+1),bool)
    for d in np.atleast_1d(axis):
        mask |= np.minimum(positions[d],[im,jm,km][d]-positions[d])<4
    return scaled([values[mask] for values in sources])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','gpu','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cases',nargs='+',default=[row[0] for row in MATRIX],choices=[row[0] for row in MATRIX])
    parser.add_argument('--mpi',action='store_true')
    parser.add_argument('--mpi-axis',choices=['x','y','z'])
    parser.add_argument('--multi-axis',action='store_true')
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--scheme',choices=['643e','643c'])
    parser.add_argument('--diffusion',choices=['643e','643c'])
    parser.add_argument('--filter',choices=['off','explicit10','compact10'])
    parser.add_argument('--workspace',choices=['scalar','full'])
    parser.add_argument('--inflow-mode',choices=['complete_state','mach_pressure'],default='complete_state')
    parser.add_argument('--mach',type=float,default=.3)
    parser.add_argument('--xslip',type=float,default=10.)
    parser.add_argument('--stretched',action='store_true')
    parser.add_argument('--nodal-reference',type=Path)
    parser.add_argument('--nonexact-me6',action='store_true',help='Switch a standard-nodal checkpoint to ME6 at step 2')
    args = parser.parse_args()
    if args.nonexact_me6 and (args.viscosity!='standard' or args.convection!='keep' or not args.mpi):
        parser.error('--nonexact-me6 requires KEEP, standard viscosity and --mpi')
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
               OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1',
               ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
               ASTR_VALIDATION_RK_STATE='1')
    report = dict(status='running',cases={})
    for label,scheme,diffusion,filtering,workspace in MATRIX:
        if label not in args.cases:
            continue
        if args.viscosity=='me6_opti':
            diffusion='643e'
        if args.convection=='standard':
            scheme='643e'
        if args.scheme:
            scheme=args.scheme
        if args.diffusion:
            diffusion=args.diffusion
        if args.filter:
            filtering=args.filter
        if args.workspace:
            workspace=args.workspace
        singles = {}
        axis = 0 if label in ('ldc','inflow','nscbc') else 'xyz'.index(label[-1])
        physical_axes = (0,1) if label=='ldc' else (axis,)
        split_axis = 'xyz'.index(args.mpi_axis) if args.mpi_axis else axis
        topologies = [(1,1,1)]
        if args.mpi:
            topology = [1,1,1];topology[split_axis]=2;topologies.append(tuple(topology))
        if args.multi_axis:
            topology = [1,1,1];topology[split_axis]=2;topology[(split_axis+1)%3]=2;topologies.append(tuple(topology))
        for topology in topologies:
            ranks = int(np.prod(topology));runs = {}
            for backend in ('cpu','gpu'):
                case = out/f'{label}_{backend}_np{ranks}'
                prepare(case,label,backend,scheme,diffusion,filtering,
                        convection=args.convection,viscosity=args.viscosity,stretched=args.stretched,
                        inflow_mode=args.inflow_mode,mach=args.mach,xslip=args.xslip)
                local = dict(env,ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),
                             ASTR_GPU_FILTER_WORKSPACE=workspace)
                if ranks==1:
                    local.update(ASTR_VALIDATION_RHS_PREFIX='outdat/rhs',ASTR_VALIDATION_RHS_STEP='0')
                executable = [str(getattr(args,backend).resolve())]
                if args.memcheck and backend=='gpu':
                    executable = [shutil.which('compute-sanitizer'),'--tool','memcheck',
                                  '--error-exitcode','70',*executable]
                command = [str(args.mpiexec.absolute()),'--mca','coll_hcoll_enable','0','-np',str(ranks),
                           *executable,'run','datin/input.case']
                with (case/'run.log').open('w') as log:
                    subprocess.run(command,cwd=case,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
                if args.memcheck and backend=='gpu':
                    assert (case/'run.log').read_text().count('ERROR SUMMARY: 0 errors')==ranks
                runs[backend] = case
                compare_q(case/FINAL/'state.h5',case/FINAL/'state.h5')
                if ranks==1:
                    singles[backend] = case
                else:
                    report['cases'][f'{label}_{backend}_np{ranks}_geometry_decomposition'] = geometry_match(singles[backend],case)
                    report['cases'][f'{label}_{backend}_np{ranks}_decomposition'] = compare_q(
                        singles[backend]/FINAL/'state.h5',case/FINAL/'state.h5')
                    resumed = out/(case.name+'_exact')
                    seed = case/'outdat/new/checkpoints/step000000000002'
                    prepare(resumed,label,backend,scheme,diffusion,filtering,restore=seed,
                            convection=args.convection,viscosity=args.viscosity,stretched=args.stretched,
                            inflow_mode=args.inflow_mode,mach=args.mach,xslip=args.xslip)
                    with (resumed/'run.log').open('w') as log:
                        subprocess.run(command,cwd=resumed,env=local,stdout=log,stderr=subprocess.STDOUT,check=True)
                    compare_fields(case/FINAL/'state.h5',resumed/FINAL/'state.h5')
                    for filename in ('control.bin','archives.bin','insitu_control.bin'):
                        assert (case/FINAL/filename).read_bytes()==(resumed/FINAL/filename).read_bytes(), filename
                    report['cases'][f'{label}_{backend}_np{ranks}_exact'] = 'datasets and control files identical; no external grid/initial field'
                    if args.nonexact_me6:
                        migrated = out/(case.name+'_me6_nonexact')
                        prepare(migrated,label,backend,scheme,diffusion,filtering,restore=seed,
                                convection=args.convection,viscosity='me6_opti',stretched=args.stretched,exact_restart=False,
                                inflow_mode=args.inflow_mode,mach=args.mach,xslip=args.xslip)
                        probe_env = dict(local,ASTR_CHECKPOINT_TEST_RESTORE_PROBE='1')
                        with (migrated/'run.log').open('w') as log:
                            subprocess.run(command,cwd=migrated,env=probe_env,stdout=log,stderr=subprocess.STDOUT,check=True)
                        report['cases'][f'{label}_{backend}_np{ranks}_nonexact_restore'] = compare_q(
                            seed/'state.h5',migrated/'outdat/restore_probe.h5')
                        compare_q(migrated/FINAL/'state.h5',migrated/FINAL/'state.h5')
                        assert 'ASTR_KEEP_RESTART statistics_new_segment_step=2 time=' in (migrated/'run.log').read_text()
                        report['cases'][f'{label}_{backend}_np{ranks}_nonexact_history'] = 'method change starts a new segment; statistics disabled in this fixture'
            entry = dict(convection_form=args.convection,viscous_form=args.viscosity,
                         convection=scheme,diffusion=diffusion,filter=filtering,workspace=workspace,
                         mapping='normal-exponential-stretch' if args.stretched else 'cartesian',
                         mach=args.mach,xslip=args.xslip if label=='411y' else None,
                         inflow_mode=args.inflow_mode if label=='inflow' else None,
                         cpu_gpu=compare_q(runs['cpu']/FINAL/'state.h5',runs['gpu']/FINAL/'state.h5'))
            if ranks==1:
                entry['same_stage_rhs'] = frozen_rhs(runs['cpu'],runs['gpu'],physical_axes,args.mach)
                if label=='ldc':
                    entry['wall_targets'] = {backend:ldc_wall_targets(runs[backend],args.mach) for backend in ('cpu','gpu')}
                if label=='inflow':
                    entry['inlet_targets'] = {backend:inflow_targets(runs[backend],args.inflow_mode,args.mach) for backend in ('cpu','gpu')}
                if label=='411y':
                    entry['wall_targets'] = {backend:mixed_wall_targets(runs[backend],args.xslip,args.mach) for backend in ('cpu','gpu')}
                if args.nodal_reference:
                    entry['same_state_nodal_band'] = {backend:nodal_band_match(
                        runs[backend],args.nodal_reference/f'{label}_{backend}_np1',physical_axes) for backend in ('cpu','gpu')}
            report['cases'][f'{label}_np{ranks}'] = entry
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(label,topology,'fields/RHS compared',flush=True)
    report['status'] = 'passed_four_step_compatibility'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
