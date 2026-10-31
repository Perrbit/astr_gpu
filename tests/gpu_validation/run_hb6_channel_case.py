#!/usr/bin/env python3
"""Run actual ASTR for the approved HB6 channel window, saving a numerical report."""
import argparse
from functools import partial
import json
from math import ceil, prod
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from hb6_channel_reference import (FORCE, MACH, REYNOLDS, GAMMA, PRANDTL,
                                   initial_fields, steady_reference)
from prepare_s1_flatplate_case import write_controller
from prepare_tgv_case import next_data_line
from run_m12_mp7_nscbc_performance import runtime, mpi_command
from run_native_precision_tgv_gate import physical_state, rhs_state
from test_high_order_wall_runtime import prepare, initial_fields as legacy_fields, MPIEXEC


def fields(grid, perturbation, wall):
    return (initial_fields(grid, perturbation=perturbation) if wall == 41 else
            legacy_fields(grid, wall_type=wall))


def estimated_dt(grid, perturbation, wall):
    coordinates, state = fields(grid, perturbation, wall)
    spacing = [float(np.ptp(c)/n) for c, n in zip(coordinates, grid)]
    sound = np.sqrt(state['t']) / MACH
    rates = [(abs(state[f'u{axis+1}'])+sound)/spacing[axis] for axis in range(3)]
    estimate = .2/sum(float(rate.max()) for rate in rates)
    # All agreed t*=1,5,10 milestones lie on complete steps with this fixed dt.
    return 1/ceil(1/estimate)


def first_step_budget(directory, grid, dt, wall):
    coordinates,_=fields(grid,0.,wall)
    dx,dy,dz=[float(np.ptp(c)/n) for c,n in zip(coordinates,grid)]
    def integral(array):
        return np.trapezoid(array[:-1,:,:-1],dx=dy,axis=1).sum(axis=(0,1))*dx*dz
    with h5py.File(directory/'outdat/new/resources/geometry.h5') as f:
        jacobian=f['q0004'][:].transpose(2,1,0)
    stages=[]
    for stage,a,b in ((1,0.,1.),(2,.75,.25),(3,1/3,2/3)):
        suffix=f'step00000000.rk{stage:02d}.rank00000000.bin'
        state={name:physical_state(directory/'diagnostics'/f'state.{name}.{suffix}','fp64')
               for name in ('pre_filter','post_filter','pre_rhs','post_update')}
        before=state['pre_rhs']
        if stage==1:
            saved=before.copy()
        rhs=rhs_state(directory/'diagnostics'/f'state.full.{suffix}','fp64')/jacobian[...,None]
        source=np.zeros_like(rhs)
        if wall==41:
            source[...,1]=FORCE
            source[...,4]=FORCE*before[...,1]/before[...,0]
        expected=a*saved+b*before+b*dt*rhs
        terms=dict(filter=integral(state['post_filter']-state['pre_filter']),
                   boundary=integral(before-state['post_filter']),
                   rk_mixing=a*integral(saved-before),
                   source=b*dt*integral(source),rhs_minus_source=b*dt*integral(rhs-source),
                   wall_update=integral(state['post_update']-expected))
        change=integral(state['post_update']-state['pre_filter'])
        residual=change-sum(terms.values())
        scaled=float(np.max(abs(residual)/(1+abs(integral(state['pre_filter'])))))
        interior=float(np.max(abs((state['post_update']-expected)[:,1:-1])/(
            1+abs(expected[:,1:-1]))))
        assert scaled<=2e-10 and interior<=2e-10,(stage,scaled,interior)
        stages.append(dict(stage=stage,terms={k:v.tolist() for k,v in terms.items()},
                           total_change=change.tolist(),reconstruction_scaled=scaled,
                           interior_rk_scaled=interior))
    return dict(stages=stages,note='Physical trapezoidal diagnostic; rhs_minus_source includes wall RHS constraints; wall_update is the post-RK state reaction, not the full boundary contribution')


def report_state(checkpoint, grid, wall, perturbation):
    with h5py.File(checkpoint/'state.h5') as f:
        q = np.stack([f[f'q{m:04d}'][:].transpose(2,1,0) for m in range(1,6)], axis=-1)
        temperature = f['q0011'][:].transpose(2,1,0)
        caches = [f[f'q{m:04d}'][:] for m in range(6,12)]
    rho = q[...,0]
    velocity = q[...,1:4]/rho[...,None]
    internal = q[...,4]-.5*rho*np.sum(velocity**2,axis=-1)
    assert np.isfinite(q).all() and np.isfinite(caches).all()
    assert rho.min()>0 and temperature.min()>0 and internal.min()>0
    coordinates, initial = fields(grid, perturbation, wall)
    dx, dy, dz = [float(np.ptp(c)/n) for c,n in zip(coordinates,grid)]

    def integral(array):
        return np.trapezoid(array[:-1,:,:-1],dx=dy,axis=1).sum(axis=(0,1))*dx*dz

    mass = float(integral(rho))
    record = dict(minimum_density=float(rho.min()), minimum_temperature=float(temperature.min()),
                  minimum_internal_energy=float(internal.min()),
                  mass=mass, initial_mass=float(integral(initial['ro'])),
                  momentum=integral(q[...,1:4]).tolist(), energy=float(integral(q[...,4])),
                  quadrature='unique periodic x/z; physical trapezoidal y; not an SBP norm')
    record['relative_mass_change'] = mass/record['initial_mass']-1
    if wall == 41:
        reference = steady_reference()
        _, uref, tref = reference.primitive(coordinates[1])
        reference_velocity = np.zeros_like(velocity)
        reference_velocity[...,0] = uref
        record['velocity_reference_scaled'] = float(np.max(abs(velocity-reference_velocity)/(1+abs(reference_velocity))))
        record['temperature_reference_scaled'] = float(np.max(abs(temperature-tref)/(1+abs(tref))))
        record['perturbation_energy'] = float(integral(.5*rho*np.sum((velocity-reference_velocity)**2,axis=-1)))
        # Independent seven-node wall derivative for the continuous-reference diagnostic.
        nodes = np.arange(7, dtype=float)
        matrix = np.stack([nodes**m for m in range(7)])
        target = np.zeros(7); target[1] = 1
        weights = np.linalg.solve(matrix,target)/dy
        mean_u = velocity[:-1,:,:-1,0].mean(axis=(0,2))
        mean_t = temperature[:-1,:,:-1].mean(axis=(0,2))
        shear=[]; heat=[]
        for side, indices in ((1,np.arange(7)),(-1,grid[1]-np.arange(7))):
            shear.append(float(side*weights@mean_u[indices]/REYNOLDS))
            conductivity=1/(REYNOLDS*PRANDTL*(GAMMA-1)*MACH**2)
            heat.append(float(weights@mean_t[indices]*conductivity))
        expected = reference.solution.sol(np.array([-1.,1.]))
        record['wall_shear'] = shear
        record['outgoing_wall_heat'] = heat
        record['wall_shear_relative_errors'] = (abs(np.array(shear)-expected[1])/abs(expected[1])).tolist()
        record['wall_heat_relative_errors'] = (abs(np.array(heat)-np.array([expected[3,0],-expected[3,1]]))/abs(expected[3])).tolist()
        record['mean_profiles'] = dict(y=coordinates[1][0,:,0].tolist(),u=mean_u.tolist(),temperature=mean_t.tolist())
    return record


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    executable=output/'astr'
    shutil.copy2(args.executable.resolve(),executable)
    dt=args.dt if args.dt is not None else estimated_dt(args.grid,args.perturbation,args.wall)
    producer=partial(initial_fields,perturbation=args.perturbation) if args.wall==41 else None
    prepare(output,args.backend,args.steps,args.restore,grid=args.grid,dt=dt,
            checkpoint_interval=args.checkpoint_interval or args.steps,
            wall_type=args.wall,initial_condition=producer,filtering=True)
    output_config=output/'datin/input.output'
    output_config.write_text(output_config.read_text().replace(
        'host_budget_bytes=134217728,device_budget_bytes=134217728',
        'host_budget_bytes=-1,device_budget_bytes=-1'))
    primary=output/'datin/input.chl'
    lines=primary.read_text().splitlines()
    marker=next(i for i,line in enumerate(lines) if 'ref_t,reynolds,mach' in line)
    lines[next_data_line(lines,marker)]=f'273.15,{REYNOLDS},{MACH}'
    primary.write_text('\n'.join(lines)+'\n')
    stride=1 if args.snapshots or args.steps<=50 else 50
    write_controller(output/'datin/controller',args.steps-1,stride,dt,feqlist=stride)
    env=runtime(','.join(map(str,args.topology)))
    for key in tuple(env):
        if key.startswith(('ASTR_INSITU_','ASTR_VALIDATION_','ASTR_WALL_BLOWING_')):
            env.pop(key)
    env.update(ASTR_GPU_FILTER_WORKSPACE=args.workspace,
               ASTR_CHANNEL_FORCE_MODE='fixed',ASTR_CHANNEL_FORCE_FIXED=str(FORCE if args.wall==41 else 0),
               ASTR_GPU_MP_POSITIVITY='off',ASTR_PERFECT_GAS_PRANDTL=str(PRANDTL),
               ASTR_CFL_DIAGNOSTICS='on',ASTR_GPU_STATISTICS_INTERVAL='1')
    if args.wall_mode!='default':
        env['ASTR_HIGH_ORDER_WALL_MODE']=args.wall_mode
    if args.snapshots:
        env.update(ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',ASTR_VALIDATION_RHS_STEP='0')
    config=dict(backend=args.backend,grid=args.grid,topology=args.topology,wall=args.wall,
                perturbation=args.perturbation,workspace=args.workspace,steps=args.steps,dt=dt,
                restore=str(args.restore) if args.restore else None,snapshots=args.snapshots,
                wall_mode=args.wall_mode)
    (output/'case.json').write_text(json.dumps(config,indent=2)+'\n')
    command=mpi_command(MPIEXEC,prod(args.topology),'pinned')+[str(executable),'run','datin/input.chl']
    with (output/'run.log').open('w') as log:
        completed=subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT)
    text=(output/'run.log').read_text()
    if completed.returncode or 'The job is done!' not in text:
        raise RuntimeError(f'ASTR failed ({completed.returncode}); see {output}/run.log')
    checkpoint=output/f'outdat/new/checkpoints/step{args.steps:012d}'
    assert (checkpoint/'COMPLETE').is_file(),checkpoint
    diagnostics=[list(map(float,line.split('=',1)[1].split())) for line in text.splitlines()
                 if line.startswith('ASTR_CFL directional/local_sum/upper_bound=')]
    assert diagnostics and np.isfinite(diagnostics).all()
    record=dict(configuration=config,maximum_cfl=float(np.max(np.array(diagnostics)[:,-1])),
                directional_local_sum_upper_bound=diagnostics,
                checkpoint=str(checkpoint),**report_state(checkpoint,args.grid,args.wall,args.perturbation))
    if args.snapshots and prod(args.topology)==1 and args.restore is None and args.wall_mode=='hermite7':
        record['first_step_budget']=first_step_budget(output,args.grid,dt,args.wall)
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('mean_profiles','directional_local_sum_upper_bound')},indent=2))
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('cpu','gpu'),required=True)
    parser.add_argument('--grid',type=lambda s:tuple(map(int,s.split(','))),default=(32,32,16))
    parser.add_argument('--topology',type=lambda s:tuple(map(int,s.split(','))),default=(1,1,1))
    parser.add_argument('--steps',type=int,default=12)
    parser.add_argument('--checkpoint-interval',type=int)
    parser.add_argument('--dt',type=float)
    parser.add_argument('--perturbation',type=float,default=0.)
    parser.add_argument('--wall',type=int,choices=(41,42,421),default=41)
    parser.add_argument('--wall-mode',choices=('hermite7','off','default'),default='hermite7')
    parser.add_argument('--workspace',choices=('scalar','full'),default='scalar')
    parser.add_argument('--restore',type=Path)
    parser.add_argument('--snapshots',action='store_true')
    run(parser.parse_args())
