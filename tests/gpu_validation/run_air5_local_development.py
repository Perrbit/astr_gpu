#!/usr/bin/env python3
"""Approved local Mach-4 development window; ASTR alone advances the solution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

import h5py
import numpy as np

from air5_radau_reference import Air5RadauReference
from check_air5_mass_closure_stages import metrics
from check_air5_precursor_admission import monitor_windows
from insitu_cfl_gate import CflGate
from prepare_air5_mach4_case import ROOT, prepare
from run_air5_sbli_preflight import environment, set_value
from run_output_air5_restart_validation import state_sanity
from run_output_restart_validation import compare_fields


CAMPAIGN = ROOT/'tests/gpu_validation/out/air5_g2_64_20261013'
DT = 2.5e-10
GRID = (63, 127, 15)
BUDGET = 48*1024**3
BUFFER_BUDGET = 256*1024**2
MPIEXEC = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
VARIANTS = {'baseline': ('joint', 'every-step', 128),
            'combined': ('source-only', 'report-steps', 32)}


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def storage_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def prepare_window(case, variant, steps, preflight):
    primary = prepare(case, grid=','.join(map(str, GRID)), maxstep=steps-1,
                      deltat='2.5d-10', use_gpu='t')
    set_value(primary, 'nondimen,diffterm,lfilter,lreadgrid,lfftz,limmbou,ltimrpt,lcomb',
              'f,t,t,f,f,f,f,t,t')
    set_value(primary, 'lrestar', 'f')
    report_stride = 1 if preflight else 100
    set_value(case/'datin/controller', 'lwsequ,lwslic,lavg,lcracon', 'f,f,f,f')
    set_value(case/'datin/controller', 'maxstep,feqchkpt,feqwsequ,feqslice,feqlist,feqavg',
              f'{steps-1},{report_stride},1000000,1000000,{report_stride},1000000')
    # The inherited TGV probes would otherwise collect every completed step.
    (case/'datin/monitor.dat').unlink(missing_ok=True)
    (case/'outdat/new').mkdir(parents=True)
    (case/'datin/input.output').write_text(f"""&output
 directory='outdat/new',exact_restart=t,checkpoint_statistics=t,
 host_budget_bytes={BUFFER_BUDGET},device_budget_bytes={BUFFER_BUDGET},
 buffer_bytes=16777216,device_reserve_bytes=0
/
&checkpoint
 enabled=t,mode='steps',interval_steps={steps if preflight else 1000},
 keep=2,initial_frame=t
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    physics = json.loads((case/'mach4_case_metadata.json').read_text())
    physics['lfilter'] = True
    save(case/'mach4_case_metadata.json', physics)
    tau = physics['domain'][1]/(physics['edge_velocity']/4.)
    source, diagnostics, block = VARIANTS[variant]
    runtime = environment('2,1,1')
    runtime.update(ASTR_GPU_HALO_TRANSPORT='pinned', ASTR_GPU_FILTER_WORKSPACE='scalar',
        ASTR_AIR5_COMPENSATION='on', ASTR_AIR5_CONVECTION_LIMITER='symmetric_species',
        ASTR_AIR5_DIFFUSION_LIMITER='layered', ASTR_AIR5_TOP_MODE='characteristic',
        ASTR_AIR5_TOP_TAU=f'{tau:.17e}', ASTR_AIR5_TOP_GPU_VALIDATION='on',
        ASTR_AIR5_TOP_RESTART_VALIDATION='on', ASTR_AIR5_FILTER_VALIDATION='on',
        ASTR_AIR5_FILTER_MODE='conservative-flux', ASTR_AIR5_SOURCE_EVALUATION=source,
        ASTR_AIR5_CHEMISTRY_DIAGNOSTICS=diagnostics, ASTR_AIR5_CHEMISTRY_BLOCK_SIZE=str(block),
        ASTR_AIR5_CHEMISTRY_TOLERANCE='baseline', ASTR_AIR5_STEP_HISTORY='off',
        ASTR_AIR5_PRIMITIVE_REUSE='off', ASTR_AIR5_CHEMISTRY_REDUCTIONS='baseline',
        ASTR_AIR5_FLOW_MONITOR_STRIDE=str(report_stride), ASTR_CFL_DIAGNOSTICS='on',
        ASTR_CFL_PROFILE_Y='1', ASTR_COMPLETE_STEP_TIMING='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    record = dict(variant=variant, steps=steps, dt=DT, intervals=list(GRID),
        report_stride=report_stride, monitor_stride=report_stride, topology=[2,1,1],
        source_evaluation=source, chemistry_diagnostics=diagnostics, chemistry_block_size=block,
        exact_restart=True, checkpoint_keep=2, physics=physics,
        cfl_scope='every update in preflight; every 100 updates plus final-state bound in long window',
        initial_state='fresh quartic profile, no restart', accumulated_statistics=False,
        initial_diagnostic='one independent copy of initial state.h5, separate from rolling restart batches')
    return runtime, record


def final_state(path, physics):
    sanity = state_sanity(path)
    model = Air5RadauReference(ROOT/'chemMech/air5_kimjo12.json')
    with h5py.File(path) as stream:
        high = np.stack([stream[f'q{m:04}'][:] for m in range(1,12)], axis=-1)
        carry = np.stack([stream[f'q{m:04}'][:] for m in range(12,23)], axis=-1)
    results = {}
    for name, q in (('q', high), ('q-minus-carry', high-carry)):
        row = metrics(q, (0,0,0))
        if row['min_species_density'] < 0 or row['transport_sum_violations']:
            raise ValueError(f'{name}: species positivity/closure failed')
        density, species = q[...,0], q[...,5:10]
        velocity = q[...,1:4]/density[...,None]
        temperature = (q[...,4]-q[...,10]-species@model.formation_energy-
            .5*np.sum(q[...,1:4]**2,axis=-1)/density)/(species@model.cv_tr)
        pressure = (species@model.gas_constant)*temperature
        tmin, tmax = model.temperature_bounds
        pmin, pmax = model.pressure_bounds
        lower = np.array([model.species_vibrational_energy(s,tmin) for s in range(5)])
        upper = np.array([model.species_vibrational_energy(s,tmax) for s in range(5)])
        if (not np.isfinite(temperature).all() or not np.isfinite(pressure).all() or
                np.any((temperature<tmin)|(temperature>tmax)) or
                np.any((pressure<pmin)|(pressure>pmax)) or
                np.any(q[...,10]<species@lower) or np.any(q[...,10]>species@upper)):
            raise ValueError(f'{name}: reconstructed two-temperature model domain failed')
        sound = np.sqrt((1+(species@model.gas_constant)/(species@model.cv_tr))*
                        (species@model.gas_constant)*temperature/density)
        spacing = np.asarray(physics['domain'])/GRID
        cfl = DT*sum(float(np.max(abs(velocity[...,axis])+sound))/spacing[axis] for axis in range(3))
        if not np.isfinite(cfl) or cfl>.5:
            raise ValueError(f'{name}: final uniform-Cartesian CFL bound {cfl} > 0.5')
        results[name] = dict(min_species_density=row['min_species_density'],
            transport_sum_violations=row['transport_sum_violations'],
            max_extended_relative_residual=row['max_extended_relative_residual'],
            temperature_range=[float(temperature.min()),float(temperature.max())],
            pressure_range=[float(pressure.min()),float(pressure.max())],
            final_uniform_cartesian_cfl_upper_bound=float(cfl))
    return dict(cached_state=sanity, represented_states=results)


def run_case(case, executable, variant, steps, preflight, campaign):
    runtime, record = prepare_window(case, variant, steps, preflight)
    command = [str(MPIEXEC), '--mca', 'coll_hcoll_enable', '0', '-np', '2',
               str(executable), 'run', 'datin/input.air5_c4']
    record.update(command=command, environment={k:v for k,v in runtime.items()
        if k.startswith(('ASTR_','OMPI_','CUDA_','UCX_','OMP_','OPENBLAS_','MKL_'))})
    save(case/'launch.json', record)
    log_path = case/'run.log'
    gate = CflGate(log_path,.5,DT,0,steps,step_stride=record['report_stride'])
    initial_batch = case/'outdat/new/checkpoints/step000000000000'
    initial_copy = case/'initial_state.h5'
    started, reported = time.monotonic(), 0.
    with log_path.open('w') as log:
        process = subprocess.Popen(command,cwd=case,env=runtime,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                gate.check()
                if not initial_copy.exists() and (initial_batch/'COMPLETE').is_file():
                    shutil.copy2(initial_batch/'state.h5',initial_copy)
                elapsed = time.monotonic()-started
                if elapsed-reported>=60:
                    reported=elapsed
                    save(case/'progress.json',dict(pid=process.pid,elapsed_seconds=elapsed,
                        next_cfl_update_index=gate.next_step,max_sampled_cfl=gate.maximum))
                code=process.poll()
                if code is not None:
                    if code:
                        raise subprocess.CalledProcessError(code,command)
                    break
                if elapsed>86400:
                    raise TimeoutError('approved 24-hour per-run budget exceeded')
                time.sleep(1.)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
            raise
    gate.check(final=True)
    if storage_bytes(campaign)>BUDGET or storage_bytes(case)>1024**3:
        raise RuntimeError('approved campaign/case disk budget exceeded')
    final = case/f'outdat/new/checkpoints/step{steps:012d}/state.h5'
    wall = np.loadtxt(case/'monitor/air5_wall.dat',ndmin=2)
    if wall.shape[1]!=11 or not np.isfinite(wall).all():
        raise ValueError('missing/nonfinite wall heat components')
    result = dict(status='bounded-development-run-passed',elapsed_seconds=time.monotonic()-started,
        max_sampled_cfl=gate.maximum,cfl_report_stride=record['report_stride'],
        initial_state=str(initial_copy),final_state=str(final),physics=record['physics'],
        state=final_state(final,record['physics']),wall_rows=len(wall),
        wall_heat_sum_max_absolute_residual=float(np.max(abs(wall[:,7]-wall[:,9]-wall[:,10]))))
    if not preflight:
        profiles=np.loadtxt(case/'monitor/air5_profiles.dat',ndmin=2)
        windows=monitor_windows(profiles,wall,[0.,2.,4.,6.],record['physics']['edge_velocity'])
        denominator=record['physics']['edge_density']*record['physics']['edge_velocity']**2
        for station in windows:
            rows=wall[wall[:,2]==station['ix']]
            for window in station['windows']:
                lo,hi=window['interval_us']
                selected=rows[(rows[:,1]>=lo*1e-6)&(rows[:,1]<hi*1e-6)]
                window.update(mean_cf=2*window['mean_tau']/denominator,
                    mean_heat_tr=float(selected[:,9].mean()),mean_heat_v=float(selected[:,10].mean()))
        save(case/'monitor_windows.json',dict(scope='initial development, not stationarity or physical admission',
            monitor='fixed z plane, not spanwise average',windows=windows))
    save(case/'result.json',result)
    return result


def run(args):
    out=args.output.resolve()
    out.relative_to(CAMPAIGN.resolve())
    steps=100 if args.stage=='preflight' else 24000
    if args.stage=='preflight':
        out.mkdir(parents=True,exist_ok=False)
        shutil.copy2(args.executable.resolve(),out/'astr')
        report=dict(status='preflight-running',scope='approved local Mach-4 development, not physical SBLI admission',
            executable_sha256=hashlib.sha256((out/'astr').read_bytes()).hexdigest(),runs={})
    else:
        report=json.loads((out/'result.json').read_text())
        if report['status']!='preflight-passed':
            raise ValueError('long window requires the matched 100-step preflight')
        if hashlib.sha256((out/'astr').read_bytes()).hexdigest()!=report['executable_sha256']:
            raise ValueError('frozen executable changed after preflight')
        report['status']='long-window-running'
    save(out/'result.json',report)
    try:
        if storage_bytes(CAMPAIGN)>BUDGET:
            raise RuntimeError('approved campaign disk budget exceeded')
        for variant in VARIANTS:
            if storage_bytes(CAMPAIGN)+1024**3>BUDGET:
                raise RuntimeError('campaign lacks the approved per-run disk reservation')
            name=f'{args.stage}_{variant}'
            print(f'CASE_START {name} steps={steps}',flush=True)
            report['active_case']=name
            save(out/'result.json',report)
            row=run_case(out/name,out/'astr',variant,steps,args.stage=='preflight',CAMPAIGN)
            report['runs'][name]=row
            save(out/'result.json',report)
            print(f'CASE_END {name} elapsed={row["elapsed_seconds"]:.3f}',flush=True)
        baseline,combined=[out/f'{args.stage}_{name}' for name in VARIANTS]
        equal={phase:compare_fields(Path(report['runs'][f'{args.stage}_baseline'][phase]),
                                  Path(report['runs'][f'{args.stage}_combined'][phase]))
               for phase in ('initial_state','final_state')}
        for name in ('air5_profiles.dat','air5_wall.dat'):
            if (baseline/'monitor'/name).read_bytes()!=(combined/'monitor'/name).read_bytes():
                raise ValueError(f'engineering monitor mismatch: {name}')
        report[f'{args.stage}_exact_datasets']=equal
        report['status']='preflight-passed' if args.stage=='preflight' else 'six-us-development-passed-not-physical-admission'
        report['campaign_bytes']=storage_bytes(CAMPAIGN)
        if report['campaign_bytes']>BUDGET:
            raise RuntimeError('approved campaign disk budget exceeded')
        report.pop('active_case',None)
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        save(out/'result.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('preflight','long'),default='preflight')
    parser.add_argument('--executable',type=Path,default=ROOT/'build_gpu_probe/bin/astr')
    run(parser.parse_args())
