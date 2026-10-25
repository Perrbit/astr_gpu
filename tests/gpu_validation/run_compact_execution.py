#!/usr/bin/env python3
"""CP8/CP9 local GPU runs; optional complete checkpoints for accuracy studies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

from prepare_m12_local_case import prepare
from run_compact_flatplate import configure, GRID

ROOT = Path(__file__).resolve().parents[2]


def output_configuration(checkpoint_interval=0):
    if not checkpoint_interval:
        return '&output\n host_budget_bytes=67108864\n/\n&checkpoint\n/\n&volume\n/\n&slices\n/\n'
    return f"""&output
 directory='outdat/output', exact_restart=t,
 host_budget_bytes=68719476736, device_budget_bytes=68719476736,
 buffer_bytes=67108864
/
&checkpoint
 enabled=t, mode='steps', interval_steps={checkpoint_interval}, keep=2, initial_frame=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
"""


def retained_checkpoints(case, steps, interval):
    scheduled = list(range(interval, steps+1, interval))
    if not scheduled or scheduled[-1]!=steps:
        scheduled.append(steps)
    expected = [f'step{step:012d}' for step in scheduled[-2:]]
    root = case/'outdat/output/checkpoints'
    batches = sorted(path.name for path in root.iterdir() if path.is_dir())
    if batches!=expected:
        raise ValueError(f'checkpoint retention differs: {batches}, expected {expected}')
    # Publication markers follow successful file closes; do not rescan field payloads.
    for name in batches:
        if not all((root/name/filename).is_file() for filename in ('state.h5','COMPLETE')):
            raise ValueError(f'checkpoint publication incomplete: {name}')
    if not (case/'outdat/output/resources/geometry.h5').is_file():
        raise ValueError('complete checkpoint geometry resource missing')
    return [str(root/name) for name in batches]


def read_rank_output(case, ranks):
    files=sorted((case/'rank_output/1').glob('rank.*/stdout'),
                 key=lambda path:int(path.parent.name.split('.')[1]))
    if [int(path.parent.name.split('.')[1]) for path in files]!=list(range(ranks)):
        raise ValueError('missing rank-local MPI output files')
    return '\n'.join(path.read_text()+'\n'+path.with_name('stderr').read_text() for path in files)


def crashfix_summary(text):
    rows = re.findall(r'^ASTR_GPU_CRASHFIX\s+(\d+)\s+(\d+)\s+(\d+)\s+([^\n]+)', text, re.MULTILINE)
    totals = re.findall(r'^ASTR_GPU_CRASHFIX_TOTAL\s+(\d+)\s+(\d+)\s+([^\n]+)', text, re.MULTILINE)
    if totals:
        if len(totals) != 1 or rows:
            raise ValueError('mixed/duplicate buffered crashfix report')
        visits, calls, payload = totals[0]
        values = np.array([float(value) for value in payload.split()])
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError('invalid buffered crashfix report')
        return dict(crashfix_repaired_node_visits=int(visits), crashfix_nonzero_calls=int(calls),
            crashfix_max_pointwise_conservative_change=values[:5].tolist(),
            crashfix_reported_state_minima=values[5:].tolist(),
            crashfix_change_scope='per-component pointwise maxima, not summed conservation defects')
    maximum = np.zeros(5)
    minima = None
    for _, _, _, payload in rows:
        values = np.array([float(value) for value in payload.split()])
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError('invalid crashfix component-change report')
        maximum = np.maximum(maximum, values[:5])
        minima = values[5:] if minima is None else np.minimum(minima, values[5:])
    return dict(crashfix_repaired_node_visits=sum(int(count) for _, _, count, _ in rows),
        crashfix_nonzero_calls=sum(int(count) > 0 for _, _, count, _ in rows),
        crashfix_max_pointwise_conservative_change=maximum.tolist(),
        crashfix_reported_state_minima=None if minima is None else minima.tolist(),
        crashfix_change_scope='per-component pointwise maxima, not summed conservation defects')


def window_times(text, ranks, steps, warmup):
    rows=re.findall(r'ASTR_GPU_RANK_RK_TIMING\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',text)
    records={(int(rank),int(step)):float(total) for rank,step,prepare_time,rk,total in rows}
    expected={(rank,step) for rank in range(ranks) for step in range(steps)}
    if len(rows)!=len(expected) or set(records)!=expected:
        raise ValueError('missing/duplicate rank/complete-step timing records')
    if not all(np.isfinite(value) and value>=0 for value in records.values()):
        raise ValueError('invalid RK timing')
    sums=[sum(records[rank,step] for step in range(warmup,steps)) for rank in range(ranks)]
    return sums,max(sums)/(steps-warmup)


def phase_times(text,ranks,steps,warmup,filtering='t'):
    rows=re.findall(r'ASTR_GPU_PHASE_TIMING\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)',text)
    records={}
    for rank,step,stage,label,seconds in rows:
        key=(int(rank),int(step),int(stage),label)
        if key in records:
            raise ValueError(f'duplicate phase record: {key}')
        records[key]=float(seconds)
    required=('convection','diffusion_flux','diffusion_halo','diffusion_rhs','rk_update')
    if filtering=='t':
        required=('filter',*required)
    for label in required:
        for rank in range(ranks):
            for step in range(steps):
                for stage in (1,2,3):
                    value=records.get((rank,step,stage,label))
                    if value is None or not np.isfinite(value) or value<0:
                        raise ValueError(f'missing/invalid phase: {(rank,step,stage,label)}')
    result={}
    for label in sorted({key[3] for key in records}):
        sums=[sum(value for (r,step,stage,name),value in records.items()
                  if r==rank and warmup<=step<steps and name==label) for rank in range(ranks)]
        result[label]=dict(rank_window_seconds=sums,seconds_per_step=max(sums)/(steps-warmup))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=('tgv', 'plate'), default='tgv')
    parser.add_argument('--source', type=Path, default=ROOT.parent/'M12_C13')
    parser.add_argument('--topology', default='2,1,1')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--grid', help='interval counts; default 32^3 TGV or CP6 plate')
    parser.add_argument('--scheme', choices=('643e','643c'), default='643c')
    parser.add_argument('--filter', choices=('f','t'), default='t')
    parser.add_argument('--deltat', type=float)
    parser.add_argument('--phase-timing', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='read-only q admissibility and resource sampling')
    parser.add_argument('--state-check', action='store_true', help='read-only complete-state check without resource sampling')
    parser.add_argument('--stage-state-check', action='store_true', help='test-only profile stage localization')
    parser.add_argument('--filter-window', help='test-only NP1 periodic-z step,stage,i,j,k capture')
    parser.add_argument('--crashfix', action='store_true', help='enable existing runtime lcracon recovery')
    parser.add_argument('--workspace', choices=('scalar','full'), default='full')
    parser.add_argument('--compact-sync', choices=('explicit','dependency'), default='explicit')
    parser.add_argument('--batch-rhs', type=int, default=65535)
    parser.add_argument('--transport', choices=('pinned','device-aware'), default='pinned')
    parser.add_argument('--nsys', action='store_true')
    parser.add_argument('--native-precision', choices=('fp64','fp32'), default='fp64')
    parser.add_argument('--statistics-interval', type=int, default=1)
    parser.add_argument('--buffer-output', action='store_true')
    parser.add_argument('--tgv-conservation', action='store_true')
    parser.add_argument('--listing-interval', type=int, help='lightweight progress/CFL interval')
    parser.add_argument('--checkpoint-interval', type=int, default=0,
                        help='complete-step checkpoint interval; retain two, default off')
    args = parser.parse_args()
    if not 0<=args.warmup<args.steps:
        parser.error('require 0 <= warmup < steps')
    if args.crashfix and args.case!='plate':
        parser.error('--crashfix is scoped to the CP9 profile plate')
    if args.statistics_interval<0:
        parser.error('--statistics-interval must be nonnegative')
    if args.checkpoint_interval<0:
        parser.error('--checkpoint-interval must be nonnegative')
    if args.tgv_conservation and (args.case!='tgv' or args.statistics_interval==0):
        parser.error('--tgv-conservation requires TGV statistics')
    if args.listing_interval is not None and args.listing_interval<1:
        parser.error('--listing-interval must be positive')
    case=args.out.resolve()
    exe=args.exe.resolve(strict=True)
    grid=args.grid or ('32,32,32' if args.case=='tgv' else ','.join(map(str,GRID)))
    dt=args.deltat if args.deltat is not None else (1e-3 if args.case=='tgv' else .005)
    if args.case=='tgv':
        prepare_command=[sys.executable,str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
            '--src-case',str(ROOT/'examples/Taylor_Green_Vortex'),'--dst-case',str(case),
            '--grid',grid,'--use-gpu','t','--scheme',args.scheme,'--lfilter',args.filter,
            '--diffterm','t','--deltat',str(dt),'--maxstep',str(args.steps-1)]
        if args.buffer_output:
            prepare_command+=['--omit-grid-resource','--lreadgrid','f','--restart','f']
        subprocess.run(prepare_command,check=True)
        primary='datin/input.tgv'
    else:
        prepare(args.source,case,'gpu',tuple(map(int,grid.split(','))))
        configure(case,'nscbc',args.scheme,False,args.filter,args.crashfix)
        controller=case/'datin/controller'
        lines=controller.read_text().splitlines()
        index=next(i for i,line in enumerate(lines) if '# maxstep,feqchkpt' in line)+1
        fields=lines[index].split(',');fields[0]=str(args.steps-1)
        lines[index]=','.join(fields)
        from prepare_m12_local_case import replace_value
        replace_value(lines,'deltat',str(dt))
        controller.write_text('\n'.join(lines)+'\n')
        metadata=json.loads((case/'fixture.json').read_text())
        metadata.update(deltat=dt,crashfix=args.crashfix)
        (case/'fixture.json').write_text(json.dumps(metadata,indent=2)+'\n')
        primary='datin/input.dat'
    (case/'datin/input.output').write_text(output_configuration(args.checkpoint_interval))
    if args.checkpoint_interval:
        (case/'outdat/output').mkdir(parents=True)
    if args.buffer_output:
        controller=case/'datin/controller'
        lines=controller.read_text().splitlines()
        index=next(i for i,line in enumerate(lines) if '# maxstep,feqchkpt' in line)+1
        fields=lines[index].split(',')
        fields[1]=str(args.steps+1)
        fields[4]=str(args.listing_interval or args.steps+1)
        lines[index]=','.join(fields)
        controller.write_text('\n'.join(lines)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=args.topology,ASTR_GPU_COMPACT_SYNC=args.compact_sync,
        ASTR_GPU_COMPACT_BATCH_RHS=str(args.batch_rhs),ASTR_GPU_COMPACT_TRANSPORT=args.transport,
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_PRECISION_MODE='fp64' if args.native_precision=='fp64' else '',
        ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_GPU_FILTER_WORKSPACE=args.workspace,ASTR_GPU_RK_TIMING='1',ASTR_GPU_RANK_RK_TIMING='1',
        ASTR_COMPLETE_STEP_TIMING='0' if args.buffer_output else '1',
        ASTR_GPU_STATISTICS_INTERVAL=str(args.statistics_interval),
        ASTR_BENCHMARK_BUFFER_OUTPUT='1' if args.buffer_output else '0',
        ASTR_NSCBC_FARFIELD_MODE='incoming_only',ASTR_NSCBC_FARFIELD_RHO='1',ASTR_NSCBC_FARFIELD_U='1',
        ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',ASTR_NSCBC_FARFIELD_T='1',
        OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_opal_cuda_support='0')
    if args.case=='plate':
        env['ASTR_WALL_BLOWING_MODE']='legacy_random'
    if args.buffer_output and not args.checkpoint_interval:
        env['ASTR_GPU_BENCHMARK_NO_FIELD_IO']='1'
    if args.tgv_conservation:
        env['ASTR_TGV_CONSERVATION']='1'
    if args.phase_timing:
        env['ASTR_GPU_PHASE_TIMING']='1'
    if args.preflight or args.state_check:
        env['ASTR_VALIDATION_RK_STATE']='1'
    if args.stage_state_check:
        env['ASTR_VALIDATION_PROFILE_STAGE_STATE']='1'
    if args.filter_window:
        env['ASTR_VALIDATION_PROFILE_FILTER_WINDOW']=args.filter_window
    if args.transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx',UCX_TLS='self,sm,cuda_copy',UCX_MEMTYPE_CACHE='n',
            UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',UCX_CUDA_IPC_ENABLE_MNNVL='no',
            OMPI_MCA_coll_hcoll_enable='0')
        env.pop('OMPI_MCA_opal_cuda_support')
    ranks=int(np.prod(tuple(map(int,args.topology.split(',')))))
    command=['mpirun','--oversubscribe','--bind-to','none','-np',str(ranks)]
    if args.buffer_output:
        command+=['--output-filename',str(case/'rank_output')]
    if args.nsys:
        command+=['nsys','profile','--trace=cuda,nvtx,mpi','--mpi-impl=openmpi','--sample=none',
            '--cpuctxsw=none','--export=sqlite','--output',str(case/'trace.rank%q{OMPI_COMM_WORLD_RANK}')]
    command += [str(exe),'run',primary]
    (case/'launch.json').write_text(json.dumps(dict(command=command,environment={k:v for k,v in env.items()
        if k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}),indent=2)+'\n')
    started=time.monotonic()
    with (case/'run.log').open('w') as stream:
        if args.preflight:
            from insitu_resource_monitor import run_monitored
            run_monitored(command,case,env,stream,case/'resources.json',timeout_seconds=3600)
        else:
            subprocess.run(command,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    elapsed=time.monotonic()-started
    text=read_rank_output(case,ranks) if args.buffer_output else (case/'run.log').read_text()
    if f'ASTR_GPU_PRECISION_MODE={args.native_precision}' not in text:
        raise ValueError('reported solver precision differs from the benchmark selection')
    sums,cost=window_times(text,ranks,args.steps,args.warmup)
    cfl=[float(v) for v in re.findall(r'current CFL:\s+(\S+)',text)]
    if not cfl or not np.isfinite(cfl).all() or max(cfl)>.5:
        raise ValueError(f'CFL preflight gate failed: {cfl}')
    if args.preflight or args.state_check:
        checks=re.findall(r'ASTR_VALIDATION_RK_STATE_PASS\s+(\d+)\s+(\d+)',text)
        if {(int(r),int(s)) for r,s in checks}!={(r,s) for r in range(ranks) for s in range(args.steps)}:
            raise ValueError('missing read-only complete-state admission records')
    report=dict(status='pass',case=args.case,topology=args.topology,steps=args.steps,warmup=args.warmup,
        compact_sync=args.compact_sync,batch_rhs=args.batch_rhs,workspace=args.workspace,transport=args.transport,
        grid_intervals=list(map(int,grid.split(','))),scheme=args.scheme,filter=args.filter,deltat=dt,
        phase_timing=args.phase_timing,preflight=args.preflight,stage_state_check=args.stage_state_check,
        state_check=args.preflight or args.state_check,
        crashfix=args.crashfix,numerical_precision=args.native_precision,
        statistics_interval=args.statistics_interval,buffered_output=args.buffer_output,
        tgv_conservation=args.tgv_conservation,listing_interval=args.listing_interval,
        cfl=cfl,launch_elapsed_seconds=elapsed,
        packed_device_bytes={name:int(value) for name,value in re.findall(
            r'COMPACT_(DERIVATIVE|FILTER)_PACKED_DEVICE_BYTES=(\d+)',text)},
        profiled=args.nsys,rank_rk_window_seconds=sums,rk_seconds_per_step=cost,
        executable_sha256=hashlib.file_digest(exe.open('rb'),'sha256').hexdigest())
    report['checkpoint_interval']=args.checkpoint_interval
    if args.checkpoint_interval:
        report['retained_checkpoints']=retained_checkpoints(case,args.steps,args.checkpoint_interval)
    report.update(crashfix_summary(text))
    if args.preflight:
        report['sampled_preflight_resources'] = json.loads((case/'resources.json').read_text())
    if args.phase_timing:
        report['phases']=phase_times(text,ranks,args.steps,args.warmup,args.filter)
    (case/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
