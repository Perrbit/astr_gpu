#!/usr/bin/env python3
"""Small-grid ASTR preflight with the A800 CPU/rank/topology matrix."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

import numpy as np

from compact_benchmark import CPU_TOPOLOGIES, configuration_matrix, mpi_environment, timing_samples
from compare_q_validation_snapshots import read_q_snapshot, comparison_values
from compare_tgv_flowstate import read_flowstate
from run_compact_derivative_tgv import assemble_final, measure

ROOT=Path(__file__).resolve().parents[2]


def rank_exec(exe, case):
    rank=int(os.environ['OMPI_COMM_WORLD_RANK'])
    cpus=sorted(os.sched_getaffinity(0))
    cores=set()
    for cpu in cpus:
        topology=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        cores.add((int((topology/'physical_package_id').read_text()),int((topology/'core_id').read_text())))
    (case/f'affinity.rank{rank:08d}.json').write_text(json.dumps(dict(rank=rank,cpus=cpus,physical_cores=sorted(cores)))+'\n')
    os.execv(str(exe),[str(exe),'run','datin/input.case'])


def prepare_and_run(args, row, name, timing=True, validation=True):
    case=args.out/name
    backend=row['backend'];ranks=row['ranks'];steps=args.steps
    command=[sys.executable,str(ROOT/'tests/gpu_validation/prepare_tgv_case.py'),
        '--src-case',str(ROOT/'examples/Taylor_Green_Vortex'),'--dst-case',str(case),
        '--use-gpu','t' if backend=='gpu' else 'f','--scheme',row['scheme'],
        '--grid',','.join(map(str,row['grid'])),'--maxstep',str(steps-1),
        '--feqchkpt','1','--feqlist','1','--lfilter','t','--diffterm','t',
        '--deltat','1.d-4','--lreadgrid','f','--restart','f','--omit-grid-resource']
    subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
    env=mpi_environment(backend,args.halo_transport,args.compact_transport,args.ucx_tls)
    env.update(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,row['topology'])),
        ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_COMPACT_SYNC='dependency',
        ASTR_GPU_COMPACT_BATCH_RHS='65535',ASTR_GPU_COMPACT_TRANSPORT=args.compact_transport,
        ASTR_GPU_HALO_TRANSPORT=args.halo_transport,ASTR_GPU_FILTER_WORKSPACE='full',
        ASTR_GPU_BENCHMARK_NO_FIELD_IO='1',ASTR_CPU_RK_TIMING='1' if backend=='cpu' and timing else '0',
        ASTR_GPU_RK_TIMING='1' if backend=='gpu' else '0',ASTR_GPU_RANK_RK_TIMING='1',
        ASTR_BENCHMARK_BUFFER_OUTPUT='1' if backend=='gpu' else '0',
        ASTR_COMPLETE_STEP_TIMING='1')
    if validation:
        env.update(ASTR_VALIDATION_RHS_PREFIX=str(case/'state'),ASTR_VALIDATION_RHS_STEP=str(steps-1))
    if backend=='gpu':
        env['ASTR_GPU_PRECISION_MODE']='fp64'
    from named_fixture import prepare_named_run
    prepare_named_run(case,'datin/input.tgv',env)
    exe=args.gpu if backend=='gpu' else args.cpu
    mapping='slot:PE=2' if backend=='gpu' else 'core'
    launch=['mpirun','--map-by',mapping,'--bind-to','core','--report-bindings','-np',str(ranks),
            sys.executable,str(Path(__file__).resolve()),'rank-exec',str(exe),str(case)]
    (case/'launch.json').write_text(json.dumps(dict(command=launch,environment={k:v for k,v in env.items()
        if k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}),indent=2)+'\n')
    started=time.monotonic()
    with (case/'run.log').open('w') as stream:
        subprocess.run(launch,cwd=case,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    text=(case/'run.log').read_text(errors='replace')
    if 'The job is done!' not in text or 'COMPUTATION CRASHED' in text:
        raise ValueError(f'ASTR did not complete: {case}')
    cfl=[float(x) for x in re.findall(r'current CFL:\s*(\S+)',text)]
    if not cfl or not np.isfinite(cfl).all() or max(cfl)>=1:
        raise ValueError(f'invalid CFL: {case}: {cfl}')
    if row['scheme']=='643c' and 'COMPACT GLOBAL: periodic unique-node' not in text:
        raise ValueError(f'global compact operator not selected: {case}')
    if backend=='gpu' and f'ASTR_GPU_HALO_TRANSPORT_SELECTED={args.halo_transport}' not in text:
        raise ValueError(f'halo transport not selected: {case}')
    bindings=[json.loads((case/f'affinity.rank{rank:08d}.json').read_text()) for rank in range(ranks)]
    if backend=='cpu':
        cores=[tuple(item['physical_cores'][0]) for item in bindings]
        if any(len(item['physical_cores'])!=1 for item in bindings) or len(set(cores))!=ranks:
            raise ValueError('CPU ranks must occupy distinct physical cores')
    field=None
    minima={}
    if validation:
        field=assemble_final(case,row['grid'],steps-1)
        rho=field[...,0]
        internal=field[...,4]-np.sum(field[...,1:4]**2,axis=-1)/(2*rho)
        if not np.isfinite(field).all() or rho.min()<=0 or internal.min()<=0:
            raise ValueError(f'nonpositive/nonfinite completed state: {case}')
        minima=dict(rho_min=float(rho.min()),internal_min=float(internal.min()))
    if list(case.rglob('*.h5')):
        raise ValueError(f'unexpected HDF5 field output: {case}')
    values=timing_samples(text,backend,ranks,steps,2) if timing else []
    report=dict(**row,case=str(case),timing_enabled=timing,launch_seconds=time.monotonic()-started,
        cfl_max=max(cfl),**minima,
        rank_max_rk_samples=values,median_rk_seconds=statistics.median(values) if values else None,
        affinity=bindings)
    devices=re.findall(r'\*\* MPI rank\s+(\d+) bound to GPU device\s+(\d+)',text)
    report['gpu_bindings']=[list(map(int,item)) for item in devices]
    report['gpu_sharing_correctness_only']=backend=='gpu' and len({device for _,device in devices})<ranks
    (case/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return case,field,report


def compare_pair(cpu, gpu, ranks, steps):
    largest=0.;absolute=0.
    for rank in range(ranks):
        for stage in (1,2,3):
            suffix=f'state.post_update.step{steps-1:08d}.rk{stage:02d}.rank{rank:08d}.bin'
            arrays=[]
            for directory in (cpu,gpu):
                snap=read_q_snapshot(directory/suffix)
                arrays.append(comparison_values(snap,True).reshape((-1,5),order='F'))
            error=measure(*arrays)
            largest=max(largest,max(error['scaled']));absolute=max(absolute,max(error['abs']))
    tables=[np.array(read_flowstate(directory)[1]) for directory in (cpu,gpu)]
    if tables[0].shape!=tables[1].shape or not np.array_equal(tables[0][:,0],tables[1][:,0]):
        raise ValueError('statistics sample identities differ')
    stats=float(np.max(np.abs(tables[0][:,1:]-tables[1][:,1:])))
    if stats>2e-10:
        raise ValueError(f'statistics maximum absolute error: {stats}')
    return dict(max_scaled=largest,max_abs=absolute,statistics_max_abs=stats)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10)
    parser.add_argument('--strong-grid',type=int,default=32)
    parser.add_argument('--weak-local',type=int,default=16)
    parser.add_argument('--halo-transport',choices=('pinned','device-aware'),default='device-aware')
    parser.add_argument('--compact-transport',choices=('pinned','device-aware'),default='device-aware')
    parser.add_argument('--ucx-tls',default='self,sm,cuda_copy')
    args=parser.parse_args()
    args.cpu=args.cpu.resolve(strict=True);args.gpu=args.gpu.resolve(strict=True);args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    rows=configuration_matrix(args.strong_grid,args.weak_local)
    (args.out/'matrix.json').write_text(json.dumps(rows,indent=2)+'\n')
    report=dict(status='running',scope='local small-grid correctness, not A800 performance',
        cpu_sha256=hashlib.file_digest(args.cpu.open('rb'),'sha256').hexdigest(),
        gpu_sha256=hashlib.file_digest(args.gpu.open('rb'),'sha256').hexdigest(),runs=[],pairs=[],cpu_topology=[])
    completed={};baselines={}
    try:
        for row in rows:
            topology=''.join(map(str,row['topology']))
            name=f"{row['family']}_{row['scheme']}_{row['backend']}_np{row['ranks']}_{topology}"
            print('START',name,flush=True)
            case,field,result=prepare_and_run(args,row,name)
            report['runs'].append(result)
            key=(row['scheme'],row['grid'],row['topology'])
            completed[row['backend'],key]=(case,field)
            if row['backend']=='cpu' and row['family']=='strong':
                baseline=baselines.setdefault(row['scheme'],field)
                report['cpu_topology'].append(dict(case=name,**measure(baseline,field)))
            if row['backend']=='gpu':
                cpu=completed['cpu',key][0]
                report['pairs'].append(dict(case=name,**compare_pair(cpu,case,row['ranks'],args.steps)))
            (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
            print('PASS',name,flush=True)
        report['timing_invariance']=[]
        for scheme in ('643e','643c'):
            row=next(r for r in rows if r['backend']=='cpu' and r['scheme']==scheme and r['family']=='strong' and r['topology']==(2,1,1))
            case,field,result=prepare_and_run(args,row,f'{scheme}_cpu_timing_off',timing=False)
            baseline=completed['cpu',(scheme,row['grid'],row['topology'])][1]
            if not np.array_equal(field,baseline):
                raise ValueError('CPU timing changed the completed state')
            report['timing_invariance'].append(dict(scheme=scheme,bitwise_equal=True,case=str(case)))
        report['status']='pass'
    except Exception as error:
        report.update(status='fail',error=str(error))
        raise
    finally:
        (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],runs=len(report['runs']),pairs=len(report['pairs']),
        max_scaled=max(r['max_scaled'] for r in report['pairs']),
        stats_max=max(r['statistics_max_abs'] for r in report['pairs'])),indent=2),flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='rank-exec':
        rank_exec(Path(sys.argv[2]),Path(sys.argv[3]))
    else:
        main()
