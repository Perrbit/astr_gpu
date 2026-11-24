"""Prepare and reduce the 256-node weak-scaling cases. No scheduler operations."""
import argparse
from collections import defaultdict
import json
from math import isclose, isfinite, prod
from pathlib import Path
import re
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/input'))
from convert_legacy_input import namelist


TILES = [(1,1,1), (1,1,2), (2,1,2), (2,2,2), (2,2,4),
         (4,2,4), (4,4,4), (4,4,8), (8,4,8)]
NODE_BLOCK = (2,2,1)
CASES = {
    'explicit': (488, '643e', 1e-4),
    'compact': (442, '643c', 1e-4),
    'air5': (336, '643e', 2e-10),
}
WARMUP = 2
TIMED = 3


def shape_product(a, b):
    return tuple(x*y for x,y in zip(a,b))


def rank_blocks(tiles):
    px,py,_ = shape_product(tiles,NODE_BLOCK)
    blocks = []
    for z in range(tiles[2]):
        for y in range(tiles[1]):
            for x in range(tiles[0]):
                origin = 2*x+px*(2*y+py*z)
                blocks.append([origin,origin+1,origin+px,origin+px+1])
    return blocks


def placement_check(config, actual):
    px,py,pz = config['mpi']
    hosts = defaultdict(set)
    seen = set()
    for node in actual:
        rank = node['rank']
        if rank in seen or not 0<=rank<px*py*pz:
            raise ValueError(f'duplicate or invalid rank {rank}')
        coords = (rank%px,rank//px%py,rank//(px*py))
        if tuple(node['coords']) != coords:
            raise ValueError(f'ASTR coordinate mismatch for rank {rank}')
        seen.add(rank)
        hosts[node['host']].add(rank)
    if {frozenset(v) for v in hosts.values()} != {frozenset(v) for v in config['rank_blocks']}:
        raise ValueError('observed node mapping does not preserve the 2x2x1 rank blocks')


def case_configuration(kind, base, tiles, mpi, steps=WARMUP+TIMED):
    _,scheme,dt = CASES[kind]
    air5 = kind == 'air5'
    mesh = shape_product((base,)*3,tiles)
    env = dict(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,mpi)), ASTR_GPU_HALO_TRANSPORT='device',
               ASTR_GPU_SYNC_MODE='dependency', ASTR_GPU_BENCHMARK_NO_FIELD_IO='1', ASTR_GPU_RK_TIMING='1')
    if kind == 'explicit':
        env.update(ASTR_GPU_FILTER_WORKSPACE='scalar', ASTR_CPP_VISCOUS_HALO='packed', ASTR_CPP_VISCOUS_OVERLAP_AXIS='z')
    elif kind == 'compact':
        env.update(ASTR_GPU_FILTER_WORKSPACE='scalar', ASTR_GPU_COMPACT_TRANSPORT='device-aware',
                   ASTR_GPU_COMPACT_SYNC='dependency', ASTR_GPU_COMPACT_BATCH_RHS='65535', ASTR_GPU_COMPACT_PIPELINE='off')
    else:
        env.update(ASTR_COMPLETE_STEP_TIMING='1', ASTR_AIR5_FLUX_HALO='packed',
                   ASTR_AIR5_CHEMISTRY_DISPATCH='specialized', ASTR_AIR5_CHEMISTRY_BLOCK_SIZE='128')
    return dict(kind=kind,nodes=prod(tiles),tiles=tiles,grid=mesh,mpi=mpi,scheme=scheme,dt=dt,
                timesteps=steps,warmup=WARMUP,repetitions=3 if prod(tiles)==1 else 2,
                timing='complete' if air5 else 'rk',environment=env)


def write_inputs(directory, config):
    datin = directory/'datin'
    datin.mkdir(parents=True,exist_ok=False)
    (directory/'outdat').mkdir()
    (directory/'outdat/new').mkdir()
    kind = config['kind']
    air5 = kind == 'air5'
    ia,ja,ka = config['grid']
    groups = dict(
        case_options=dict(flowtype='air5tgv' if air5 else 'tgv',use_gpu=True,ltimrpt=True,iomode='h'),
        mesh=dict(ia=ia,ja=ja,ka=ka,lihomo=True,ljhomo=True,lkhomo=True,lreadgrid=False,
                  lfftk=False,benchmark_tgv_tiles=list(config['tiles'])),
        physics=dict(nondimen=not air5,diffterm=True,lcomb=air5,num_species=5 if air5 else 0,
                     turbmode='none',ref_tem=1000. if air5 else 273.15),
        numerics=dict(convection_form='standard',viscous_form='standard',rkscheme='rk3',lfilter=not air5,
                      alfa_filter=.49,kcutoff=48,conschm=config['scheme'],difschm=config['scheme'],
                      filter_scheme='compact10' if kind=='compact' else 'explicit10',filter_scope='xyz'),
        boundary=dict(bctype=[1]*6,twall=[0]*6),
        initialization=dict(lrestart=False,ninit=0,limmbou=False),
        precision_options=dict(gpu_precision_mode='fp64'))
    if air5:
        groups['physics'].update(ref_len=.01,ref_vel=10.,ref_den=1.)
        groups['species_transport'] = dict(schmidt=[1.]*5)
        groups['chemistry'] = dict(source_mode='coupled',convection_limiter='symmetric_species',
                                   diffusion_limiter='layered',compensation='off')
    else:
        groups['physics'].update(reynolds=1600.,mach=.1)
    (datin/'input.case').write_text(namelist(groups))
    (datin/'input.run').write_text(namelist(dict(run_control=dict(max_steps=config['timesteps']),
        timestep=dict(mode='fixed',fixed_dt=config['dt']),logging=dict(list_steps=1),
        repair=dict(crashfix=False),statistics=dict(enabled=False))))
    (datin/'input.output').write_text(namelist(dict(output=dict(directory='outdat/new',exact_restart=False,
        host_budget_bytes=-1,device_budget_bytes=-1,device_reserve_bytes=0),
        checkpoint=dict(enabled=False),volume=dict(enabled=False),slices=dict(enabled=False))))


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    configurations = {}
    for kind,(base,_,_) in CASES.items():
        for tiles in TILES:
            config = case_configuration(kind,base,tiles,shape_product(tiles,NODE_BLOCK))
            config['rank_blocks'] = rank_blocks(tiles)
            name = f'{kind}_{config["nodes"]:03d}nodes'
            write_inputs(root/'cases'/name,config)
            configurations[name] = config
    (root/'campaign.json').write_text(json.dumps(configurations,indent=2)+'\n')
    print(f'Prepared {len(configurations)} case templates. No large files or launch scripts were generated.')


TIMESTEP = re.compile(r'^ASTR_TIMESTEP complete_step=(\d+)\s+time_start=\s*(\S+)'
                      r'\s+time_end=\s*(\S+)\s+dt_used=\s*(\S+)\s+dt_next=\s*(\S+)$',re.M)


def reduce_run(log, config):
    steps,dt = config['timesteps'],config['dt']
    clocks = TIMESTEP.findall(log)
    clock_ok = (len(clocks)==steps and all(
        int(step)==idx and isclose(float(start),(idx-1)*dt,rel_tol=1e-12)
        and isclose(float(end),idx*dt,rel_tol=1e-12)
        and float(used)==float(next_dt)==dt
        for idx,(step,start,end,used,next_dt) in enumerate(clocks,1)))
    if not clock_ok:
        raise ValueError('wrong fixed-step clock or incomplete run')
    complete = config['timing']=='complete'
    prefix = 'ASTR_COMPLETE_STEP_TIMING' if complete else 'ASTR_GPU_RANK_RK_TIMING'
    rows = {}
    for line in log.splitlines():
        if not line.startswith(prefix+' '):
            continue
        parts = line.split()
        a,b = map(int,parts[1:3])
        step,rank = (a,b) if complete else (b,a)
        value = float(parts[-1])
        if (step,rank) in rows or not isfinite(value) or value<=0:
            raise ValueError(f'invalid timing at step {step}, rank {rank}')
        rows[step,rank] = value
    ranks = prod(config['mpi'])
    if rows.keys() != {(s,r) for s in range(steps) for r in range(ranks)}:
        raise ValueError('missing or unexpected rank-step timing')
    if complete and re.findall(r'^AIR5_CHEMISTRY half=([12]) ',log,re.M) != ['1','2']*steps:
        raise ValueError('incomplete AIR5 chemical half-step sequence')
    samples = []
    for step in range(config['warmup'],steps):
        rank = max(range(ranks),key=lambda r: rows[step,r])
        samples.append(dict(step=step+1,slowest_rank=rank,seconds=rows[step,rank]))
    return dict(samples=samples,median_seconds=median(s['seconds'] for s in samples))


def normalized_enstrophy(value, tiles):
    return value/tiles[0]**2


def summarize_campaign(campaign, runs):
    groups = {}
    missing = []
    for name,config in campaign.items():
        repetitions = []
        for r in range(1,config['repetitions']+1):
            path = runs/name/f'r{r}'/'run.log'
            if not path.is_file():
                missing.append(str(path))
                continue
            repetitions.append(reduce_run(path.read_text(),config))
        if len(repetitions) == config['repetitions']:
            seconds = median(r['median_seconds'] for r in repetitions)
            groups[name] = dict(seconds_per_step=seconds,repetitions=repetitions,
                                intervals_per_second=prod(config['grid'])/seconds)
    for name,data in groups.items():
        baseline = groups.get(f'{campaign[name]["kind"]}_001nodes')
        if baseline is not None:
            data['weak_efficiency'] = baseline['seconds_per_step']/data['seconds_per_step']
    return dict(cases=groups,missing_logs=missing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare').add_argument('output',type=Path)
    campaign_action = sub.add_parser('summarize')
    campaign_action.add_argument('campaign',type=Path)
    campaign_action.add_argument('runs',type=Path)
    for name,target in [('check-placement','placement'),('reduce','log')]:
        action = sub.add_parser(name)
        action.add_argument('campaign',type=Path)
        action.add_argument('case')
        action.add_argument(target,type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output)
        return
    campaign = json.loads(args.campaign.read_text())
    if args.action == 'summarize':
        print(json.dumps(summarize_campaign(campaign,args.runs),indent=2))
        return
    config = campaign[args.case]
    if args.action == 'check-placement':
        placement_check(config,json.loads(args.placement.read_text()))
        print('Observed node blocks match ASTR coordinates')
    else:
        print(json.dumps(reduce_run(args.log.read_text(),config),indent=2))


if __name__ == '__main__':
    main()
