#!/usr/bin/env python3
"""Approved same-topology M12 ten-step versus five-plus-five restart gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

import h5py
import numpy as np
from PIL import Image

from compare_flowstate import read_flowstate
from run_m12_local_compare import TOPOLOGIES,run,check_pair
from run_m12_insitu_acceptance import configuration,PRODUCTS
from run_output_restart_validation import compare_fields,archive_schedule_payload

FINAL='outdat/new/checkpoints/step000000000010'


def checkpoint_transfers(case,ranks):
    result=[]
    with h5py.File(case/FINAL/'state.h5') as state:
        partitions=state['partitions'][...].reshape(-1,8)
    if len(partitions)!=ranks:
        raise ValueError('checkpoint transfer partition count differs')
    for rank,partition in enumerate(partitions):
        scalar=int(np.prod(partition[3:6]+1+2*partition[6]))*8
        expected=Counter({scalar:3,3*scalar:1,5*scalar:1})
        with sqlite3.connect((case/f'trace.rank{rank}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
            rows=db.execute("select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e "
                "on e.id=m.copyKind where e.name='CUDA_MEMCPY_KIND_DTOH' and m.bytes>=? "
                "and not exists(select 1 from NVTX_EVENTS n where m.start>=n.start and m.end<=n.end "
                "and (n.text like 'ASTR_IS8_%' or n.text like 'ASTR_X4_%'))",(scalar,)).fetchall()
        if Counter(n for n, in rows)!=expected:
            raise ValueError('large checkpoint transfer shape/count attribution differs')
        result.append({'rank':rank,'checkpoint_d2h_bytes':11*scalar,'copies':dict(expected),
            'scope':'bounded capture: five-component q, three-component velocity and rho/p/T scalar packs outside postprocessing'})
    return result


def exact_checkpoint(reference,candidate):
    fields=compare_fields(reference/FINAL/'state.h5',candidate/FINAL/'state.h5')
    for name in ('control.bin','insitu_control.bin'):
        if (reference/FINAL/name).read_bytes()!=(candidate/FINAL/name).read_bytes():
            raise ValueError('exact restart control differs: '+name)
    if archive_schedule_payload(reference/FINAL/'archives.bin')!=archive_schedule_payload(candidate/FINAL/'archives.bin'):
        raise ValueError('archive schedule differs')
    compare_fields(reference/'outdat/new/resources/geometry.h5',candidate/'outdat/new/resources/geometry.h5')
    for name in ('grid.2d','flowini2d.h5','inlet.prof','wallbs.dat'):
        if (reference/'outdat/new/resources'/name).read_bytes()!=(candidate/'outdat/new/resources'/name).read_bytes():
            raise ValueError('frozen restart resource differs: '+name)
        if (candidate/'datin'/name).exists():
            raise ValueError('restart still depends on original resource: '+name)
    # Primitive caches, all halo extras and q are already compared bit for bit.
    with h5py.File(reference/FINAL/'state.h5') as state:
        names=[name for name in state if name.startswith('q')]
        if not names or not all(np.isfinite(state[name][...]).all() for name in names):
            raise ValueError('nonfinite completed-step checkpoint')
        if np.any(state['q0006'][...]<=0) or np.any(state['q0011'][...]<=0):
            raise ValueError('nonpositive physical-node primitive cache')
        extras=state['rank_extras'][...]
        offset=0
        for partition in state['partitions'][...].reshape(-1,8):
            count=int(partition[-1]);chunk=extras[offset:offset+count].reshape(11,-1)
            offset+=count
            # Exterior nonperiodic padding has q=0; unused GPU primitives may be 0/0.
            # Keep those exact bytes, but apply finite-state gates to populated halos.
            if not np.isfinite(chunk[:6]).all() or np.any(chunk[0]<0) or \
                    not np.isfinite(chunk[6:,chunk[0]>0]).all():
                raise ValueError('nonfinite populated halo in completed-step checkpoint')
        if offset!=len(extras):
            raise ValueError('checkpoint halo inventory differs')
    return fields


def exact_continuation(reference,first,resumed,ranks):
    for path in sorted((resumed/'diagnostics').glob('state.*.bin')):
        if '.initialized.' in path.name:
            continue  # Initialization precedes native restore; it is not resumed advancement.
        if path.read_bytes()!=(reference/'diagnostics'/path.name).read_bytes():
            raise ValueError('continued stage/diagnostic changed: '+path.name)
    for path in sorted((first/'diagnostics').glob('state.*.bin')):
        if path.read_bytes()!=(reference/'diagnostics'/path.name).read_bytes():
            raise ValueError('checkpoint observer changed stage: '+path.name)
    headers=[];values=[]
    for case in (reference,first,resumed):
        header,value=read_flowstate(case)
        headers.append(header);values.append(value)
    if headers[0]!=headers[1] or headers[0]!=headers[2] or \
            values[0].tobytes()!=np.concatenate(values[1:]).tobytes():
        raise ValueError('first-stage statistics sequence differs')
    return {'stage_and_diagnostic_bytes':'exact','listing_statistics':'exact', 'ranks':ranks}


def exact_render(reference,first,resumed,ranks):
    for case,steps in ((reference,(5,10)),(first,(5,)),(resumed,(10,))):
        expected={f'{p}.step{s:08d}.jpeg' for s in steps for p in PRODUCTS}
        directory=case/'outdat/render'
        if {p.name for p in directory.glob('*.jpeg')}!=expected or list(directory.glob('*.vtp')):
            raise ValueError('duplicate/missing frame or host geometry')
        for step in steps:
            for rank in range(ranks):
                name=f'mesh_step{step:08d}_rank{rank}.json'
                actual=json.loads((directory/name).read_text())
                expected_record=json.loads((reference/'outdat/render'/name).read_text())
                if actual!=expected_record or actual['geometry_host_bytes']!=0:
                    raise ValueError('final product identity/fields differ: '+name)
            for product in PRODUCTS:
                name=f'{product}.step{step:08d}.jpeg'
                if not (directory/name).with_suffix('.eps').is_file():
                    raise ValueError('missing EPS counterpart')
                with Image.open(directory/name) as a, Image.open(reference/'outdat/render'/name) as b:
                    aa,bb=np.asarray(a),np.asarray(b)
                    if aa.tobytes()!=bb.tobytes() or aa.shape!=(384,1536,3) or not np.any(aa<220):
                        raise ValueError('decoded image continuation differs: '+name)
            for rank in range(ranks):
                for prefix in ('plane','surface','trace.instantaneous_streamlines'):
                    name=f'{prefix}.step{step}.rank{rank}.bin'
                    if (case/'diagnostics'/name).read_bytes()!=(reference/'diagnostics'/name).read_bytes():
                        raise ValueError('final geometry/physical product fields differ: '+name)
    return {'steps':[5,10], 'decoded_pixels':'exact','products':list(PRODUCTS),'duplicate_frames':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--topologies',nargs='+',choices=TOPOLOGIES,default=TOPOLOGIES)
    parser.add_argument('--backends',nargs='+',choices=('cpu','gpu'),default=('cpu','gpu'))
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--sanitizer',type=Path)
    parser.add_argument('--trace-only',action='store_true',help='Separate clean resumed-frame transfer gate')
    parser.add_argument('--nsys',type=Path)
    args=parser.parse_args()
    if args.memcheck and not args.sanitizer:
        parser.error('--memcheck requires --sanitizer')
    if args.trace_only and (not args.nsys or not args.library or args.memcheck or args.backends!=['gpu']):
        parser.error('--trace-only requires GPU, --library and --nsys without memcheck')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.executable=args.executable.resolve(strict=True)
    args.diagnostics=False
    report={'status':'running','checks':[]}
    try:
        if args.trace_only:
            from run_m12_device_observation import check_transfers
            config=configuration(args.library.resolve(strict=True))
            for topology in args.topologies:
                ranks=int(np.prod([int(v) for v in topology.split(',')]))
                plain=run(args,'gpu',topology,case_label='plain',checkpoint_interval=1000000000)
                baseline=json.loads((plain/'resources.sampled.json').read_text())
                first=run(args,'gpu',topology,case_label='seed',insitu_config=config,steps=5,
                    checkpoint_interval=5,observation='images',resource_baseline=baseline)
                resumed=run(args,'gpu',topology,case_label='trace',insitu_config=config,
                    checkpoint_interval=5,observation='trace',resource_baseline=baseline,
                    restore=first/'outdat/new/checkpoints/step000000000005')
                compare_fields(plain/FINAL/'state.h5',resumed/FINAL/'state.h5')
                def rng_tail(case):
                    return (case/FINAL/'control.bin').read_bytes().split(b'ASTRWR01',1)[1]
                if rng_tail(plain)!=rng_tail(resumed):
                    raise ValueError('clean render/restart capture changed RNG state')
                report['checks'].append({'topology':topology,'transfers':check_transfers(
                    resumed,ranks,steps=(10,),checkpoint=True),'checkpoint_transfers':checkpoint_transfers(resumed,ranks),
                    'checkpoint_transfer_scope':
                    'Full q and primitive-cache checkpoint transfers are outside product/render ranges; not zero total D2H'})
            report['status']='passed'
            print('PASS clean resumed-frame transfer attribution',flush=True)
            return
        for topology in args.topologies:
            ranks=int(np.prod([int(v) for v in topology.split(',')]))
            continuous={}
            for backend in args.backends:
                config=configuration(args.library.resolve(strict=True)) if backend=='gpu' and args.library else None
                prefix=backend
                plain=None;baseline=None
                if config:
                    plain=run(args,backend,topology,case_label='gpu_plain',checkpoint_interval=1000000000)
                    baseline=json.loads((plain/'resources.sampled.json').read_text())
                reference=run(args,backend,topology,case_label=prefix+'_continuous',
                    insitu_config=config,checkpoint_interval=5,product_oracles=bool(config),resource_baseline=baseline)
                first=run(args,backend,topology,case_label=prefix+'_first',
                    insitu_config=config,checkpoint_interval=5,steps=5,product_oracles=bool(config),resource_baseline=baseline)
                source=first/'outdat/new/checkpoints/step000000000005'
                resumed=run(args,backend,topology,case_label=prefix+'_resumed',
                    insitu_config=config,checkpoint_interval=5,restore=source,product_oracles=bool(config),resource_baseline=baseline)
                check={'backend':backend,'topology':topology,'datasets':exact_checkpoint(reference,resumed),
                    'continuation':exact_continuation(reference,first,resumed,ranks)}
                if config:
                    check['render']=exact_render(reference,first,resumed,ranks)
                    compare_fields(reference/FINAL/'state.h5',plain/FINAL/'state.h5')
                    for path in sorted((plain/'diagnostics').glob('state.*.bin')):
                        if path.read_bytes()!=(reference/'diagnostics'/path.name).read_bytes():
                            raise ValueError('checkpoint/render observer changed stage: '+path.name)
                    check['render_observer']='authoritative state and halo caches bitwise unchanged'
                    check['resource_baseline']=str(plain/'resources.sampled.json')
                continuous[backend]=reference
                report['checks'].append(check)
                print('PASS M12 exact restart '+backend+' topology='+topology,flush=True)
            if set(continuous)=={'cpu','gpu'}:
                report['checks'][-1]['cpu_gpu']=check_pair(continuous['cpu'],continuous['gpu'],ranks)
            (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
