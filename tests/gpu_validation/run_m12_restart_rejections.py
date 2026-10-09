#!/usr/bin/env python3
"""Reject damaged/mismatched M12 restart state before the first resumed RK."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from prepare_m12_local_case import prepare
from test_output_series_repair import seal_file_records,repair


def altered_source(source,root,defect):
    shutil.copytree(source.parent.parent/'resources',root/'resources')
    batch=root/'checkpoints'/source.name
    shutil.copytree(source,batch)
    path=batch/'control.bin'
    raw=bytearray(path.read_bytes())
    offset=raw.index(b'ASTRWR01')
    if defect=='resource':
        (root/'resources/wallbs.dat').unlink()
        return batch
    if defect=='corrupt':
        raw[-1]^=1
        path.write_bytes(raw)
        return batch  # Deliberately do not repair the integrity receipt.
    if defect=='missing':
        del raw[offset:]
    elif defect=='compiler':
        raw[offset+24]^=1
    elif defect=='runtime':
        raw[offset+24+2*16384]^=1
    elif defect=='flag':
        start=offset+24+2*16384+64+6*4
        raw[start:start+4]=np.asarray([2],dtype='<i4').tobytes()
    elif defect=='mode_version':
        raw[:8]=b'ASTROC04'
    else:
        raise ValueError('unknown checkpoint defect')
    path.write_bytes(raw)
    names=[row.split()[0] for row in (batch/'MANIFEST').read_text().splitlines()[2:]]
    (batch/'MANIFEST').write_text(seal_file_records(batch,names,'ASTR_CHECKPOINT_BUNDLE 1'))
    size,crc=repair.fingerprint(batch/'MANIFEST')
    (batch/'COMPLETE').write_text(f'ASTR_COMPLETE_1 {size} {crc:016X}\n')
    return batch


def rejected(args,label,source,message,topology='1,1,1',mode='legacy_random'):
    case=args.output/label
    prepare(args.source,case,'cpu')
    config=(case/'datin/input.output').read_text()
    config=config.replace("directory='outdat/new',",f"directory='outdat/new',restore_directory='{source}',")
    (case/'datin/input.output').write_text(config)
    for name in ('grid.2d','flowini2d.h5','inlet.prof','wallbs.dat'):
        (case/'datin'/name).unlink()
    if mode=='current':
        # Supply a valid alternate driver so its missing phase file cannot mask
        # the checkpoint's deliberate random-mode mismatch rejection.
        (case/'datin/wallbs_phase.dat').write_text(' '.join(['0.5']*15)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(ASTR_FORCE_MPI_TOPOLOGY=topology,ASTR_WALL_BLOWING_MODE=mode,
               ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_PRECISION_MODE='fp64')
    ranks=int(np.prod([int(v) for v in topology.split(',')]))
    command=[str(args.mpiexec),'--mca','pml','ob1','--mca','btl','self,vader,tcp',
        '--mca','coll_hcoll_enable','0','--oversubscribe','-np',str(ranks),
        str(args.executable),'run','datin/input.dat']
    with (case/'run.log').open('wb') as log:
        result=subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    text=(case/'run.log').read_text()
    if not result.returncode or message not in text or 'ASTR_CFL complete_step=' in text or 'The job is done!' in text:
        raise ValueError('missing pre-advancement rejection: '+label)
    size=sum(p.stat().st_size for p in case.rglob('*') if p.is_file())
    if size>4*1024**3:
        raise ValueError('approved directory budget exceeded')
    return {'defect':label,'returncode':result.returncode,'message':message,'before_resumed_rk':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reference','output','executable','mpiexec'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.executable=args.executable.resolve(strict=True)
    source=args.reference/'cpu_first_np1_1x1x1/outdat/new/checkpoints/step000000000005'
    source=source.resolve(strict=True)
    cases={'missing':'missing random wall RNG metadata',
        'compiler':'random wall compiler/runtime mismatch','runtime':'random wall compiler/runtime mismatch',
        'flag':'invalid random wall RNG rank state','corrupt':'invalid new checkpoint bundle',
        'resource':'invalid new checkpoint bundle','mode_version':'random wall checkpoint RNG mode/version mismatch'}
    checks=[]
    for defect,message in cases.items():
        modified=altered_source(source,args.output/('source_'+defect),defect)
        checks.append(rejected(args,defect,modified,message))
    checks.append(rejected(args,'forcing_mode',source,'extruded profile restart requires legacy_random checkpoint',mode='current'))
    checks.append(rejected(args,'rank_count',source,'random wall RNG format/rank count mismatch',topology='2,1,1'))
    x=args.reference/'cpu_first_np2_2x1x1/outdat/new/checkpoints/step000000000005'
    checks.append(rejected(args,'topology',x.resolve(strict=True),'random wall topology mismatch',topology='1,2,1'))
    (args.output/'summary.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2)+'\n')
    print(f'PASS {len(checks)} M12 pre-advancement restart rejections',flush=True)


if __name__=='__main__':
    main()
