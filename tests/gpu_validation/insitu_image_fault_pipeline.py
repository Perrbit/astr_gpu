"""Test-only publication faults after collective capture; no solver changes."""
import errno
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
import image_publication

base=runpy.run_path(str(ROOT/'scripts/insitu/tgv_pipeline.py'))
options=base['options']
rank=base['rank']
fault=base['output'].resolve().parent.parent.name.partition('_fault_')[2]
original=image_publication.write_payload


def write_payload(stream,payload):
    if rank==0 and fault and 'q_surface.step00000002.eps.partial' in str(stream.name):
        if fault=='unknown':
            raise RuntimeError('injected unknown image failure')
        code={'eacces':errno.EACCES,'enospc':errno.ENOSPC,'eio':errno.EIO,
              'journal':errno.EIO,'geometry':errno.EIO}[fault]
        stream.write(payload[:23])
        raise OSError(code,'injected image publication fault')
    original(stream,payload)


image_publication.write_payload=write_payload
if fault=='journal':
    original_open=Path.open

    def open_path(path,*args,**kwargs):
        if rank==1 and path.name.startswith('missing.q_surface.step00000002.'):
            raise OSError(errno.ENOSPC,'injected missing-image journal failure')
        return original_open(path,*args,**kwargs)

    Path.open=open_path
elif fault=='geometry':
    original_create=base['pv'].CreateWriter

    def create_writer(filename,*args,**kwargs):
        # Fail on every rank before entering a geometry collective.
        if 'step00000002.pvtp' in filename:
            raise OSError(errno.EIO,'injected geometry failure')
        return original_create(filename,*args,**kwargs)

    base['pv'].CreateWriter=create_writer

catalyst_execute=base['catalyst_execute']
catalyst_finalize=base['catalyst_finalize']
