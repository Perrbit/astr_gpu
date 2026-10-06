"""Test-only faults around the real resident Catalyst entry."""
import errno
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
base=runpy.run_path(str(ROOT/'scripts/insitu/device_render_pipeline.py'))
options=base['options']
fault=base['output'].resolve().parent.parent.name.partition('_fault_')[2]


def catalyst_initialize():
    if fault=='initialize':
        raise RuntimeError('ASTR_RESIDENT_INITIALIZATION_FAILURE_INJECTED')


original_capture=base['capture_image']
original_publish=base['publish_pair']


def capture(view):
    if fault=='draw':
        raise RuntimeError('ASTR_RESIDENT_DRAW_FAILURE_INJECTED')
    return original_capture(view)


def publish(path,jpeg,eps):
    if fault=='image' and 'step00000002' in path.name:
        from image_publication import write_payload
        module=original_publish.__globals__

        def write(stream,payload):
            if '.eps.partial' in stream.name:
                stream.write(payload[:2])
                raise OSError(errno.EIO,'ASTR_RESIDENT_IMAGE_FAILURE_INJECTED')
            return write_payload(stream,payload)

        module['write_payload']=write
        try:
            return original_publish(path,jpeg,eps)
        finally:
            module['write_payload']=write_payload
    return original_publish(path,jpeg,eps)


base['render_product'].__globals__['capture_image']=capture
base['render_product'].__globals__['publish_pair']=publish
catalyst_execute=base['catalyst_execute']
catalyst_finalize=base['catalyst_finalize']
