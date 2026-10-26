"""Faults at matching optional-product boundaries, never in solver state."""
from pathlib import Path
import runpy
import sys
import os

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
from paraview import catalyst

resident=len(catalyst.get_args())==4 and catalyst.get_args()[3] in ('standard-device','direct-device')
base=runpy.run_path(str(ROOT/'scripts/insitu'/('device_render_pipeline.py' if resident else 'tgv_pipeline.py')))
options=base['options']
namespace=base['catalyst_execute'].__globals__
fault=base['output'].resolve().parent.parent.name.partition('_fault_')[2]
rank=base['rank']
active=False
if fault=='extraction':
    os.environ['ASTR_INSITU_TEST_PRODUCT_FAIL']='q_surface:4:1'
original_capture=namespace['capture_image']


def capture(*args,**kwargs):
    result=original_capture(*args,**kwargs)
    item=namespace['render_objects'].get('q_surface')
    view=item.get('view') if isinstance(item,dict) else (item[0] if item else None)
    if active and fault=='capture' and rank==1 and args[0] is view:
        raise RuntimeError('injected rank-local screenshot failure')
    return result


namespace['capture_image']=capture
original_payload=base['publish_pair'].__globals__['write_payload']


def payload(stream,data):
    if active and fault in ('image','diagnostics') and 'q_surface.' in stream.name:
        stream.write(data[:2])
        raise OSError(5,'injected image write failure')
    if active and fault=='velocity' and 'velocity_slice.step00000002.' in stream.name:
        stream.write(data[:2])
        raise OSError(5,'injected main slice image write failure')
    return original_payload(stream,data)


base['publish_pair'].__globals__['write_payload']=payload
original_writer=base['pv'].CreateWriter


def writer(path,*args,**kwargs):
    if active and fault=='geometry' and 'q_surface.' in path:
        raise OSError(5,'injected geometry writer failure')
    return original_writer(path,*args,**kwargs)


base['pv'].CreateWriter=writer
original_text=Path.write_text
original_open=Path.open


def write_text(path,*args,**kwargs):
    if fault=='diagnostics' and rank==1 and path.name.startswith(('mesh_step','lifecycle_rank')):
        raise OSError(5,'injected optional diagnostic write failure')
    return original_text(path,*args,**kwargs)


def open_path(path,*args,**kwargs):
    if fault=='diagnostics' and rank==1 and path.name.startswith('missing.'):
        raise OSError(5,'injected missing-image journal write failure')
    return original_open(path,*args,**kwargs)


Path.write_text=write_text
Path.open=open_path


def catalyst_execute(info):
    global active
    active=int(info.timestep)==2
    if active and fault=='diagnostics' and rank==1:
        report=base['output']/f'resources.rank{rank:08d}.csv'
        report.rename(report.with_suffix('.before_failure.csv'))
        report.mkdir()
    namespace['catalyst_execute'](info)


catalyst_finalize=base['catalyst_finalize']
