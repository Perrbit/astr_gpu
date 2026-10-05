"""Run with pvbatch: repeat a frozen TGV field through the real render script."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace

from paraview import catalyst, simple as pv, servermanager


def proxy_counts():
    manager = servermanager.ProxyManager()
    return {group: len(manager.GetProxiesInGroup(group)) for group in
            ('sources', 'views', 'representations', 'lookup_tables', 'scalar_bars')}


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Cannot read probe RSS')


def device_memory(expected):
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
    for line in result.stdout.splitlines():
        uuid, used, free = [item.strip() for item in line.split(',')]
        if uuid == 'GPU-'+expected:
            return int(used)*1024**2, int(free)*1024**2
    raise RuntimeError('Cannot observe the selected render GPU')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--grid', type=int, default=32)
    parser.add_argument('--frames', type=int, default=4)
    parser.add_argument('--full-fields', action='store_true')
    parser.add_argument('--structured-grid', action='store_true')
    parser.add_argument('--require-stable-proxies', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts/insitu'))
    sys.path.insert(0, str(args.pipeline.resolve().parent))
    from egl_identity import EGL
    devices = [item for item in EGL().devices() if item['uuid'] is not None]
    if not devices:
        raise RuntimeError('No CUDA-capable EGL device')
    import os
    ordinal = int(os.environ['VTK_EGL_DEVICE_INDEX'])
    expected = next(item['uuid'] for item in devices if item['egl_index'] == ordinal)
    baseline_rss = rss_bytes()
    baseline_device, _ = device_memory(expected)
    catalyst.get_script_filename = lambda: str(args.pipeline.resolve())
    catalyst.get_args = lambda: [str(args.output.resolve()), expected]
    spec = importlib.util.spec_from_file_location('render_pipeline', args.pipeline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = pv.ProgrammableSource()
    fixture.OutputDataSetType = 'vtkStructuredGrid' if args.structured_grid else 'vtkImageData'
    fixture.Script = f'''
import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkPoints
n = {args.grid}
data = self.GetOutputDataObject(0)
data.SetDimensions(n+1, n+1, n+1)
x,y,z = np.meshgrid(*([np.linspace(0,2*np.pi,n+1)]*3), indexing='ij')
if {args.structured_grid!r}:
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.column_stack([a.ravel(order='F') for a in (x,y,z)]), deep=True))
    data.SetPoints(points)
else:
    data.SetSpacing(2*np.pi/n, 2*np.pi/n, 2*np.pi/n)
fields = {{'u': np.sin(x)*np.cos(y)*np.cos(z),
          'v': -np.cos(x)*np.sin(y)*np.cos(z), 'w': np.zeros_like(x),
          'Q_rs': (np.sin(x)**2*np.sin(y)**2-np.cos(x)**2*np.cos(y)**2)*np.cos(z)**2}}
if {args.full_fields!r}:
    names = ['q1','q2','q3','q4','q5','rho','pressure','temperature',
             'du_dx','dv_dx','dw_dx','du_dy','dv_dy','dw_dy',
             'du_dz','dv_dz','dw_dz','divergence','omega_x','omega_y','omega_z']
    for name in names:
        fields[name] = np.ones_like(x)
for name,values in fields.items():
    item = numpy_to_vtk(values.ravel(order='F'), deep=True)
    item.SetName(name)
    data.GetPointData().AddArray(item)
'''
    fixture.UpdatePipeline()
    original = module.source
    module.source = fixture
    rows = []
    report = {'status': 'running', 'grid': args.grid, 'frozen_field': True,
              'mesh_type': 'vtkStructuredGrid' if args.structured_grid else 'vtkImageData',
              'field_count': 25 if args.full_fields else 4, 'frames': rows}
    try:
        for step in range(1, args.frames+1):
            # Keep values frozen, but replace the input arrays as Catalyst does per frame.
            fixture.GetClientSideObject().Modified()
            module.catalyst_execute(SimpleNamespace(timestep=step, time=step*1e-4))
            row = {'step': step, 'proxies': proxy_counts(), 'rss_bytes': rss_bytes()}
            used, free = device_memory(expected)
            row.update(device_increment_bytes=used-baseline_device, device_free_bytes=free)
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row['rss_bytes']-baseline_rss > 16*1024**3 or used-baseline_device > 6*1024**3 or free < 2*1024**3:
                raise RuntimeError('Frozen probe exceeds the approved demonstration budget')
        if args.require_stable_proxies and [row for row in rows if row['proxies'] != rows[0]['proxies']]:
            raise AssertionError('Proxy registrations grow for the same frozen field')
        report['status'] = 'passed-frozen-proxy-lifecycle'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        module.catalyst_finalize()
        pv.Delete(fixture)
        pv.Delete(original)
        report['after_finalize'] = {'proxies': proxy_counts(), 'rss_bytes': rss_bytes()}
        (args.output/'probe.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
