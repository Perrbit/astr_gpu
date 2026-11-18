"""Same-complete-step authority checks for the existing small plate archives."""
import json

import h5py
import numpy as np

from run_cpp_multi_tgv_gate import layouts
from run_cpp_tgv_gate import compare, logged_steps, namelist, snapshot

FIELDS = ('density', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'temperature')
PLANES = (('i', 8), ('j', 0), ('k', 16))


def configure(case):
    (case/'outdat/new').mkdir()
    (case/'datin/input.output').write_text(namelist(dict(output=dict(directory='outdat/new', exact_restart=True,
        host_budget_bytes=-1, device_budget_bytes=-1, device_reserve_bytes=0),
        checkpoint=dict(enabled=False), volume=dict(enabled=True, mode='steps', interval_steps=2,
        initial_frame=False, final_frame=False), slices=dict(enabled=False))))
    definitions = [dict(id=f'plane-{axis}{index}', kind='slice', enable_request=1, source='instantaneous',
        fields=list(FIELDS), sampling=dict(mode='grid_index', axis=axis, index=index,
            index_ranges={other:[0,16] for other in 'ijk' if other!=axis}),
        output=dict(channel='data', format='hdf5-xdmf'), schedule=dict(mode='steps', interval=2))
        for axis,index in PLANES]
    (case/'input.products.jsonl').write_text(''.join(json.dumps(item)+'\n' for item in
        [dict(catalog_schema='astr-insitu-products-v1'), *definitions]))


def check(case):
    with h5py.File(case/'datin/grid.flatplate.h5') as source:
        coordinates = np.stack([source[name][...] for name in 'xyz'], axis=-1)
    status = json.loads((case/'outdat/new/products.status.json').read_text())
    if len(status['products'])!=3 or any(p['state']!='active' or p['last_output_step']!=4 for p in status['products']):
        raise AssertionError('plate numerical slice product status/phase')
    partitions = layouts(case);clocks = logged_steps(case);rows = {}
    for step in (2,4):
        q = np.empty((17,17,17,5))
        for rank,layout in sorted(partitions.items()):
            field = snapshot(case/f'outdat/rhs.post_update.step{step-1:08d}.rk03.rank{rank:08d}.bin')
            q[tuple(slice(o,o+n+1) for o,n in zip(layout['origin'],layout['upper']))] = field
        rho = q[...,0];velocity = q[...,1:4]/rho[...,None]
        # The named fixture is nondimensional Mach2/gamma1.4; no flow advancement here.
        pressure = (q[...,4]-.5*rho*np.sum(velocity**2,axis=-1))/(1./(1.4-1.))
        physical = np.concatenate((rho[...,None],velocity,pressure[...,None],(5.6*pressure/rho)[...,None]),axis=-1)
        expected = physical.transpose(2,1,0,3);time = clocks[step-1,2]
        base = case/'outdat/new/fields';frame = base/'segment00000000'/f'step{step:012d}'
        if not (frame/'COMPLETE').is_file():raise AssertionError('missing completed plate volume')
        errors = []
        with h5py.File(frame/'data.h5') as data,h5py.File(base/'resources/data.h5') as geometry:
            identity = data['metadata'][...]
            if identity[1]!=step or identity[2:5].view(np.float64)[0]!=time:raise AssertionError('plate volume phase')
            if not np.array_equal(geometry['coordinates'][...],coordinates):raise AssertionError('plate archive geometry')
            for c,name in enumerate(FIELDS):
                if data[name].dtype!=np.dtype('float64'):raise AssertionError('plate archive field precision')
                errors.append(max(compare(data[name][...][...,None],expected[...,c,None])))
            vector = np.stack([data['velocity_'+axis][...] for axis in 'xyz'],axis=-1)
            if vector.tobytes()!=data['velocity'][...].tobytes():raise AssertionError('plate vector/component bits')
        rows[f'volume/{step}'] = errors
        for axis,index in PLANES:
            segments = list((case/'outdat/new/slice_products').glob(f'plane-{axis}{index}/*/segment*'))
            if len(segments)!=1:raise AssertionError('plate numerical slice segment')
            segment = segments[0]
            if sorted(p.name for p in segment.glob('step*.h5'))!=[f'step{n:012d}.h5' for n in (0,2,4)]:
                raise AssertionError('plate numerical slice schedule')
            selected = [slice(None)]*3;selected[2-'ijk'.index(axis)] = index;selected = tuple(selected);errors=[]
            with h5py.File(segment/f'step{step:012d}.h5') as data,h5py.File(segment/'geometry.h5') as geometry:
                if data['step'][()]!=step or data['clock'][0]!=time:raise AssertionError('plate numerical slice phase')
                if not np.array_equal(geometry['coordinates'][...],coordinates[selected]):raise AssertionError('plate slice index/geometry')
                for c,name in enumerate(FIELDS):
                    if data[name].dtype!=np.dtype('float64'):raise AssertionError('plate slice precision')
                    errors.append(max(compare(data[name][...][...,None],expected[selected][...,c,None])))
                if 'velocity' in data or 'coordinates' in data:raise AssertionError('plate slice repeated vector/geometry')
            rows[f'{axis}{index}/{step}'] = errors
    return dict(complete_steps=rows, initial_slices='inventory only, no q0 numerical comparison')
