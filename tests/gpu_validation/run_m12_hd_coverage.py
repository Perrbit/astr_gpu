#!/usr/bin/env python3
"""Local M12 HD first-output coverage, not a transition or production test."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

import h5py
import numpy as np
from PIL import Image

from prepare_m12_local_case import prepare, REFINED_INTERVALS, FULL_INTERVALS, replace_value
from run_m12_boundary_compare import environment
from insitu_cfl_gate import CflGate
from insitu_resource_monitor import run_monitored

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
from image_publication import eps_output_enabled
LIBRARY = Path('/home/dell/workspace/astr_dependencies/build/paraview-6.1.1-astr-device-gcc13/lib/catalyst')
MPIEXEC = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')
GIB = 1024**3
W_COLOR_MAX = .01  # Local first-output preview, not a production color calibration.


def insitu_configuration(images=True):
    intervals = (250, 100, 250) if images else (10**9,)*3
    return f"""&insitu_run
 enabled=t, statistics=f, render=t, initial_frame=f, final_frame=f,
 products='m12_diagnostics',streamline_seeds='bl-layered256',
 processing_backend='device',derivative_backend='gpu',rendering_pipeline='standard-device',
 postprocess_transport='pinned',output_directory='outdat/render',
 host_budget_bytes={16*GIB},device_budget_bytes={6*GIB},device_reserve_bytes={2*GIB},
 implementation_path='{LIBRARY}',pipeline_file='{ROOT}/scripts/insitu/device_render_pipeline.py',
 product_ids='q_surface.image','velocity_slice.image','instantaneous_streamlines.image',
 product_modes='steps','steps','steps',product_steps={','.join(map(str, intervals))}
/
"""


def prepare_probe_map(case, shape):
    with h5py.File(case/'datin/grid.2d') as grid:
        axes = (grid['x'][0, :], grid['y'][:, 0], np.linspace(0., 90., shape[2]+1))
    records = []
    for x in (27., 54., 81., 135., 270., 450., 675., 900.):
        for y in (.5, 1., 2.):
            for z in (22.5, 45., 67.5):
                target = (x, y, z)
                ijk = [int(np.abs(axis-value).argmin()) for axis, value in zip(axes, target)]
                actual = [float(axis[i]) for axis, i in zip(axes, ijk)]
                if ijk[2] == shape[2]:
                    raise ValueError('Probe maps to a periodic duplicate endpoint')
                records.append(dict(id=len(records)+1, target=target, indices=ijk, actual=actual))
    if shape in (REFINED_INTERVALS, FULL_INTERVALS) and any(
            p['actual'][0] >= axes[0][-51] or p['actual'][1] >= axes[1][-21] for p in records):
        raise ValueError('A probe lies inside the configured sponge')
    (case/'datin/insitu_probes.dat').write_text('ASTR_M12_PROBES_V1 72\n'+''.join(
        ' '.join(map(str, p['indices']+list(p['target'])+p['actual']))+'\n' for p in records))
    (case/'probe_map.json').write_text(json.dumps(dict(
        method='nearest original physical node, no interpolation',
        sponge_start=dict(x=float(axes[0][-51]), y=float(axes[1][-21])), probes=records), indent=2)+'\n')
    return records


def prepare_case(source, case, shape, steps, images, checkpoint=False):
    prepare(source, case, 'gpu', shape)
    path = case/'datin/controller'
    lines = path.read_text().splitlines()
    replace_value(lines, 'maxstep,feqchkpt', f'{steps-1},1,999999,999999,1,999999')
    path.write_text('\n'.join(lines)+'\n')
    (case/'outdat/render').mkdir()
    (case/'datin/input.insitu').write_text(insitu_configuration(images))
    if checkpoint:
        from run_m12_boundary_compare import configuration
        # Bounded independent numerical gate only, never the coverage runs.
        # Retain step 200 plus the automatic final state, within keep=2.
        configuration(case, 'extrapolation', shape, steps, archive=False, checkpoint_interval=200)
    prepare_probe_map(case, shape)


def check_compact(case, steps):
    records = np.genfromtxt(case/'outdat/render/probes.csv', delimiter=',', names=True)
    if len(records) != 72*steps or not np.isfinite(np.column_stack([records[k] for k in records.dtype.names])).all():
        raise ValueError('Incomplete/nonfinite sparse monitor sequence')
    np.testing.assert_array_equal(records['step'], np.repeat(np.arange(1, steps+1), 72))
    np.testing.assert_array_equal(records['probe'], np.tile(np.arange(1, 73), steps))
    np.testing.assert_allclose(records['time'], records['step']*.02, rtol=0, atol=2e-13)
    if any(np.any(records[k] <= 0) for k in ('rho_rhoinf', 'p_pinf', 'T_Tinf')):
        raise ValueError('Nonpositive monitor density, pressure or temperature')
    walls = {}
    for step in range(50, steps+1, 50):
        pieces = [np.genfromtxt(case/'outdat/render'/f'wall.step{step:08d}.rank{r:04d}.csv',
                              delimiter=',', names=True) for r in range(2)]
        wall = np.concatenate(pieces)
        wall.sort(order='global_i')
        if not np.array_equal(wall['global_i'], np.arange(len(wall))) or \
                not np.isfinite(np.column_stack([wall[k] for k in wall.dtype.names])).all():
            raise ValueError('Wall ownership or finite gate failed')
        walls[step] = wall
    return records, walls


def expected_images(step, weak=False):
    if step % 100 == 0:
        result = {name: size for name, size in (
            [('velocity_slice', (4096, 1024)), ('temperature_slice', (4096, 1024)),
             ('spanwise_slice', (5120, 1536))]+[
            (f'{field}.x{w*100}_{(w+1)*100}', (5120, 1024))
            for field in ('velocity_slice', 'temperature_slice') for w in range(3)])}
        if weak:
            result['spanwise_slice_weak'] = (5120, 1536)
        return result
    return {'q_surface': (4096, 1024), 'q_surface.local': (3840, 2160),
            'instantaneous_streamlines': (3840, 2160)}


def visible_field_region(name, region):
    # A weak field on the fixed main range is legitimately almost white.
    threshold = 250 if name == 'spanwise_slice' else 220
    return np.count_nonzero(np.any(region < threshold, axis=-1)) >= 50


def check_images(case, steps):
    directory = case/'outdat/render'
    expected = set()
    receipts = []
    for step in sorted(set(range(100, steps+1, 100)) | set(range(250, steps+1, 250))):
        first = json.loads((directory/f'mesh_step{step:08d}_rank0.json').read_text())
        spanwise = first['products'].get('spanwise_slice', {})
        weak = spanwise.get('weak_display', {}).get('eligible', False)
        images = expected_images(step, weak)
        for rank in range(2):
            record = json.loads((directory/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            if set(record['products']) != set(images) or record['geometry_host_bytes'] != 0:
                raise ValueError('HD product coverage or host geometry gate failed')
            for name, size in images.items():
                item = record['products'][name]
                if item['image'] != 'published' or item['image_size'] != list(size) or item['dpi'] != 300 or \
                        item['jpeg_quality'] != 95 or item['eps_source'] != 'original screenshot RGB':
                    raise ValueError('HD image encoding/publication gate failed')
                if item['empty'] and not name.startswith('q_surface'):
                    raise ValueError('An expected nonempty HD product is empty: '+name)
                if name == 'spanwise_slice' and (item['color_range'] != [-W_COLOR_MAX, W_COLOR_MAX] or
                        item['outside_fixed_color_range']):
                    raise ValueError('HD preview spanwise-velocity color range is incorrect or saturated')
                if 'slice' in name and item.get('lighting_enabled', False):
                    raise ValueError('A two-dimensional slice has lighting enabled')
                if name == 'spanwise_slice' and 'weak_display' in item:
                    amplitude = max(map(abs, item['actual_range']))
                    if item['weak_display']['eligible'] != (0. < amplitude < .01*W_COLOR_MAX):
                        raise ValueError('Weak display does not match the physical range trigger')
                if name == 'spanwise_slice_weak':
                    main = record['products']['spanwise_slice']
                    half = main['weak_display']['half_range']
                    if item['color_field'] != 'w_Uinf_weak' or item['color_range'] != [-half, half] or \
                            not item['auxiliary'] or not item['shared_device_geometry'] or item['outside_fixed_color_range'] or \
                            any(item[key] != main[key] for key in ('local_points', 'local_cells', 'global_cells',
                                'local_geometry_bounds', 'physical_bounds', 'actual_range')):
                        raise ValueError('Weak auxiliary changed geometry, field identity or color range')
                if name == 'instantaneous_streamlines' and item['seeding'] != dict(
                        layout='bl-layered256', seeds=256, direction_trajectories=512):
                    raise ValueError('HD streamline seed identity differs')
                if name == 'instantaneous_streamlines' and (item['integration_step_limit'] != 4096 or
                        sum(item['termination_counts'].values()) != 512):
                    raise ValueError('Incomplete HD trajectory termination accounting')
            receipts.append(record)
        for name, size in images.items():
            path = directory/f'{name}.step{step:08d}.jpeg'
            expected.add(path.name)
            if eps_output_enabled() and not path.with_suffix('.eps').is_file():
                raise ValueError('Missing raw RGB EPS counterpart')
            with Image.open(path) as image:
                if image.size != size or image.info.get('dpi') != (300, 300):
                    raise ValueError('Pixel dimensions or DPI differ')
                # Nonwhite geometry is checked away from the lower annotations.
                pixels = np.asarray(image.convert('RGB'))
                region = pixels[int(.1*size[1]):int(.7*size[1]), int(.1*size[0]):int(.85*size[0])]
                if not name.startswith('q_surface') and not visible_field_region(name, region):
                    raise ValueError('Missing visible geometry: '+name)
    if {p.name for p in directory.glob('*.jpeg')} != expected or list(directory.glob('*.vtp')):
        raise ValueError('Unscheduled frames or host geometry output')
    return receipts


def check_same_state_compact(case, steps):
    """CPU FP64 evaluation on the GPU's complete-step checkpoint, short gate only."""
    path = case/'outdat/new/checkpoints'/f'step{steps:012d}/state.h5'
    with h5py.File(path) as state, h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        q = np.stack([state[f'q{c:04d}'][...] for c in range(1, 6)], axis=-1)
        x = geometry['q0001'][...]
        z = geometry['q0003'][...]
        dx = geometry['q0005'][...]
        dy = geometry['q0009'][...]
    rho = q[..., 0]
    velocity = q[..., 1:4]/rho[..., None]
    pressure = (q[..., 4]-.5*rho*np.sum(velocity**2, axis=-1))*.4
    temperature = pressure/rho*(1.4*12.**2)
    records, walls = check_compact(case, steps)
    observed = records[records['step'] == steps]
    mapping = json.loads((case/'probe_map.json').read_text())['probes']
    expected = np.array([np.r_[velocity[k,j,i], rho[k,j,i], pressure[k,j,i]*(1.4*12.**2), temperature[k,j,i]]
                         for p in mapping for i,j,k in [p['indices']]])
    actual = np.column_stack([observed[k] for k in ('u_Uinf','v_Uinf','w_Uinf','rho_rhoinf','p_pinf','T_Tinf')])
    probe_error = float(np.max(np.abs(expected-actual)))
    u, v = velocity[:, :, :, 0], velocity[:, :, :, 1]
    du = (-1.5*u[:,0,:]+2.*u[:,1,:]-.5*u[:,2,:])*dy[:,0,:]
    dt = (-1.5*temperature[:,0,:]+2.*temperature[:,1,:]-.5*temperature[:,2,:])*dy[:,0,:]
    dv = np.zeros_like(du)
    for i in range(du.shape[1]):
        row = v[:, 0, :]
        if i == 0:
            value = -1.5*row[:,i]+2.*row[:,i+1]-.5*row[:,i+2]
        elif i == row.shape[1]-1:
            value = 1.5*row[:,i]-2.*row[:,i-1]+.5*row[:,i-2]
        elif i in (1, row.shape[1]-2):
            value = .5*(row[:,i+1]-row[:,i-1])
        elif i in (2, row.shape[1]-3):
            value = (2./3.)*(row[:,i+1]-row[:,i-1])-(1./12.)*(row[:,i+2]-row[:,i-2])
        else:
            value = .75*(row[:,i+1]-row[:,i-1])-.15*(row[:,i+2]-row[:,i-2])+\
                    (1./60.)*(row[:,i+3]-row[:,i-3])
        dv[:,i] = value*dx[:,0,i]
    tw = temperature[:,0,:]
    mu = tw*np.sqrt(tw)*(1.+110.3/594.3)/(tw+110.3/594.3)/1.e4
    intervals = np.diff(z[:,0,:], axis=0)
    weights = .5*(intervals+np.roll(intervals,1,axis=0))
    cf = np.sum(weights*(2.*mu*(du+dv))[:-1],axis=0)/np.sum(weights,axis=0)
    heat = np.sum(weights*(mu/(.72*(.4*12.**2))*dt)[:-1],axis=0)/np.sum(weights,axis=0)
    wall = walls[steps]
    wall_error = float(max(np.max(np.abs(cf-wall['Cf'])),
                           np.max(np.abs(heat-wall['q_into_wall_rhoinf_Uinf3']))))
    if not np.isfinite([probe_error,wall_error]).all() or max(probe_error,wall_error)>2e-10:
        raise ValueError(f'CPU/GPU same-state compact gate failed: {probe_error}, {wall_error}')
    return dict(probe_max_abs=probe_error, wall_max_abs=wall_error,
                reference='CPU FP64 evaluation of the identical completed-step GPU state; no independent CFD evolution')


def check_same_state_slices(case, steps):
    """Check GPU plane reductions against independent physical-plane interpolation."""
    errors = {}
    with h5py.File(case/'outdat/new/resources/geometry.h5') as geometry:
        x = geometry['q0001'][0, 0, :]
        y = geometry['q0002'][0, :, 0]
        z = geometry['q0003'][:, 0, 0]
    sample_steps = [step for step in range(100, steps+1, 100)
                    if (case/'outdat/new/checkpoints'/f'step{step:012d}/state.h5').is_file()]
    if not sample_steps:
        raise ValueError('Missing same-state checkpoint for plane validation')
    for step in sample_steps:
        with h5py.File(case/'outdat/new/checkpoints'/f'step{step:012d}/state.h5') as state:
            q = np.stack([state[f'q{c:04d}'][...] for c in range(1, 6)], axis=-1)
        rho = q[..., 0]
        velocity = q[..., 1:4]/rho[..., None]
        temperature = (q[..., 4]-.5*rho*np.sum(velocity**2, axis=-1))*.4/rho*(1.4*12.**2)
        k = int(np.searchsorted(z, 45., side='left'))-1
        fz = (45.-z[k])/(z[k+1]-z[k])
        j = int(np.searchsorted(y, 1., side='left'))-1
        fy = (1.-y[j])/(y[j+1]-y[j])
        fields = dict(velocity_slice=(1.-fz)*velocity[k, :, :, 0]+fz*velocity[k+1, :, :, 0],
                      temperature_slice=(1.-fz)*temperature[k]+fz*temperature[k+1])
        # A plane through the interior of a tetrahedral layer also intersects
        # face/body diagonals, not just the vertical structured-grid edges.
        tetrahedra = ((0,1,3,7),(0,3,2,7),(0,2,6,7),(0,6,4,7),(0,4,5,7),(0,5,1,7))
        edges = {tuple(sorted((a,b))) for tet in tetrahedra for a in tet for b in tet if a < b}
        spanwise = []
        for a, b in edges:
            if ((a>>1)&1) == ((b>>1)&1):
                continue
            if (a>>1)&1:
                a,b = b,a
            za,xa,zb,xb = a>>2,a&1,b>>2,b&1
            values = ((1.-fy)*velocity[za:za+len(z)-1,j,xa:xa+len(x)-1,2]+
                      fy*velocity[zb:zb+len(z)-1,j+1,xb:xb+len(x)-1,2])
            positions = (1.-fy)*x[xa:xa+len(x)-1]+fy*x[xb:xb+len(x)-1]
            spanwise.append(values[:,(positions>=0.) & (positions<=300.)].ravel())
        selected_spanwise = np.concatenate(spanwise)
        spanwise_range = [float(selected_spanwise.min()),float(selected_spanwise.max())]
        for rank in range(2):
            receipt = json.loads((case/'outdat/render'/f'mesh_step{step:08d}_rank{rank}.json').read_text())
            for field, values in fields.items():
                for window in range(4):
                    name = field if window == 0 else f'{field}.x{(window-1)*100}_{window*100}'
                    observed = receipt['products'][name]['actual_range']
                    lo = 0. if window == 0 else (window-1)*100.
                    hi = 1100. if window == 0 else window*100.
                    xm = (x >= lo) & (x <= hi)
                    ym = y <= (150. if window == 0 else 20.)
                    selected = values[np.ix_(ym, xm)]
                    expected = [float(selected.min()), float(selected.max())]
                    error = float(np.max(np.abs(np.asarray(observed)-expected)))
                    if not np.isfinite(error) or error > 2e-10:
                        raise ValueError(f'CPU/GPU same-state plane range differs: {step}/{field}/{window}: {error}')
                    errors[f'{step}/{field}/{window}'] = error
            error = float(np.max(np.abs(np.asarray(receipt['products']['spanwise_slice']['actual_range'])-spanwise_range)))
            if not np.isfinite(error) or error > 2e-10:
                raise ValueError(f'CPU/GPU same-state spanwise plane range differs: {step}: {error}')
            errors[f'{step}/spanwise_slice/0'] = error
    return dict(max_abs=max(errors.values(), default=0.), ranges=errors,
                reference='CPU FP64 linear physical-plane interpolation on identical completed-step q')


def run_case(output, source, executable, shape, steps, images, baseline=None, checkpoint=False, sanitizer=False):
    case = output/('images_on' if images else 'images_off')
    prepare_case(source, case, shape, steps, images, checkpoint)
    runtime = environment('2,1,1')
    runtime.update(ASTR_GPU_MP_POSITIVITY='flux', ASTR_GPU_RK_TIMING='1',
                   ASTR_INSITU_WRITE_EPS='1' if eps_output_enabled() else '0',
                   ASTR_INSITU_CONFIG='datin/input.insitu', ASTR_INSITU_TIMING='1',
                   ASTR_INSITU_M12_W_COLOR_MAX=str(W_COLOR_MAX),
                   ASTR_INSITU_TEST_ORACLE_IO='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for key in ('DISPLAY', 'PYTHONPATH', 'CATALYST_IMPLEMENTATION_PREFER_ENV', 'VTK_EGL_DEVICE_INDEX'):
        runtime.pop(key, None)
    command = [str(MPIEXEC), '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,vader,tcp',
               '--bind-to', 'none', '-np', '2']
    if sanitizer:
        command += ['compute-sanitizer', '--tool', 'memcheck', '--error-exitcode', '99',
                    '--log-file', str(case/'memcheck.%p.log')]
    command += [str(executable), 'run', 'datin/input.dat']
    receipt = dict(status='prepared', shape=shape, steps=steps, dt=.02, topology='2,1,1', images=images,
        fields=False, checkpoint=checkpoint, wall_curves_every=50, probes_every=1,
        image_intervals=dict(slices=100, q=250, streamlines=250), command=command,
        numerical_scope='first-output coverage, not transition/production validation',
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        pipeline_sha256=hashlib.sha256((ROOT/'scripts/insitu/device_render_pipeline.py').read_bytes()).hexdigest(),
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (case/'datin').iterdir()},
        runtime={k: v for k, v in runtime.items() if k.startswith(('ASTR_', 'OMPI_MCA_'))})
    path = case/'receipt.json'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    gate = CflGate(case/'run.log', 1., .02, 0, steps)
    def progress():
        gate.check()
        if sum(p.stat().st_size for p in case.rglob('*') if p.is_file()) > 8*GIB:
            raise ValueError('8 GiB run-directory budget exceeded')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) > 32*GIB:
            raise ValueError('32 GiB coverage-batch budget exceeded')
    started = time.monotonic()
    receipt['status'] = 'running'
    path.write_text(json.dumps(receipt, indent=2)+'\n')
    validation_started = None
    try:
        with (case/'run.log').open('wb') as stream:
            resources = run_monitored(command, case, runtime, stream, case/'resources.json', baseline=baseline,
                device_extra_budget_bytes=6*GIB, host_extra_budget_bytes=16*GIB, device_reserve_bytes=2*GIB,
                timeout_seconds=14400, check_progress=progress)
        receipt['wall_seconds'] = resources['elapsed_seconds']
        validation_started = time.monotonic()
        gate.check(final=True)
        if 'The job is done!' not in (case/'run.log').read_text(errors='replace'):
            raise ValueError('Missing normal solver termination')
        check_compact(case, steps)
        if images:
            check_images(case, steps)
        elif list((case/'outdat/render').glob('*.jpeg')):
            raise ValueError('Images-off case unexpectedly rendered')
        if not checkpoint and any(p.is_file() for name in ('checkpoints', 'fields')
                                  for p in (case/'outdat/new'/name).rglob('*')):
            raise ValueError('Coverage must not write checkpoints or full fields')
        if sanitizer:
            logs = list(case.glob('memcheck.*.log'))
            if len(logs) != 2 or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                raise ValueError('Compact-kernel memcheck failed')
        if checkpoint:
            receipt['same_state_compact'] = check_same_state_compact(case, steps)
            if images:
                receipt['same_state_slices'] = check_same_state_slices(case, steps)
        receipt.update(status='completed', max_cfl=gate.maximum, resources=resources)
    except BaseException as exc:
        receipt.update(status='failed', error=str(exc), max_cfl=gate.maximum)
        raise
    finally:
        receipt.setdefault('wall_seconds', time.monotonic()-started)
        if validation_started is not None:
            receipt['validation_seconds'] = time.monotonic()-validation_started
        path.write_text(json.dumps(receipt, indent=2)+'\n')
    return case, resources


def finish_comparison(output, component_check=False):
    off, on = [output/label for label in ('images_off', 'images_on')]
    a, b = [json.loads((case/'receipt.json').read_text()) for case in (off, on)]
    if any(record['status'] != 'completed' for record in (a, b)) or any(
            a[key] != b[key] for key in ('shape', 'steps', 'dt', 'topology', 'executable_sha256')):
        raise ValueError('Only matched, completed coverage runs can be finalized')
    steps = a['steps']
    for case in (off, on):
        check_compact(case, steps)
    check_images(on, steps)
    if component_check:
        with h5py.File(off/'outdat/new/checkpoints'/f'step{steps:012d}/state.h5') as a, \
                h5py.File(on/'outdat/new/checkpoints'/f'step{steps:012d}/state.h5') as b:
            for key in a:
                if key.startswith('q'):
                    np.testing.assert_array_equal(a[key][...], b[key][...], err_msg='Observer changed authoritative state')
        control = Path('outdat/new/checkpoints')/f'step{steps:012d}/control.bin'
        if (off/control).read_bytes() != (on/control).read_bytes():
            raise ValueError('Observer changed solver controls or wall RNG state')
    for filename in ['probes.csv']+[f'wall.step{s:08d}.rank{r:04d}.csv' for s in range(50,steps+1,50) for r in range(2)]:
        if (off/'outdat/render'/filename).read_bytes() != (on/'outdat/render'/filename).read_bytes():
            raise ValueError('Images change compact physical diagnostics: '+filename)
    from plot_m12_hd_diagnostics import plot_curves
    plot_curves(on/'outdat/render')
    comparison = dict(status='passed-local-coverage-not-physical-validation',
                      images_off_seconds=a['wall_seconds'], images_on_seconds=b['wall_seconds'],
                      added_seconds=b['wall_seconds']-a['wall_seconds'],
                      timing_scope='monitored MPI execution including initialization/rendering/encoding, excluding post-run checks and curve plotting',
                      baseline='same compact diagnostics enabled, image clocks disabled',
                      same_monitor_and_wall_data=True)
    (output/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
    print(json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--component-check', action='store_true')
    parser.add_argument('--memcheck', action='store_true')
    parser.add_argument('--finalize-only', action='store_true',
                        help='Check completed runs and finish missing curves/summary without running MPI')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.finalize_only:
        if args.memcheck:
            parser.error('--finalize-only requires a completed images-off/on pair')
        finish_comparison(args.output, args.component_check)
        return
    if args.executable is None:
        parser.error('--executable is required for a new coverage run')
    missing = [name for name in ('latex', 'dvipng', 'dvips', 'gs')
               if not args.memcheck and shutil.which(name) is None]
    if missing:
        parser.error('Curve rendering tools not in PATH: '+', '.join(missing))
    args.output.mkdir(parents=True, exist_ok=False)
    executable = args.executable.resolve(strict=True)
    (args.output/'bin').mkdir()
    frozen = args.output/'bin/astr'
    shutil.copy2(executable, frozen)
    shape = (64, 32, 24) if args.component_check else REFINED_INTERVALS
    steps = 50 if args.memcheck else 250
    off, baseline = run_case(args.output, args.source, frozen, shape, steps, False,
                             checkpoint=args.component_check, sanitizer=args.memcheck)
    if args.memcheck:
        print('PASS bounded compact-kernel memcheck', flush=True)
        return
    run_case(args.output, args.source, frozen, shape, steps, True, baseline,
             checkpoint=args.component_check)
    finish_comparison(args.output, args.component_check)


if __name__ == '__main__':
    main()
