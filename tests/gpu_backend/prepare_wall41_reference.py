"""Freeze low-order physical-end states from existing CPU/CUDA Fortran solvers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'gpu_validation'))
from run_keep_simple_boundary_gate import prepare
from run_cpp_multi_tgv_gate import raw_field
from run_cpp_tgv_gate import compare as compare_fields,checkpoint
from generate_curvilinear_tgv_grid import mapped_grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--precision',choices=['fp64','fp32'],default='fp64')
    parser.add_argument('--native-reference',type=Path,help='actual original dual-runtime native32 executable')
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gradients', action='store_true')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--force-fixed', type=float, default=0.)
    parser.add_argument('--filter', choices=['off','explicit10','compact10'], default='off')
    parser.add_argument('--grid', type=int, default=16)
    parser.add_argument('--convection-scheme', choices=['643e','643c'], default='643e')
    parser.add_argument('--diffusion-scheme', choices=['643e','643c'], default='643e')
    parser.add_argument('--viscous-form', choices=['standard','me6_opti'], default='standard')
    parser.add_argument('--convection-form',choices=['standard','keep'],default='keep')
    parser.add_argument('--mapping', choices=['cartesian','wavy'], default='cartesian')
    parser.add_argument('--boundary-type',type=int,choices=[41,42,50,60,411,421],default=41)
    parser.add_argument('--xslip',type=float,default=10.)
    parser.add_argument('--wall-axes',nargs='+',choices=['x','y','z'],default=['x','y','z'])
    args = parser.parse_args()
    if args.precision=='fp32' and args.native_reference is None:
        parser.error('native32 freezing requires an actual native executable')
    if args.boundary_type in (411,421) and args.wall_axes!=['y']:
        parser.error('411/421 are implemented only on y faces')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    dtype=np.float32 if args.precision=='fp32' else np.float64
    def compare(a,b):
        return compare_fields(np.asarray(a,np.float64),np.asarray(b,np.float64),limit=None if args.precision=='fp32' else 2e-10)
    def read_field(path,primitive=False):
        return raw_field(path,primitive=primitive,precision=args.precision)
    if args.native_reference:shutil.copy2(args.native_reference,root/'original_native')
    report = {}
    for axis, letter in enumerate('xyz'):
        if letter not in args.wall_axes:
            continue
        states = []
        for backend, binary in [('cpu', 'astr_cpu'), ('gpu', 'astr_cuda_fortran')]:
            directory = root/(backend+'_'+letter)
            prepare(directory, str(args.boundary_type)+letter, backend, scheme=args.convection_scheme, diffusion=args.diffusion_scheme, filtering=args.filter,
                    steps=args.steps, force_fixed=args.force_fixed, intervals=args.grid, viscosity=args.viscous_form,
                    convection=args.convection_form,xslip=args.xslip)
            if args.mapping=='wavy':
                coordinates=mapped_grid(args.grid,args.grid,args.grid,.15,letter+'-wavy')[:3]
                # Preserve the fixture's 2pi/pi/2pi lengths. Initial fields
                # remain prescribed in logical coordinates on this new mesh.
                coordinates[1][...]*=.5
                with h5py.File(directory/'datin/grid.simple.h5','r+') as grid:
                    for name,value in zip('xyz',coordinates):
                        grid[name][...]=value.transpose(2,1,0)
            env = os.environ | dict(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0', ASTR_FORCE_MPI_TOPOLOGY='1,1,1',
                ASTR_GPU_SYNC_MODE='explicit', ASTR_VALIDATION_RHS_STEP='0', ASTR_VALIDATION_RHS_PREFIX='outdat/rhs')
            if args.steps>1:
                env['ASTR_VALIDATION_RHS_STEP_END'] = str(args.steps-1)
            if args.gradients:
                env['ASTR_VALIDATION_GRADIENTS'] = '1'
            with (directory/'run.log').open('w') as log:
                executable=args.native_reference if args.native_reference else args.reference/'bin'/binary
                subprocess.run([str(args.mpiexec), '-n', '1', str(executable.resolve()), 'run'],
                    cwd=directory, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            with h5py.File(directory/f'outdat/new/checkpoints/step{args.steps:012d}/state.h5') as state:
                if state['q0001'].dtype.itemsize!=np.dtype(dtype).itemsize:
                    raise AssertionError('actual physical reference numerical kind mismatch')
                body=np.stack([state[f'q{c:04d}'][:].astype(np.float64) for c in range(1,6)],axis=-1)
                if not np.isfinite(body).all() or not (body[...,0]>0).all() or not (
                    body[...,4]-.5*np.sum(body[...,1:4]**2,axis=-1)/body[...,0]>0).all():
                    raise AssertionError('physical reference nonfinite/nonpositive state')
            stage_states = []
            for stage in (1, 2, 3):
                suffix = f'.step00000000.rk{stage:02}.rank00000000.bin'
                header, values = read_field(directory/('outdat/rhs.pre_rhs_primitives'+suffix), primitive=True)
                h = header[3];upper = header[:3]
                physical = values[tuple(slice(h,h+n+1) for n in upper)+(slice(None),)]
                if args.boundary_type in (60,411,421):
                    with h5py.File(directory/'outdat/new/resources/geometry.h5') as geometry:
                        normal=np.stack([geometry[f'q{5+axis+3*c:04}'][...].transpose(2,1,0) for c in range(3)],axis=-1)
                    normal/=np.linalg.norm(normal,axis=-1)[...,None]
                for side in (0, 1):
                    wall = np.take(physical, 0 if side==0 else upper[axis], axis=axis)
                    v1=np.take(physical,1 if side==0 else upper[axis]-1,axis=axis)
                    v2=np.take(physical,2 if side==0 else upper[axis]-2,axis=axis)
                    p1=v1[...,4];p2=v2[...,4]
                    compare(wall[...,4:5], ((4*p1-p2)/3)[...,None])
                    if args.boundary_type in (411,421):
                        n=np.take(normal,0 if side==0 else upper[axis],axis=axis)
                        extrapolated=(4*v1-v2)/3
                        expected_velocity=extrapolated[...,1:4]
                        expected_velocity=expected_velocity-n*np.sum(n*expected_velocity,axis=-1)[...,None]
                        slip=np.ones(expected_velocity.shape[:-1],dtype=bool)
                        if args.boundary_type==411:
                            with h5py.File(directory/'outdat/new/resources/geometry.h5') as geometry:
                                wx=geometry['q0001'][...].transpose(2,1,0)[:,0,:]
                            slip=(wx<=args.xslip) if side==0 else np.zeros_like(slip)
                        compare(wall[...,1:4],np.where(slip[...,None],expected_velocity,0.))
                        compare(wall[...,5:6],np.where(slip[...,None],extrapolated[...,5:6],1.))
                        continue
                    if args.boundary_type in (50,60):
                        extrapolated=(4*v1-v2)/3
                        expected_velocity=extrapolated[...,1:4]
                        if args.boundary_type==60:
                            n=np.take(normal,0 if side==0 else upper[axis],axis=axis)
                            expected_velocity=expected_velocity-n*np.sum(n*expected_velocity,axis=-1)[...,None]
                        compare(wall[...,1:4],expected_velocity)
                        temperature_selected=(axis==0) if args.boundary_type==50 else not(axis==2 and side==1)
                        if temperature_selected:
                            compare(wall[...,5:6],extrapolated[...,5:6])
                            density=wall[...,4:5]/wall[...,5:6]*(1.4*.3**2)
                            compare(wall[...,:1],density)
                        else:
                            compare(wall[...,:1],extrapolated[...,:1])
                            temperature=wall[...,4:5]/wall[...,:1]*(1.4*.3**2)
                            compare(wall[...,5:6],temperature)
                        continue
                    compare(wall[...,1:4], np.zeros_like(wall[...,1:4]))
                    if args.boundary_type==41:
                        expected_temperature=np.ones_like(wall[...,5:6])
                    else:
                        t1=np.take(physical,1 if side==0 else upper[axis]-1,axis=axis)[...,5]
                        t2=np.take(physical,2 if side==0 else upper[axis]-2,axis=axis)[...,5]
                        expected_temperature=((4*t1-t2)/3)[...,None]
                    compare(wall[...,5:6], expected_temperature)
                if args.gradients:
                    with (directory/('outdat/rhs.gradients'+suffix)).open('rb') as source:
                        gh = np.fromfile(source, np.int32, 5)
                        nodes = int(np.prod(np.asarray(upper)+1))
                        gradient = np.fromfile(source, dtype, nodes*12)
                    if tuple(gh[:3]) != tuple(upper) or gradient.size != nodes*12:
                        raise AssertionError('gradient snapshot layout')
                    metric = np.zeros((*values.shape[:3], 3, 3),dtype=dtype,order='F')
                    with h5py.File(directory/'outdat/new/resources/geometry.h5') as geometry:
                        for b in range(3):
                            for a in range(3):
                                metric[tuple(slice(h,h+n+1) for n in upper)+(a,b)] = geometry[f'q{5+a+3*b:04}'][...].transpose(2,1,0)
                    with (directory/f'gradient_input.rk{stage:02}.bin').open('xb') as packed:
                        np.array([*upper,h], np.int32).tofile(packed)
                        values.ravel(order='F').tofile(packed)
                        metric.ravel(order='F').tofile(packed)
                        gradient.tofile(packed)
                stage_states.append(physical)
            states.append(stage_states)
            with h5py.File(directory/'outdat/new/resources/geometry.h5') as geometry:
                points = np.indices(values.shape[:3])-h
                wrapped = tuple(points[a] % upper[a] for a in range(3))
                metric = np.empty((*values.shape[:3],3,3),dtype=dtype,order='F')
                for b in range(3):
                    for a in range(3):
                        physical_metric = geometry[f'q{5+a+3*b:04}'][...].transpose(2,1,0)
                        metric[...,a,b] = physical_metric[wrapped]
                        metric[tuple(slice(h,h+n+1) for n in upper)+(a,b)] = physical_metric
                physical_jac = geometry['q0004'][...].transpose(2,1,0)
                jac = physical_jac[wrapped]
                jac[tuple(slice(h,h+n+1) for n in upper)] = physical_jac
                off_diagonal=max(float(np.max(abs(metric[...,a,b]))) for a in range(3) for b in range(3) if a!=b)
                if not np.all(np.isfinite(physical_jac)) or np.min(physical_jac)<=0:
                    raise AssertionError('Reference geometry has nonpositive/nonfinite Jacobian')
                if args.mapping=='wavy' and off_diagonal<=1e-12:
                    raise AssertionError('Wavy reference did not exercise non-diagonal metrics')
                report.setdefault('geometry',{})[backend+'_'+letter]=dict(
                    jacobian_range=[float(np.min(physical_jac)),float(np.max(physical_jac))],
                    off_diagonal_max=off_diagonal)
            with (directory/'geometry_input.bin').open('xb') as packed:
                np.array([*upper,h],np.int32).tofile(packed)
                metric.ravel(order='F').tofile(packed)
                jac.ravel(order='F').tofile(packed)
        report[letter] = [compare(a,b) for a,b in zip(*states)]
        report.setdefault('terminal_by_axis',{})[letter]=compare(
            checkpoint(root/('gpu_'+letter)/f'outdat/new/checkpoints/step{args.steps:012d}/state.h5'),
            checkpoint(root/('cpu_'+letter)/f'outdat/new/checkpoints/step{args.steps:012d}/state.h5'))
    report['force_fixed'] = args.force_fixed
    report['filter_scheme'] = args.filter
    report['grid'] = args.grid
    report['convection_scheme'] = args.convection_scheme
    report['diffusion_scheme'] = args.diffusion_scheme
    report['viscous_form'] = args.viscous_form
    report['convection_form'] = args.convection_form
    report['mapping'] = args.mapping
    report['boundary_type'] = args.boundary_type
    report['wall_axes'] = args.wall_axes
    report['xslip'] = args.xslip
    report['precision']=args.precision
    if args.native_reference:
        report['original_binary_sha256']=hashlib.sha256(args.native_reference.read_bytes()).hexdigest()
        shutil.copy2(__file__,root/'preparation.py')
    (root/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'CPU/old GPU bc{args.boundary_type} {args.wall_axes}, {args.precision} state/closure differences recorded')
    if args.native_reference:
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        root.chmod(0o555)


if __name__ == '__main__':
    main()
