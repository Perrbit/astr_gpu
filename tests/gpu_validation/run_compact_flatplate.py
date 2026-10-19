#!/usr/bin/env python3
"""CP6 same-operator flat-plate gate; CFD advancement is performed by ASTR."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np

from prepare_m12_local_case import prepare, replace_value
from compare_q_validation_snapshots import read_q_snapshot, comparison_values
from compare_flowstate import read_flowstate
from run_compact_derivative_tgv import measure, check_primitive_eos


GRID = (96, 48, 32)
DT = 0.005
STEPS = 10


def configure(case, boundary, scheme, blowing, filtering='f'):
    path = case / 'datin/input.dat'
    lines = path.read_text().splitlines()
    replace_value(lines, 'conschm,difschm,rkscheme', f'743e,{scheme},rk3')
    if filtering=='t':
        index = next(i for i,line in enumerate(lines) if line.startswith('# nondimen,diffterm'))+1
        flags = lines[index].split(',')
        flags[2] = 't'
        lines[index] = ','.join(flags)
    if boundary == 'nscbc':
        start = next(i for i, line in enumerate(lines) if line.startswith('# bctype')) + 1
        lines[start+1] = '22,1.d0'
        lines[start+3] = '52'
    path.write_text('\n'.join(lines)+'\n')
    path = case / 'datin/controller'
    lines = path.read_text().splitlines()
    replace_value(lines, 'deltat', '5.d-3')
    path.write_text('\n'.join(lines)+'\n')
    if not blowing:
        path = case / 'datin/wallbs.dat'
        lines = path.read_text().splitlines()
        lines[2] = '0.0,0.0,20.0,40.0,60.0'
        path.write_text('\n'.join(lines)+'\n')
    metadata = json.loads((case/'fixture.json').read_text())
    metadata.update(deltat=DT, boundary=boundary, diffusion=scheme,
                    blowing='original legacy_random' if blowing else 'zero amplitude',filter=filtering)
    changed = {'643e diffusion'} if scheme=='643c' else set()
    if boundary=='nscbc':
        changed.add('boundary types')
    if not blowing:
        changed.add('wall blowing parameters')
    metadata['unchanged'] = [item for item in metadata['unchanged'] if item not in changed]
    metadata['changes'] += ['CP6 dt=0.005']+sorted(changed)
    (case/'fixture.json').write_text(json.dumps(metadata, indent=2)+'\n')


def assemble(case, step, phase='post_update', stage=3):
    rows = np.loadtxt(case/'datin/parallel.info', skiprows=3, dtype=int, ndmin=2)
    # Physical x/y ends are unique unknowns, z's terminal is a periodic copy.
    shape = (GRID[0]+1, GRID[1]+1, GRID[2])
    field = np.empty((*shape, 5))
    owners = np.zeros(shape, dtype=np.int32)
    for rank, _, _, _, im, jm, km, i0, j0, k0 in rows:
        snapshot = read_q_snapshot(case/f'diagnostics/state.{phase}.step{step:08d}.rk{stage:02d}.rank{rank:08d}.bin')
        values = comparison_values(snapshot, True).reshape((im+1,jm+1,km+1,5), order='F')
        ni = im + int(i0+im == GRID[0])
        nj = jm + int(j0+jm == GRID[1])
        region = (slice(i0,i0+ni),slice(j0,j0+nj),slice(k0,k0+km))
        field[region] = values[:ni,:nj,:km]
        owners[region] += 1
    if not np.all(owners == 1):
        raise ValueError('flat-plate physical/periodic ownership is not unique')
    return field


def geometry_reference(case):
    """Independent dense line algebra for the admitted separable geometry."""
    with h5py.File(case/'datin/grid.h5') as grid:
        coordinates = [grid['x'][0,0,:],grid['y'][0,:,0],grid['z'][:,0,0]]
    derivatives = []
    residuals = []
    for f in coordinates[:2]:
        n = len(f)
        matrix = np.eye(n)+np.eye(n,k=1)/3+np.eye(n,k=-1)/3
        matrix[0,1] = matrix[-1,-2] = 2
        matrix[1,0] = matrix[1,2] = matrix[-2,-1] = matrix[-2,-3] = .25
        rhs = np.empty(n)
        rhs[0] = -2.5*f[0]+2*f[1]+.5*f[2]
        rhs[-1] = 2.5*f[-1]-2*f[-2]-.5*f[-3]
        rhs[1] = .75*(f[2]-f[0])
        rhs[-2] = .75*(f[-1]-f[-3])
        rhs[2:-2] = (7/9)*(f[3:-1]-f[1:-3])+(1/36)*(f[4:]-f[:-4])
        derivative = np.linalg.solve(matrix,rhs)
        residual = np.max(np.abs(matrix@derivative-rhs))/(
            np.linalg.norm(matrix,np.inf)*np.max(np.abs(derivative))+np.max(np.abs(rhs)))
        if residual>1e-12:
            raise ValueError('independent coordinate line residual exceeds 1e-12')
        derivatives.append(derivative)
        residuals.append(float(residual))
    z = coordinates[2]
    slope = (z[-1]-z[0])/(len(z)-1)
    if not np.allclose(z,z[0]+slope*np.arange(len(z)),rtol=0,atol=2e-10):
        raise ValueError('CP6 z extrusion is not affine')
    dx,dy = derivatives
    expected = {'jacob':np.broadcast_to(slope*dy[None,:,None]*dx[None,None,:],(len(z),len(dy),len(dx))),
                'dxi11':np.broadcast_to(1/dx[None,None,:],(len(z),len(dy),len(dx))),
                'dxi22':np.broadcast_to(1/dy[None,:,None],(len(z),len(dy),len(dx))),
                'dxi33':np.full((len(z),len(dy),len(dx)),1/slope)}
    result = {'coordinate_residuals':residuals,'fields':{}}
    with h5py.File(case/'diagnostics/geometry.h5') as geometry:
        for key in geometry:
            actual = geometry[key][...]
            reference = expected[key] if key in expected else np.zeros_like(actual)
            result['fields'][key] = measure(reference[...,None],actual[...,None])
    return result


def viscous_scale(case):
    """Global maximum over rank-local estimates, not a compact/RK stability proof."""
    max_diffusivity = 0.0
    for qfile in sorted((case/'diagnostics').glob('state.pre_rhs.step00000000.rk01.rank*.bin')):
        snap = read_q_snapshot(qfile)
        im,jm,km,hm,nq = snap.header
        q = comparison_values(snap, True).reshape((im+1,jm+1,km+1,nq), order='F')
        rho = q[...,0]
        internal = q[...,4]-0.5*np.sum(q[...,1:4]**2,axis=-1)/rho
        temperature = internal*0.4*1.4*12**2/rho
        s = 110.4/594.3
        viscosity = temperature*np.sqrt(temperature)*(1+s)/(temperature+s)/1e4
        diffusivity = np.maximum(4/3*viscosity/rho, 1.4/0.72*viscosity/rho)
        max_diffusivity = max(max_diffusivity,float(np.max(diffusivity)))
    with h5py.File(case/'datin/grid.h5') as grid:
        spacing = [float(np.min(np.diff(grid[key][...],axis=axis)))
                   for key,axis in (('x',2),('y',1),('z',0))]
    rate = max_diffusivity*sum(1/h**2 for h in spacing)
    return {'spacing_min':spacing, 'max_diffusivity':max_diffusivity,
            'dt_diffusivity_sum_inverse_h2':DT*rate,
            'inverse_diffusivity_rate':1/rate,
            'interpretation':'local spacing estimate, not the compact discrete spectral radius'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=Path, required=True)
    parser.add_argument('--gpu', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--topology', default='1,1,1')
    parser.add_argument('--boundary', choices=('basic','nscbc'), default='basic')
    parser.add_argument('--diffusion', choices=('643c','643e'), default='643c')
    parser.add_argument('--blowing', action='store_true')
    parser.add_argument('--filter', choices=('f','t'), default='f')
    parser.add_argument('--workspace', choices=('scalar','full'), default='scalar')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--compact-transport', choices=('pinned','device-aware'), default='pinned')
    instrumentation = parser.add_mutually_exclusive_group()
    instrumentation.add_argument('--gpu-sanitizer', choices=('memcheck','initcheck'))
    instrumentation.add_argument('--wall-call-check', action='store_true')
    args = parser.parse_args()
    if args.blowing and args.baseline:
        raise ValueError('random blowing is compared only at a fixed topology')
    if args.wall_call_check and not args.blowing:
        parser.error('--wall-call-check requires --blowing')
    out = args.out.resolve()
    if not args.check_only:
        out.mkdir(parents=True, exist_ok=False)
    ranks = int(np.prod(tuple(map(int,args.topology.split(',')))))
    report = {'status':'running','grid':GRID,'topology':args.topology,'dt':DT,
              'full_steps':STEPS,'boundary':args.boundary,'diffusion':args.diffusion,
              'blowing':args.blowing,'compact_transport':args.compact_transport,
              'filter':args.filter,'workspace':args.workspace}
    for backend,exe in (('cpu',args.cpu),('gpu',args.gpu)):
        case = out/backend
        if not args.check_only:
            prepare(args.source,case,backend,GRID)
            configure(case,args.boundary,args.diffusion,args.blowing,args.filter)
            env = {k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
            env.update(ASTR_FORCE_MPI_TOPOLOGY=args.topology,ASTR_WALL_BLOWING_MODE='legacy_random',
                       ASTR_GPU_COMPACT_COMPARE='1',ASTR_GPU_COMPACT_TRANSPORT=args.compact_transport,
                       ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_PRECISION_MODE='fp64',
                       ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_FILTER_WORKSPACE=args.workspace,
                       ASTR_NSCBC_FARFIELD_MODE='incoming_only',
                       ASTR_NSCBC_FARFIELD_RHO='1',ASTR_NSCBC_FARFIELD_U='1',
                       ASTR_NSCBC_FARFIELD_V='0',ASTR_NSCBC_FARFIELD_W='0',ASTR_NSCBC_FARFIELD_T='1',
                       ASTR_GEOMETRY_DUMP='diagnostics/geometry.h5',
                       ASTR_VALIDATION_RHS_PREFIX='diagnostics/state',ASTR_VALIDATION_RHS_STEP='0',
                       ASTR_VALIDATION_RHS_STEP_SECONDARY=str(STEPS-1),
                       OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                       OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_opal_cuda_support='0')
            if backend=='gpu' and args.compact_transport=='device-aware':
                env.update(OMPI_MCA_pml='ucx',OMPI_MCA_coll_hcoll_enable='0',
                           UCX_MEMTYPE_CACHE='n',UCX_TLS='self,sm,cuda_copy',
                           UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',
                           UCX_CUDA_IPC_ENABLE_MNNVL='no')
                env.pop('OMPI_MCA_opal_cuda_support')
            command = ['mpirun','--oversubscribe','--bind-to','none','-np',str(ranks)]
            if args.wall_call_check:
                command += ['--tag-output','gdb','--batch','--return-child-result',
                            '-x',str(Path(__file__).with_name('compact_wall_calls.gdb').resolve()),'--args']
            if backend=='gpu' and args.gpu_sanitizer:
                command += ['compute-sanitizer','--tool',args.gpu_sanitizer,'--error-exitcode','99',
                            '--log-file',str(case/(args.gpu_sanitizer+'.%p.log'))]
                if args.gpu_sanitizer=='memcheck':
                    command += ['--leak-check','full']
            command += [str(exe.resolve()),'run','datin/input.dat']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        text = (case/'run.log').read_text(errors='replace')
        if 'COMPUTATION CRASHED' in text or 'ERROR STOP' in text:
            raise ValueError(f'{backend} numerical run failed')
        if args.wall_call_check:
            counts = re.findall(r'\[\d+,(\d+)\]<stdout>:CP6_WALL_CALLS=(\d+)',text)
            ownership = np.loadtxt(case/'datin/parallel.info',skiprows=3,dtype=int,ndmin=2)
            expected = {int(row[0]):3*STEPS if row[2]==0 else 0 for row in ownership}
            observed = {int(rank):int(count) for rank,count in counts}
            if len(counts)!=ranks or observed!=expected:
                raise ValueError(f'{backend} wall calls differ: {observed}, expected {expected}')
            report[backend+'_wall_calls'] = observed
        if args.diffusion=='643c' and 'COMPACT GLOBAL: physical-end plate' not in text:
            raise ValueError(f'{backend} did not select the CP6 global operator')
        with h5py.File(case/'diagnostics/geometry.h5') as geometry:
            if any(not np.isfinite(geometry[key][...]).all() for key in geometry):
                raise ValueError(f'{backend} has nonfinite geometry; check the grid UDF')
            if np.min(geometry['jacob'][...])<=0:
                raise ValueError(f'{backend} has a nonpositive grid Jacobian')
        if args.diffusion=='643c':
            report[backend+'_geometry_reference'] = geometry_reference(case)
        cfl = [float(v) for v in re.findall(r'current CFL:\s*(\S+)',text)]
        if not cfl or not np.isfinite(cfl).all() or max(cfl)>.5:
            raise ValueError(f'{backend} CFL exceeds the approved 0.5: {cfl}')
        report[backend+'_cfl'] = max(cfl)
        if args.diffusion=='643c':
            eos = []
            for path in sorted((case/'diagnostics').glob('state.pre_rhs.step*.bin')):
                eos.append(check_primitive_eos(path,path.with_name(path.name.replace('pre_rhs.','pre_rhs_primitives.')),mach=12))
            if len(eos)!=6*ranks:
                raise ValueError('missing first/last pre-RHS EOS snapshots')
            report[backend+'_eos'] = eos
        if backend=='gpu':
            times = [float(v) for v in re.findall(r'ASTR_CFL complete_step=\d+ state_time=\s*(\S+)',text)]
            if len(times)!=STEPS or abs(times[-1]-STEPS*DT)>1e-15:
                raise ValueError('not ten complete GPU RK steps')
            checks = re.findall(r'COMPACT_SAME_STATE rank/kind/axis abs/scaled/residual:\s*'
                                r'(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',text)
            if args.diffusion=='643c' and len(checks)!=6*ranks:
                raise ValueError(f'expected six derivative/viscous same-state checks per rank, got {len(checks)}')
            report['same_state'] = [[int(v) for v in row[:3]]+[float(v) for v in row[3:]] for row in checks]
            if args.filter=='t' and args.diffusion=='643c':
                checks = re.findall(r'COMPACT_FILTER_SAME_STATE rank/axis/component abs/scaled/residual:\s*'
                                    r'(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)',text)
                count = (15 if args.workspace=='scalar' else 3)*ranks
                if len(checks)!=count:
                    raise ValueError(f'expected {count} filter same-state checks, got {len(checks)}')
                report['filter_same_state'] = [[int(v) for v in row[:3]]+[float(v) for v in row[3:]] for row in checks]
            if args.gpu_sanitizer:
                logs = list(case.glob(args.gpu_sanitizer+'.*.log'))
                if len(logs)!=ranks or any('ERROR SUMMARY: 0 errors' not in p.read_text() for p in logs):
                    raise ValueError('missing clean per-rank sanitizer report')
    report['phase_fields'] = {}
    with h5py.File(out/'cpu/diagnostics/geometry.h5') as a,h5py.File(out/'gpu/diagnostics/geometry.h5') as b:
        for key in a:
            if not np.array_equal(a[key][...],b[key][...]):
                raise ValueError(f'CPU/GPU initialized geometry differs: {key}')
    for path in sorted((out/'cpu/diagnostics').glob('state.*.bin')):
        if not path.name.startswith(('state.pre_rhs.step','state.post_update.step')):
            continue
        a,b = (read_q_snapshot(c/'diagnostics'/path.name) for c in (out/'cpu',out/'gpu'))
        if a.header!=b.header:
            raise ValueError('CPU/GPU phase identity differs')
        report['phase_fields'][path.name] = measure(
            comparison_values(a,True).reshape((-1,5),order='F'),
            comparison_values(b,True).reshape((-1,5),order='F'))
    if len(report['phase_fields'])!=12*ranks:
        raise ValueError('incomplete phase field inventory')
    ha,va = read_flowstate(out/'cpu')
    hb,vb = read_flowstate(out/'gpu')
    if ha!=hb or not np.array_equal(va[:,:1],vb[:,:1]):
        raise ValueError('statistics samples differ')
    report['statistics'] = measure(va[:,2:],vb[:,2:])
    report['viscous_scale_global'] = viscous_scale(out/'cpu')
    if args.baseline:
        previous = json.loads((args.baseline/'summary.json').read_text())
        for key in ('grid','dt','boundary','diffusion','blowing','full_steps','filter'):
            if previous[key]!=json.loads(json.dumps(report[key])):
                raise ValueError(f'partition baseline differs in {key}')
        report['topology_fields'] = {b:measure(assemble(args.baseline/b,STEPS-1),assemble(out/b,STEPS-1))
                                     for b in ('cpu','gpu')}
    report['status'] = 'pass'
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(out/'summary.json',flush=True)


if __name__=='__main__':
    main()
