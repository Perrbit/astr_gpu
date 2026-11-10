"""Real-solver FP64 periodic KEEP field and exact-restart comparisons."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case, compare_fields
from compare_q_validation_snapshots import read_q_snapshot
from run_keep_metric_audit import audit_geometry

ROOT = Path(__file__).resolve().parents[2]
FINAL = Path('outdat/new/checkpoints/step000000000004')


def check_filtered_primitives(directory):
    suffix='.step00000000.rk01.rank00000000.bin'
    state=read_q_snapshot(directory/'outdat'/('rhs.pre_rhs'+suffix))
    im,jm,km,hm,nq=state.header
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    physical=(slice(hm,hm+im+1),slice(hm,hm+jm+1),slice(hm,hm+km+1))
    q=state.values.reshape((*shape,nq),order='F')[physical]
    with (directory/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
        header=np.fromfile(stream,dtype=np.int32,count=4)
        primitive=np.fromfile(stream,dtype=np.float64).reshape((*shape,6),order='F')[physical]
    if tuple(header)!=(im,jm,km,hm):
        raise AssertionError('filtered primitive snapshot layout differs')
    rho=q[...,0]
    velocity=q[...,1:4]/rho[...,None]
    pressure=0.4*(q[...,4]-0.5*np.sum(q[...,1:4]*velocity,axis=-1))
    # This driver uses the fixed perfect-gas TGV fixture, gamma=1.4 and Ma=0.1.
    expected=np.concatenate([rho[...,None],velocity,pressure[...,None],(1.4*0.1**2*pressure/rho)[...,None]],axis=-1)
    differences=np.max(abs(primitive-expected)/np.maximum(1,np.maximum(abs(primitive),abs(expected))),axis=(0,1,2))
    if not np.isfinite(differences).all() or np.any(differences>2e-10):
        raise AssertionError(f'filtered physical primitives do not match current q: {directory}: {differences}')
    return dict(zip(['rho','u','v','w','p','T'],map(float,differences)))


def compare_frozen_rhs(first,second,stage='conv'):
    suffix='.step00000000.rk01.rank00000000.bin'
    cpu=read_q_snapshot(first/'outdat'/('rhs.pre_rhs'+suffix))
    gpu=read_q_snapshot(second/'outdat'/('rhs.pre_rhs'+suffix))
    if cpu.header!=gpu.header:
        raise AssertionError('frozen-state layouts differ')
    im,jm,km,hm,nq=cpu.header
    shape=(im+2*hm+1,jm+2*hm+1,km+2*hm+1)
    indices=np.indices(shape)-hm
    sizes=np.array([im,jm,km])[:,None,None,None]
    outside=(indices<0)|(indices>sizes)
    # Physical nodes and face halos used by D6/Pade; no transverse corner reads.
    mask=(outside.sum(axis=0)<=1)&((indices>=-3)&(indices<=sizes+3)).all(axis=0)
    q1=cpu.values.reshape((*shape,nq),order='F')[mask]
    q2=gpu.values.reshape((*shape,nq),order='F')[mask]
    if not np.array_equal(q1,q2):
        raise AssertionError(f'authoritative frozen q differs: {np.max(abs(q1-q2))}')
    geometry=compare_fields(first/'outdat/new/resources/geometry.h5',second/'outdat/new/resources/geometry.h5')
    primitive=[]
    rhs=[]
    for directory in (first,second):
        with (directory/'outdat'/('rhs.pre_rhs_primitives'+suffix)).open('rb') as stream:
            header=np.fromfile(stream,dtype=np.int32,count=4)
            data=np.fromfile(stream,dtype=np.float64)
        if tuple(header)!=(im,jm,km,hm) or data.size!=np.prod(shape)*6:
            raise AssertionError('primitive snapshot layout differs')
        primitive.append(data.reshape((*shape,6),order='F')[mask])
        contributions=[]
        for label in (('full','conv') if stage=='diffusion' else (stage,)):
            with (directory/'outdat'/('rhs.'+label+suffix)).open('rb') as stream:
                header=np.fromfile(stream,dtype=np.int32,count=4)
                data=np.fromfile(stream,dtype=np.float64)
            if tuple(header)!=(im,jm,km,nq) or data.size!=(im+1)*(jm+1)*(km+1)*nq:
                raise AssertionError('RHS snapshot layout differs')
            contributions.append(data.reshape((im+1,jm+1,km+1,nq),order='F'))
        rhs.append(contributions[0]-contributions[1] if stage=='diffusion' else contributions[0])
    if any(not np.isfinite(x).all() for x in primitive+rhs):
        raise AssertionError('nonfinite frozen-state input or RHS')
    primitive_error=float(np.max(abs(primitive[0]-primitive[1])/np.maximum(1.0,
        np.maximum(abs(primitive[0]),abs(primitive[1])))))
    if primitive_error>2e-10:
        raise AssertionError('derived primitive representation differs beyond roundoff comparison gate')
    result={'authoritative_q_bit_identical_on_stencil':True,'geometry_datasets_bit_identical':geometry,
            'primitive_max_scaled':primitive_error,'rhs':{}}
    if stage=='diffusion':
        result['max_viscous_rhs']=[float(np.max(abs(value))) for value in rhs]
        if not all(value>0 for value in result['max_viscous_rhs']):
            raise AssertionError('ME6 was selected but native viscous RHS is zero')
        if any(np.any(value[...,0]!=0) for value in rhs):
            raise AssertionError('ME6 changed total density RHS')
    for component in range(nq):
        x,y=rhs[0][...,component],rhs[1][...,component]
        difference=abs(x-y)
        scaled=difference/np.maximum(1.0,np.maximum(abs(x),abs(y)))
        result['rhs'][f'q{component+1:04d}']=dict(max_absolute=float(difference.max()),
            max_scaled=float(scaled.max()),index=list(map(int,np.unravel_index(scaled.argmax(),scaled.shape))))
        if scaled.max()>2e-10:
            raise AssertionError(f'frozen RHS differs in component {component+1}: {scaled.max()}')
    return result


def compare_q(first, second):
    result = {}
    with h5py.File(first) as a, h5py.File(second) as b:
        for component in range(1, 6):
            name = f'q{component:04d}'
            x, y = a[name][:], b[name][:]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise AssertionError(f'nonfinite {name}')
            difference = abs(x-y)
            scaled = difference/np.maximum(1.0, np.maximum(abs(x), abs(y)))
            result[name] = dict(max_absolute=float(difference.max()), max_scaled=float(scaled.max()),
                                index=list(map(int,np.unravel_index(scaled.argmax(),scaled.shape))))
            if scaled.max() > 2e-10:
                raise AssertionError(f'KEEP field mismatch {name}: {result[name]}')
        for state in (a,b):
            rho = state['q0001'][:]
            internal = state['q0005'][:] - sum(state[f'q{q:04d}'][:]**2 for q in (2,3,4))/(2*rho)
            if not (rho>0).all() or not (internal>0).all():
                raise AssertionError('nonpositive density/internal energy')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--axes',nargs='*',choices=['x','y','z'],default=['x','y','z'])
    parser.add_argument('--memcheck',action='store_true')
    parser.add_argument('--filter',choices=['off','explicit10','compact10'],default='off')
    parser.add_argument('--form',choices=['standard','keep'],default='keep')
    parser.add_argument('--scheme',choices=['643e','643c'],default='643e')
    parser.add_argument('--diffusion-scheme',choices=['643e','643c'])
    parser.add_argument('--grid',help='Global intervals, e.g. 32,32,32')
    parser.add_argument('--mapping',choices=['periodic'])
    parser.add_argument('--cartesian-grid',type=Path,help='Existing separable periodic grid')
    parser.add_argument('--viscous-form',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--transport',choices=['pinned','device-aware'],default='pinned')
    parser.add_argument('--multi-axis',action='store_true',help='Add NP=4 xy and NP=8 xyz dependencies')
    parser.add_argument('--rhs',action='store_true',help='Compare first-stage frozen q and convection RHS at NP=1')
    opts = parser.parse_args()
    opts.output = opts.output.resolve()
    opts.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    numerics={'convection_form':opts.form,'viscous_form':opts.viscous_form}
    if opts.diffusion_scheme:
        numerics['difschm']=opts.diffusion_scheme
    if opts.filter!='off':
        numerics['filter_scheme']=opts.filter
    filtered=opts.filter!='off'
    grid=opts.grid or ('32,32,32' if opts.mapping else '16,16,16')
    geometry_options=dict(grid=grid,tgv_mapping=opts.mapping,tgv_cartesian_grid=opts.cartesian_grid)
    geometric=opts.mapping is not None or opts.cartesian_grid is not None
    report = dict(grid_intervals=list(map(int,grid.split(','))),steps=4,convection_form=opts.form,
                  diffusion_scheme=opts.diffusion_scheme or opts.scheme,
                  scheme=opts.scheme,viscous_form=opts.viscous_form,filter=opts.filter,transport=opts.transport,
                  cases={},geometry={},filtered_primitives={})
    singles = {}
    matrix=[(1,'x',None)]+[(2,axis,None) for axis in opts.axes]
    if opts.multi_axis:
        matrix += [(4,'xy',(2,2,1)),(8,'xyz',(2,2,2))]
    for ranks,axis,topology in matrix:
        finals = {}
        runs = {}
        for backend in ('cpu','gpu'):
            args = SimpleNamespace(output=opts.output,executable=getattr(opts,backend).resolve(),
                mpiexec=opts.mpiexec.resolve(),case='tgv',mode='steps',restart_step=2,initial_dimension=0,
                initial_restart=False,legacy_statistics=False,statistics=False,filter_workspace='scalar',
                force='feedback',axis=axis if len(axis)==1 else 'x',no_samples=True,wall_samples=False,unrestricted=True)
            label = f'keep_{axis}'
            continuous,_ = run_case(args,ROOT,backend,ranks,label,4,lfilter=filtered,scheme=opts.scheme,
                                    numerics=numerics,memcheck=opts.memcheck and backend=='gpu',
                                    topology=topology,solver_halo_transport=opts.transport,
                                    rhs_snapshot_step=0 if (opts.rhs or filtered) and ranks==1 else None,**geometry_options)
            runs[backend]=continuous
            if filtered and ranks==1:
                report['filtered_primitives'][backend]=check_filtered_primitives(continuous)
            if opts.form=='keep' and 'CONVECTION FORM: original arithmetic-average KEEP' not in (continuous/'run.log').read_text():
                raise AssertionError('KEEP input selection missing from startup')
            identity=b'astr-original-keep6c-v1' if opts.scheme=='643c' else b'astr-original-keep6-v1'
            if opts.form=='keep' and identity not in (continuous/FINAL/'control.bin').read_bytes():
                raise AssertionError('explicit KEEP numerical identity missing')
            if opts.form=='keep' and b'/g1' not in (continuous/FINAL/'control.bin').read_bytes():
                raise AssertionError('KEEP G1 geometry identity missing')
            if geometric:
                metric=audit_geometry(continuous/'outdat/new/resources/geometry.h5',opts.scheme)
                report['geometry'][f'{backend}_np{ranks}_{axis}']=metric
                if max(metric['metric_identity_max_per_component'])>2e-10 or \
                   max(metric['constant_state_rhs_max_per_component'])>2e-10:
                    raise AssertionError(f'G1 geometry/free-stream gate failed: {continuous}')
            if opts.viscous_form=='me6_opti':
                if 'ASTR_VISCOUS_FORM=me6_opti' not in (continuous/'run.log').read_text():
                    raise AssertionError('ME6 input selection missing from startup')
                if b'/me6opti-podd-v1' not in (continuous/FINAL/'control.bin').read_bytes():
                    raise AssertionError('ME6 numerical identity missing')
            finals[backend] = continuous/FINAL/'state.h5'
            if ranks==1:
                singles[backend] = finals[backend]
            if ranks>1:
                report['cases'][f'{backend}_np{ranks}_{axis}_vs_np1'] = compare_q(singles[backend],finals[backend])
            if ranks==2 and axis=='x':
                seed = continuous/'outdat/new/checkpoints/step000000000002'
                resumed,_ = run_case(args,ROOT,backend,ranks,label+'_resumed',4,restore=seed,lfilter=filtered,
                                     numerics=numerics,scheme=opts.scheme,solver_halo_transport=opts.transport,**geometry_options)
                compare_fields(continuous/FINAL/'state.h5',resumed/FINAL/'state.h5')
                for filename in ('control.bin','archives.bin','insitu_control.bin'):
                    if (continuous/FINAL/filename).read_bytes()!=(resumed/FINAL/filename).read_bytes():
                        raise AssertionError(f'exact KEEP restart differs: {backend} {filename}')
                report['cases'][f'{backend}_np2_x_exact_restart'] = 'all state datasets and controls identical'
                if backend=='cpu' and opts.viscous_form=='me6_opti':
                    changed_viscous=dict(numerics,viscous_form='standard')
                    run_case(args,ROOT,backend,ranks,label+'_changed_viscous_exact',4,restore=seed,
                        lfilter=filtered,numerics=changed_viscous,scheme=opts.scheme,reject='compact operator identity mismatch',**geometry_options)
                    migrated,_=run_case(args,ROOT,backend,ranks,label+'_changed_viscous_nonexact',4,restore=seed,
                        lfilter=filtered,numerics=changed_viscous,scheme=opts.scheme,exact_restart=False,**geometry_options)
                    compare_q(migrated/FINAL/'state.h5',migrated/FINAL/'state.h5')
                    report['cases']['changed_viscous_form']='exact rejected; explicit non-exact migration completed'
                if backend=='cpu' and opts.filter!='off':
                    changed_filter=dict(numerics,filter_scheme='compact10' if opts.filter=='explicit10' else 'explicit10')
                    run_case(args,ROOT,backend,ranks,label+'_changed_filter_exact',4,restore=seed,
                        lfilter=True,numerics=changed_filter,scheme=opts.scheme,
                        reject='compact operator identity mismatch',**geometry_options)
                    report['cases']['changed_filter'] = 'exact restart rejected different effective filter'
                if backend=='cpu' and opts.form=='keep':
                    changed=dict(numerics,convection_form='standard')
                    if opts.diffusion_scheme and opts.diffusion_scheme!=opts.scheme:
                        # Isolate a valid gradient change rather than asking standard convection to use mismatched shared fds.
                        changed=dict(numerics,difschm=opts.scheme)
                    run_case(args,ROOT,backend,ranks,label+'_changed_exact',4,restore=seed,lfilter=filtered,
                             numerics=changed,scheme=opts.scheme,reject='compact operator identity mismatch',**geometry_options)
                    migrated,_ = run_case(args,ROOT,backend,ranks,label+'_changed_nonexact',4,restore=seed,
                        lfilter=filtered,numerics=changed,scheme=opts.scheme,exact_restart=False,**geometry_options)
                    compare_q(migrated/FINAL/'state.h5',migrated/FINAL/'state.h5')
                    report['cases']['changed_form'] = 'exact rejected; explicit non-exact migration completed'
        report['cases'][f'cpu_gpu_np{ranks}_{axis}'] = compare_q(finals['cpu'],finals['gpu'])
        if opts.rhs and ranks==1:
            report['frozen_rhs']=compare_frozen_rhs(runs['cpu'],runs['gpu'])
            report['frozen_full_rhs']=compare_frozen_rhs(runs['cpu'],runs['gpu'],stage='full')
            if opts.viscous_form=='me6_opti':
                report['frozen_viscous_rhs']=compare_frozen_rhs(runs['cpu'],runs['gpu'],stage='diffusion')
        (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'KEEP np{ranks} {axis}: fields compared',flush=True)
    report['status']='passed'
    (opts.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
