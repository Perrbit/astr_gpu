"""Check native statistics windows when KEEP or ME6 operators change on restart."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_output_restart_validation import run_case
from run_output_restart_validation import compare_fields
from run_keep_periodic_gate import compare_q
from compare_q_validation_snapshots import read_q_snapshot
from insitu_statistics_reference import reference

ROOT=Path(__file__).resolve().parents[2]


def mean_metadata(path,kind):
    with h5py.File(path) as state:
        values=state['metadata'][:]
        if kind=='insitu':
            segment=state['statistics_segment'][:]
            regional=values[4:].view(np.float64)
            return dict(first_step=int(segment[2]),first_time=float(segment[3:4].view(np.float64)[0]),
                        last_step=int(values[2]),last_time=float(values[3:4].view(np.float64)[0]),
                        duration=float(regional[7]))
    return dict(samples=int(values[1]),first_step=int(values[2]),last_step=int(values[3]),
                first_time=float(values[4:5].view(np.float64)[0]),
                last_time=float(values[7:8].view(np.float64)[0]),
                duration=float(values[8:9].view(np.float64)[0]))


def read_q(path):
    with h5py.File(path) as state:
        q=np.stack([state[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,6)],axis=-1)
    if not np.isfinite(q).all() or not (q[...,0]>0).all():
        raise AssertionError(f'Nonfinite state or nonpositive density: {path}')
    internal=q[...,4]-.5*np.sum(q[...,1:4]**2,axis=-1)/q[...,0]
    if not np.isfinite(internal).all() or not (internal>0).all():
        raise AssertionError(f'Nonfinite or nonpositive internal energy: {path}')
    return q


def formal_quantities(path):
    with h5py.File(path) as state:
        data=np.stack([state[f'q{c:04d}'][:-1,:-1,:-1].transpose(2,1,0) for c in range(1,35)],axis=-1)
    duration=data[...,7];mass=data[...,8]
    if not np.isfinite(data).all() or not (duration>0).all() or not (mass>0).all():
        raise AssertionError('Invalid covered statistics or density-weighted duration')
    return np.concatenate([duration[...,None],(mass/duration)[...,None],data[...,9:15],
                           data[...,15:24]/duration[...,None],data[...,24:33]/mass[...,None]],axis=-1)


def independent_new_mean(case,kind):
    snapshot=read_q_snapshot(case/'outdat/rhs.post_update.step00000004.rk03.rank00000000.bin')
    im,jm,km,hm,nq=snapshot.header
    middle=snapshot.values.reshape((im+2*hm+1,jm+2*hm+1,km+2*hm+1,nq),order='F')
    middle=middle[hm:hm+im+1,hm:hm+jm+1,hm:hm+km+1]
    final=case/'outdat/new/checkpoints/step000000000006'
    fields=[read_q(case/'outdat/restore_probe.h5'),middle,read_q(final/'state.h5')]
    if kind=='insitu':
        samples=np.zeros((3,*middle[:-1,:-1,:-1].shape[:3],11))
        for index,q in enumerate(fields):
            q=q[:-1,:-1,:-1]
            samples[index,...,5]=q[...,0]
            samples[index,...,6:9]=q[...,1:4]/q[...,0,None]
        expected=reference(samples,np.array([.0005,.001,.0005]))
        actual=formal_quantities(final/'statistics.h5')
        target=np.concatenate([expected[...,:8],expected[...,14:32]],axis=-1)
        error=float(np.max(abs(actual-target)))
        if not np.isfinite(error) or error>2e-10:
            raise AssertionError(f'New-window Reynolds/Favre means or covariance differ: {error}')
        return error

    def moments(q):
        rho=q[...,0];u=q[...,1:4]/rho[...,None]
        p=.4*(q[...,4]-.5*np.sum(q[...,1:4]*u,axis=-1))
        t=(1.4*.1**2)*p/rho
        return np.stack([rho,q[...,1],q[...,2],q[...,3],p,rho*t,
                         rho*u[...,0]**2,rho*u[...,1]**2,rho*u[...,2]**2,
                         rho*u[...,0]*u[...,1],rho*u[...,0]*u[...,2],rho*u[...,1]*u[...,2]],axis=-1)

    expected=.0005*(moments(fields[0])+2*moments(fields[1])+moments(fields[2]))
    with h5py.File(final/'statistics.h5') as state:
        observed=np.stack([state[f'q{c:04d}'][:].transpose(2,1,0) for c in range(1,13)],axis=-1)
    a=observed/.002;b=expected/.002
    error=float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
    if not np.isfinite(error) or error>2e-10:
        raise AssertionError(f'New-window moments contain other history or have the wrong quadrature: {error}')
    return error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ranks',type=int,nargs='+',default=[1])
    parser.add_argument('--statistics-path',choices=['mean','insitu'],default='mean')
    parser.add_argument('--convection-form',choices=['keep','standard'],default='keep')
    parser.add_argument('--source-viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--target-viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--source-scheme',choices=['643e','643c'],default='643e')
    parser.add_argument('--target-scheme',choices=['643e','643c'],default='643c')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')

    def runtime(steps):
        return f"""&run_control max_steps={steps} /
&timestep mode='fixed',fixed_dt=1.d-3 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled={'t' if args.statistics_path=='mean' else 'f'},sample_clock='steps',sample_steps=1,write_steps=0 /
"""

    report=dict(status='running',scope='16-cubed periodic operator switch, no filter or repair; fixed dt=.001',
                statistics_path=args.statistics_path,convection_form=args.convection_form,
                source_scheme=args.source_scheme,target_scheme=args.target_scheme,
                source_viscosity=args.source_viscosity,target_viscosity=args.target_viscosity,cases={})
    for ranks in args.ranks:
        runs={}
        for backend in (['cpu','gpu'] if args.gpu else ['cpu']):
            opts=SimpleNamespace(output=out,executable=getattr(args,backend).resolve(),mpiexec=args.mpiexec.resolve(),
                case='tgv',mode='steps',restart_step=4,initial_dimension=0,initial_restart=False,
                legacy_statistics=args.statistics_path=='mean',statistics=args.statistics_path=='insitu',
                filter_workspace='scalar',force='feedback',
                axis='x',no_samples=True,wall_samples=False,unrestricted=True)
            common=dict(lfilter=False,scheme=args.source_scheme,numerics=dict(
                convection_form=args.convection_form,viscous_form=args.source_viscosity,
                conschm=args.source_scheme,difschm='643e'))
            if args.statistics_path=='insitu':
                # Pass the existing normalized unlimited sentinel to the legacy provider envelope.
                unlimited=np.iinfo(np.int64).max//4096
                common['insitu_config']=f"""&insitu_run
 enabled=t,statistics=t,render=f,statistics_window=0.d0,1.d0,
 output_directory='outdat',host_budget_bytes={unlimited},device_budget_bytes={unlimited},device_reserve_bytes=0
/
"""
            source,_=run_case(opts,ROOT,backend,ranks,'mean_source',6,run_config_text=runtime(6),**common)
            seed=source/'outdat/new/checkpoints/step000000000004'
            before=mean_metadata(seed/'statistics.h5',args.statistics_path)
            entry=dict(source=str(seed),source_statistics=before,same_method={})
            for exact in (True,False):
                same,_=run_case(opts,ROOT,backend,ranks,'mean_same_'+str(exact).lower(),6,
                    restore=seed,exact_restart=exact,run_config_text=runtime(6),**common)
                final=Path('outdat/new/checkpoints/step000000000006')
                metadata=mean_metadata(same/final/'statistics.h5',args.statistics_path)
                count_ok=args.statistics_path=='insitu' or metadata['samples']==7
                if metadata['first_step']!=0 or not count_ok or abs(metadata['duration']-.006)>2e-10:
                    raise AssertionError('Same-method history was unexpectedly reset')
                if exact:
                    for name in ('state.h5','statistics.h5'):
                        compare_fields(source/final/name,same/final/name)
                    for name in ('control.bin','archives.bin','insitu_control.bin'):
                        if (source/final/name).read_bytes()!=(same/final/name).read_bytes():
                            raise AssertionError(f'Exact statistics continuation changed {name}')
                else:
                    compare_q(source/final/'state.h5',same/final/'state.h5')
                entry['same_method'][str(exact)]=metadata
            changed={**common,'scheme':args.target_scheme,'numerics':dict(
                convection_form=args.convection_form,viscous_form=args.target_viscosity,
                difschm='643e',conschm=args.target_scheme)}
            target,_=run_case(opts,ROOT,backend,ranks,'mean_migrated',6,restore=seed,exact_restart=False,
                              run_config_text=runtime(6),restore_probe=True,
                              rhs_snapshot_step=4 if ranks==1 else None,**changed)
            # Nonexact restore retains its existing shared/periodic-node average.
            restored_field=compare_q(seed/'state.h5',target/'outdat/restore_probe.h5')
            final=target/'outdat/new/checkpoints/step000000000006'
            after=mean_metadata(final/'statistics.h5',args.statistics_path)
            count_ok=args.statistics_path=='insitu' or after['samples']==3
            valid=count_ok and after['first_step']==4 and after['first_time']==.004 and abs(after['duration']-.002)<2e-10
            entry.update(target=str(target),target_statistics=after,restored_field=restored_field,
                         status='passed' if valid else 'failed_new_statistics_window')
            report['cases'][f'{backend}_np{ranks}']=entry
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            if not valid:
                raise AssertionError('Nonexact operator switch retained old statistics history')
            if ranks==1:
                entry['independent_normalized_moments_error']=independent_new_mean(target,args.statistics_path)
            runs[backend]=final/'state.h5'
            print(backend,ranks,'new window',after,'same-method histories preserved',flush=True)
        if args.gpu:
            report['cases'][f'cpu_gpu_np{ranks}']=compare_q(runs['cpu'],runs['gpu'])
            if args.statistics_path=='insitu':
                a=formal_quantities(runs['cpu'].parent/'statistics.h5')
                b=formal_quantities(runs['gpu'].parent/'statistics.h5')
                difference=float(np.max(abs(a-b)))
                if difference>2e-10:
                    raise AssertionError(f'CPU/GPU new-window statistics differ: {difference}')
                report['cases'][f'cpu_gpu_np{ranks}_statistics_max_abs']=difference
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
