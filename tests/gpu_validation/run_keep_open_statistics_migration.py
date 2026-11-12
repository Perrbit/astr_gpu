"""Native CPU mean44/GPU spanwise-statistics window checks on the approved KEEP open fixture."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from run_keep_statistics_migration_probe import mean_metadata
from run_output_restart_validation import run_case,compare_fields
from run_keep_periodic_gate import compare_q
from run_keep_open_gate import wall_inputs

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('outdat/new/checkpoints/step000000000006')


def metadata(path,backend):
    if backend=='cpu':
        return mean_metadata(path,'mean')
    with h5py.File(path) as state:
        m=state['metadata'][:]
    return dict(samples=int(m[7]),first_step=int(m[8]),last_step=int(m[9]),
                first_time=float(m[10:11].view(np.float64)[0]),
                last_time=float(m[12:13].view(np.float64)[0]),duration=float(m[13:14].view(np.float64)[0]))


def compare_spanwise(cpu_file,gpu_file,geometry_file):
    with h5py.File(geometry_file) as geometry:
        z=geometry['q0003'][:]
    differences=[]
    with h5py.File(cpu_file) as cpu,h5py.File(gpu_file) as gpu:
        # CPU mean44 stores p before rho*T; device plane statistics use the reverse order.
        for component,cpu_component in enumerate((1,2,3,4,6,5,7,8,9,10,11,12),start=1):
            values=cpu[f'q{cpu_component:04d}'][:]
            expected=np.sum(.5*np.diff(z,axis=0)*(values[1:]+values[:-1]),axis=0)/2e-5
            actual=gpu[f'q{component+1:04d}'][0,:,:]/2e-5
            error=float(np.max(abs(actual-expected)/np.maximum(1,np.maximum(abs(actual),abs(expected)))))
            if not np.isfinite(error) or error>2e-10:
                raise AssertionError(f'New-window spanwise integral differs in moment {component}: {error}')
            differences.append(error)
    return differences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--target-viscosity',choices=['standard','me6_opti'],default='standard')
    parser.add_argument('--source-convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--target-convection',choices=['keep','standard'],default='keep')
    parser.add_argument('--source-scheme',choices=['543e','743e','643e','643c'],default='643e')
    parser.add_argument('--target-scheme',choices=['543e','743e','643e','643c'],default='643c')
    parser.add_argument('--diffusion',choices=['643e','643c'],default='643e')
    parser.add_argument('--outlet',type=int,choices=[21,22],default=22)
    parser.add_argument('--top',type=int,choices=[50,51,52],default=52)
    parser.add_argument('--inlet-mode',choices=['prof','intp'],default='prof')
    parser.add_argument('--blowing',choices=['off','modal','counter','legacy_random'],default='off')
    parser.add_argument('--extruded',action='store_true')
    args=parser.parse_args()
    for convection,scheme in ((args.source_convection,args.source_scheme),(args.target_convection,args.target_scheme)):
        if convection=='keep' and scheme not in ('643e','643c'):
            parser.error('KEEP requires its sixth-order explicit or compact convection')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    wallbs,wall_phase=wall_inputs(out,args.blowing)
    os.environ.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    def runtime(enabled):
        return f"""&run_control max_steps=6 /
&timestep mode='fixed',fixed_dt=1.d-5 /
&logging list_steps=1 /
&repair crashfix=f /
&statistics enabled={'t' if enabled else 'f'},sample_clock='steps',sample_steps=1,write_steps=0 /
"""
    def numerics(convection,viscosity,scheme):
        selection=dict(convection_form=convection,viscous_form=viscosity,conschm=scheme,difschm=args.diffusion)
        if scheme in ('543e','743e'):
            selection.update(recon_schem=5 if scheme=='743e' else 3,lchardecomp=scheme=='743e')
        return selection
    report=dict(status='running',scope='16-cubed static profile, NP=2 x; no filter/repair',
                boundaries=[11,args.outlet,41,args.top,1,1],
                source_viscosity=args.source_viscosity,target_viscosity=args.target_viscosity,
                inlet_mode=args.inlet_mode,blowing=args.blowing,extruded=args.extruded,
                source_convection=args.source_convection,target_convection=args.target_convection,
                source_scheme=args.source_scheme,target_scheme=args.target_scheme,diffusion=args.diffusion,cases={})
    runs={}
    for backend in ('cpu','gpu'):
        opts=SimpleNamespace(output=out,executable=getattr(args,backend).resolve(),mpiexec=args.mpiexec.resolve(),
            case='dynamic' if args.inlet_mode=='intp' else 'curve',mode='steps',restart_step=4,
            initial_dimension=2 if args.extruded else 3,initial_restart=False,
            legacy_statistics=True,statistics=False,filter_workspace='scalar',force='fixed',
            axis='x',no_samples=True,wall_samples=False,unrestricted=True,inflow_count=12)
        boundary=dict(nscbc_farfield_mode='nonreflecting') if args.top==52 else {}
        if args.blowing!='off':
            boundary['wall_blowing_mode']='legacy_random' if args.blowing=='legacy_random' else 'current'
        common=dict(lfilter=False,scheme=args.source_scheme,numerics=numerics(args.source_convection,args.source_viscosity,args.source_scheme),
                    boundary_options=boundary,flatplate_warp=(0,0),
                    flatplate_bctype=(args.outlet,args.top),initial_resource=args.initial.resolve(),run_config_text=runtime(True),
                    rk_state_check=True,rk_stage_state_check=True,wallbs_resource=wallbs,wall_phase_resource=wall_phase,
                    flatplate_extruded=args.extruded)
        source,_=run_case(opts,ROOT,backend,2,'open_mean_source',6,**common)
        off,_=run_case(opts,ROOT,backend,2,'open_mean_off',6,**{**common,'run_config_text':runtime(False)})
        with h5py.File(source/FINAL/'state.h5') as a,h5py.File(off/FINAL/'state.h5') as b:
            if any(not np.array_equal(a[f'q{c:04d}'][:],b[f'q{c:04d}'][:]) for c in range(1,6)):
                raise AssertionError('Mean-statistics ON/OFF changed the conservative trajectory')
        seed=source/'outdat/new/checkpoints/step000000000004'
        entry={'source':str(source),'statistics_on_off':'five conservative arrays bit-identical'}
        for exact in (True,False):
            same,_=run_case(opts,ROOT,backend,2,'open_mean_same_'+str(exact).lower(),6,restore=seed,
                            exact_restart=exact,**common)
            values=metadata(same/FINAL/'statistics.h5',backend)
            if values['samples']!=7 or values['first_step']!=0 or abs(values['duration']-6e-5)>2e-10:
                raise AssertionError(f'Existing open-flow statistics history was reset: {backend} {values}')
            if exact:
                for name in ('state.h5','statistics.h5'):
                    compare_fields(source/FINAL/name,same/FINAL/name)
                if args.inlet_mode=='intp':
                    compare_fields(source/FINAL/'inflow.h5',same/FINAL/'inflow.h5')
                for name in ('control.bin','archives.bin','insitu_control.bin'):
                    if (source/FINAL/name).read_bytes()!=(same/FINAL/name).read_bytes():
                        raise AssertionError(f'Open-flow exact continuation changed {name}')
            entry['same_'+str(exact)]=values
        changed={**common,'scheme':args.target_scheme,'numerics':numerics(args.target_convection,args.target_viscosity,args.target_scheme)}
        case,_=run_case(opts,ROOT,backend,2,'open_mean_changed',6,restore=seed,exact_restart=False,restore_probe=True,**changed)
        restored=compare_q(seed/'state.h5',case/'outdat/restore_probe.h5')
        values=metadata(case/FINAL/'statistics.h5',backend)
        if values['samples']!=3 or values['first_step']!=4 or abs(values['first_time']-4e-5)>2e-10 or abs(values['duration']-2e-5)>2e-10:
            raise AssertionError(f'Open-flow method switch retained old history: {backend} {values}')
        entry.update(changed=values,target=str(case),restored_field=restored)
        report['cases'][backend]=entry;runs[backend]=case
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(backend,values,flush=True)
    report['cpu_gpu_fields']=compare_q(runs['cpu']/FINAL/'state.h5',runs['gpu']/FINAL/'state.h5')
    geometry_file=Path(report['cases']['cpu']['source'])/'outdat/new/resources/geometry.h5'
    report['first12_spanwise_normalized_integral_max_scaled']=compare_spanwise(
        runs['cpu']/FINAL/'statistics.h5',runs['gpu']/FINAL/'statistics.h5',geometry_file)
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
