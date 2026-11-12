"""Run the shipped KEEP TGV input, rather than generating another case template."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from run_keep_periodic_gate import compare_q
from run_output_restart_validation import compare_fields

ROOT=Path(__file__).resolve().parents[2]
FINAL=Path('checkpoints/step000000000004')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--gpu',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['keep','mp5-me6'],default='keep')
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'examples/KEEP_Taylor_Green_Vortex/datin'
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    env.update(OMP_NUM_THREADS='1',ASTR_FORCE_MPI_TOPOLOGY='2,1,1',ASTR_GPU_SYNC_MODE='explicit',
               ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',
               OMPI_MCA_osc='pt2pt',OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    source_name='input.case' if args.variant=='keep' else 'input.mp5_me6.case'
    report={'status':'running','source':str(source/source_name),'variant':args.variant,
            'np':2,'topology':[2,1,1],'cases':{}}
    for scheme in (('643e','643c') if args.variant=='keep' else ('543e',)):
        states={}
        for backend in ('cpu','gpu'):
            case=out/(backend+'_'+scheme);case.mkdir();shutil.copytree(source,case/'datin')
            path=case/'datin/input.case'
            text=(case/'datin'/source_name).read_text()
            if args.variant=='keep':
                text=text.replace("conschm = '643e'",f"conschm = '{scheme}'")
            if backend=='cpu':
                text=text.replace('use_gpu = .true.','use_gpu = .false.')
            path.write_text(text)
            (case/'outdat/new').mkdir(parents=True)
            command=[str(args.mpiexec.resolve()),'--mca','coll_hcoll_enable','0','-np','2',
                     str(getattr(args,backend).resolve()),'run','datin/input.case']
            with (case/'run.log').open('w') as log:
                subprocess.run(command,cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            final=case/'outdat/new'/FINAL
            selection='CONVECTION FORM: original arithmetic-average KEEP' if args.variant=='keep' else 'ASTR_VISCOUS_FORM=me6_opti'
            if selection not in (case/'run.log').read_text():
                raise AssertionError('Example did not select its declared numerical form')
            if args.variant=='mp5-me6' and b'/mp5v1' not in (final/'control.bin').read_bytes():
                raise AssertionError('Example is missing the corrected MP5/ME6 dispatch identity')
            states[backend]=final/'state.h5'
            compare_q(states[backend],states[backend])
            resumed=out/(backend+'_'+scheme+'_resumed');resumed.mkdir()
            shutil.copytree(case/'datin',resumed/'datin')
            output=resumed/'datin/input.output'
            output.write_text(output.read_text().replace("directory = 'outdat/new'","directory = 'outdat/continued'")
                .replace("restore_directory = ''",f"restore_directory = '{case}/outdat/new/checkpoints/step000000000002'"))
            (resumed/'outdat/continued').mkdir(parents=True)
            with (resumed/'run.log').open('w') as log:
                subprocess.run(command,cwd=resumed,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            continuation=resumed/'outdat/continued'/FINAL
            compare_fields(final/'state.h5',continuation/'state.h5')
            for name in ('control.bin','archives.bin','insitu_control.bin'):
                if (final/name).read_bytes()!=(continuation/name).read_bytes():
                    raise AssertionError('Example exact restart changed '+name)
            report['cases'][backend+'_'+scheme]={'command':command,'state':str(final),
                'exact_restart':'all state datasets and three controls identical'}
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(backend,scheme,'continuous and exact 2+2 compared',flush=True)
        differences=compare_q(states['cpu'],states['gpu'])
        report['cases']['cpu_gpu_'+scheme]=differences
        print(scheme,'CPU/GPU maximum scaled:',max(v['max_scaled'] for v in differences.values()),flush=True)
    report['status']='passed'
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
