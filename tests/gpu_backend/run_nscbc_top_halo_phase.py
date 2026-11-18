"""Read-only diagnosis of original CPU bc52 RHS before/after its final halo refresh."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_nscbc_outlet_components import compile_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mpiexec',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name('nscbc_top_halo_phase_probe.F90')
    binary,info=compile_reference(args.cpu_build,out/'build',False,source)
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    data={};report=dict(binary=info,runs={})
    for ranks in (1,2):
        case=out/f'np{ranks}';case.mkdir()
        with (case/'run.log').open('w') as log:
            subprocess.run([str(args.mpiexec.resolve()),'--bind-to','none','-n',str(ranks),str(binary),str(case)],
                cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        fields=[];rows=[]
        for rank in range(ranks):
            with (case/f'phase.rank{rank:08d}.bin').open('rb') as f:
                header=np.fromfile(f,np.int32,5);shape=tuple(header[:3]+1)+(5,)
                arrays=[np.fromfile(f,np.float64,int(np.prod(shape))).reshape(shape,order='F') for _ in range(4)]
                cache_error=float(np.fromfile(f,np.float64,1)[0])
            if not all(np.isfinite(a).all() for a in arrays):raise AssertionError('nonfinite diagnostic state')
            fields.append((header,arrays))
            rows.append(dict(rank=rank,changed_physical_words=int(header[4]),
                physical_primitive_cache_error=cache_error,rhs_refresh_difference=float(np.max(abs(arrays[0]-arrays[1])))))
        data[ranks]=fields;report['runs'][str(ranks)]=rows
    ref=data[1][0][1];errors=dict(original_rhs=0.,refreshed_rhs=0.,physical_q=0.)
    for header,arrays in data[2]:
        start=int(header[3]);stop=start+int(header[2])+1
        for name,index in [('original_rhs',0),('refreshed_rhs',1),('physical_q',2)]:
            errors[name]=max(errors[name],float(np.max(abs(arrays[index]-ref[index][:,:,start:stop,:]))))
    report['np1_np2_z_max_abs']=errors
    report['physical_words_unchanged']=all(row['changed_physical_words']==0 for rows in report['runs'].values() for row in rows)
    shutil.copyfile(source,out/'probe_source.F90')
    (out/'diagnosis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='binary'},indent=2))


if __name__=='__main__':main()
