"""Exercise the actual shared Fortran mask helper without a CFD advance."""
import argparse
from pathlib import Path
import os
import subprocess


def main():
    parser=argparse.ArgumentParser()
    for name in ('mpifort','mpiexec','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[2]
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    source=(repo/'src/parallel.F90').read_text()
    start=source.index('  subroutine exchange_shock_mask_faces(')
    end=source.index('  end subroutine exchange_shock_mask_faces',start)
    helper=source[start:end+len('  end subroutine exchange_shock_mask_faces')]+'\n'
    (root/'exchange_shock_mask_faces.inc').write_text(helper)
    probe=repo/'tests/gpu_backend/shock_mask_faces_probe.F90'
    command=[str(args.mpifort),'-I'+str(root),str(probe),'-o',str(root/'probe')]
    env=os.environ|dict(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0')
    with (root/'build.log').open('w') as log:
        subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for ranks in (1,2):
        with (root/f'np{ranks}.log').open('w') as log:
            subprocess.run([str(args.mpiexec),'--bind-to','none','-n',str(ranks),str(root/'probe')],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print((root/f'np{ranks}.log').read_text().strip())


if __name__=='__main__':main()
