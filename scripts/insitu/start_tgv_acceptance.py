"""Launch frozen IS7 host or bounded IS8 device products without validation imports."""
import argparse
import os
from pathlib import Path
import shutil
from string import Template
import subprocess


def prepare(output,library,enabled=True,processing_backend='host',postprocess_transport=None):
    if processing_backend not in ('host','device'):
        raise ValueError('Unsupported processing backend')
    if (processing_backend=='device' and postprocess_transport not in ('pinned','device-aware')) or (
            processing_backend=='host' and postprocess_transport is not None):
        raise ValueError('Device processing requires explicit independent face transport; host does not use it')
    preset=Path(__file__).resolve().parent/'presets/tgv32'
    pipeline=Path(__file__).resolve().parent/'tgv_pipeline.py'
    library=library.resolve(strict=True)
    if "'" in str(library) or "'" in str(pipeline):
        raise ValueError('Namelist paths cannot contain apostrophes')
    output.mkdir(parents=True,exist_ok=False)
    (output/'datin').mkdir()
    (output/'outdat/render').mkdir(parents=True)
    (output/'outdat/new').mkdir()
    for name in ('input.tgv','controller','input.output'):
        shutil.copyfile(preset/name,output/'datin'/name)
    content=Template((preset/'insitu.nml.in').read_text()).substitute(
        enabled='t' if enabled else 'f',library=library,pipeline=pipeline)
    if processing_backend=='device':
        content=content.replace("derivative_backend='gpu',",
            f"derivative_backend='gpu',processing_backend='device',postprocess_transport='{postprocess_transport}',")
    (output/'insitu.nml').write_text(content)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True,help='Catalyst ParaView implementation directory')
    p.add_argument('--executable',type=Path)
    p.add_argument('--mpiexec',type=Path)
    p.add_argument('--np',type=int,choices=(1,2),default=1)
    p.add_argument('--processing-backend',choices=('host','device'),default='host')
    p.add_argument('--postprocess-transport',choices=('pinned','device-aware'))
    p.add_argument('--off',action='store_true',help='Only disable the new in-situ function; keep native checkpoints')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if not args.prepare_only and (args.executable is None or args.mpiexec is None):
        p.error('Launching requires explicit --executable and --mpiexec')
    output=args.output.resolve()
    prepare(output,args.library,not args.off,args.processing_backend,args.postprocess_transport)
    if args.prepare_only:return
    env={k:v for k,v in os.environ.items() if not k.startswith('ASTR_')}
    for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):
        env.pop(key,None)
    env.update(ASTR_INSITU_CONFIG=str(output/'insitu.nml'),ASTR_INSITU_TIMING='1',
        ASTR_FORCE_MPI_TOPOLOGY=f'{args.np},1,1',ASTR_GPU_SYNC_MODE='explicit',
        ASTR_GPU_HALO_TRANSPORT='pinned',ASTR_GPU_PRECISION_MODE='fp64',ASTR_GPU_FILTER_WORKSPACE='scalar')
    if args.postprocess_transport=='device-aware':
        env.update(OMPI_MCA_pml='ucx',OMPI_MCA_coll='^hcoll,ucc,cuda',OMPI_MCA_osc='pt2pt',
            UCX_MEMTYPE_CACHE='n',UCX_CUDA_COPY_ENABLE_FABRIC='no',UCX_CUDA_COPY_DMABUF='no',
            UCX_CUDA_IPC_ENABLE_MNNVL='no',UCX_TLS='self,sm,cuda_copy,cuda_ipc')
    elif args.postprocess_transport=='pinned':
        env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,tcp',OMPI_MCA_osc='pt2pt',
            OMPI_MCA_opal_cuda_support='0',OMPI_MCA_coll_ucc_enable='0')
    command=[str(args.mpiexec.resolve(strict=True)),'--mca','coll_hcoll_enable','0',
        '-np',str(args.np),str(args.executable.resolve(strict=True)),'run','datin/input.tgv']
    with (output/'run.log').open('w') as log:
        subprocess.run(command,cwd=output,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
