"""Read-only instantaneous/saved-mean TGV preview from a full GPU checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
UNLIMITED=(2**63-1)//4096


def snapshot(root):
    result={}
    for path in root.rglob('*'):
        if not path.is_file():continue
        with path.open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        result[str(path.relative_to(root))]=[path.stat().st_size,path.stat().st_mtime_ns,checksum]
    return result


def compare_m12_trajectories(text,reference,ranks):
    original=(reference/'run.log').read_text();result={}
    for rank in range(ranks):
        pattern=r'^ASTR_INSITU_M12_TRAJECTORY rank='+str(rank)+r' (.*)$'
        actual=re.findall(pattern,text,re.M);expected=re.findall(pattern,original,re.M)
        if not actual or actual!=expected:raise AssertionError(('M12 trajectory records differ',rank,len(actual),len(expected)))
        result[str(rank)]=dict(records=len(actual),text_exact=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('binary','source-case','implementation','mpiexec','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--freeze-only',action='store_true')
    p.add_argument('--ranks',type=int,choices=(1,2),default=1)
    p.add_argument('--axis',choices=('x','y','z'),default='x')
    p.add_argument('--step',type=int,default=4)
    p.add_argument('--pipeline',choices=('standard-device','direct-device'),default='standard-device')
    p.add_argument('--transport',choices=('pinned','device-aware'),default='pinned')
    p.add_argument('--memcheck',action='store_true')
    p.add_argument('--means',action='store_true')
    p.add_argument('--profile',choices=('tgv','m12'),default='tgv')
    p.add_argument('--legacy-source',action='store_true',help='explicit offline conversion of the frozen source input')
    p.add_argument('--window',type=float,nargs=2,default=(0.0005,0.0115))
    a=p.parse_args();root=a.output.resolve();root.mkdir(parents=True,exist_ok=False)
    binary=root/'astr';shutil.copy2(a.binary,binary);binary.chmod(0o555);shutil.copy2(__file__,root/'runner.py')
    source=a.source_case.resolve();source_root=source/'outdat/new';bundle=source_root/f'checkpoints/step{a.step:012d}'
    if not (bundle/'COMPLETE').is_file():raise AssertionError('existing completed source required; do not generate CFD here')
    (root/'datin').mkdir();(root/'outdat/render').mkdir(parents=True)
    if a.profile=='m12' and (a.means or a.ranks!=2 or a.axis!='x'):
        raise ValueError('existing M12 preview is instantaneous NP2-x only')
    if a.profile=='m12' or a.legacy_source:
        sys.path.insert(0,str(ROOT/'scripts/input'))
        from convert_legacy_input import convert_case,convert_controller,namelist
        groups=convert_case((source_root/'resources/input.txt').read_text())
        (root/'datin/input.case').write_text(namelist(groups))
        (root/'datin/input.run').write_text(namelist(convert_controller((source/'datin/controller').read_text())))
    else:
        shutil.copy2(source/'datin/input.case',root/'datin/input.case')
        shutil.copy2(source/'datin/input.run',root/'datin/input.run')
    (root/'datin/input.output').write_text(f"""&output
 directory='outdat/new',restore_directory='{bundle}',exact_restart=t,
 host_budget_bytes={UNLIMITED},device_budget_bytes={UNLIMITED},device_reserve_bytes=0
/
&checkpoint
 enabled=f
/
&volume
 enabled=f
/
&slices
 enabled=f
/
""")
    (root/'datin/input.insitu').write_text(f"""&insitu_run
 enabled=t,statistics={str(a.means)[0]},render=t,products='streamlines',mean_streamline_render={str(a.means)[0]},
 statistics_window={a.window[0]},{a.window[1]},
 processing_backend='device',derivative_backend='gpu',rendering_pipeline='{a.pipeline}',
 postprocess_transport='{a.transport}',output_directory='outdat/render',
 host_budget_bytes={UNLIMITED},device_budget_bytes={UNLIMITED},device_reserve_bytes=0,
 implementation_path='{a.implementation.resolve()}',pipeline_file='{ROOT}/scripts/insitu/device_render_pipeline.py',
 streamline_views_file='datin/views.nml',step_interval=1
/
""")
    (root/'datin/views.nml').write_text("""&streamline_views
 candidate_ids='native','oblique',velocity='instantaneous','instantaneous',
 seeds='line16','line16',color_field='u','u',line_width=2.,3.,
 camera_position(:,1)=3.14,3.14,12.,camera_position(:,2)=11.,9.,10.,
 focal_point(:,1)=3.14,3.14,3.14,focal_point(:,2)=3.14,3.14,3.14,
 view_up(:,1)=0.,1.,0.,view_up(:,2)=0.,0.,1.,parallel_scale=4.,6.5,
 display_bounds(:,1)=0.,6.283185307179586,0.,6.283185307179586,0.,6.283185307179586,
 display_bounds(:,2)=1.,5.,1.,5.,0.,6.283185307179586,color_range(:,1)=-1.,1.,color_range(:,2)=-1.,1.
/
""")
    candidates=['native','oblique']
    if a.means:
        candidates+=['reynolds','favre']
        path=root/'datin/views.nml'
        content=path.read_text().replace("candidate_ids='native','oblique'","candidate_ids='native','oblique','reynolds','favre'")
        content=content.replace("velocity='instantaneous','instantaneous'","velocity='instantaneous','instantaneous','reynolds','favre'")
        content=content.replace("seeds='line16','line16'","seeds='line16','line16','line16','line16'")
        content=content.replace("color_field='u','u'","color_field='u','u','u','u'")
        content=content.replace('line_width=2.,3.,','line_width=2.,3.,2.,2.,')
        content=content.replace('\n/\n',"""
 camera_position(:,3)=3.14,3.14,12.,camera_position(:,4)=3.14,3.14,12.,
 focal_point(:,3)=3.14,3.14,3.14,focal_point(:,4)=3.14,3.14,3.14,
 view_up(:,3)=0.,1.,0.,view_up(:,4)=0.,1.,0.,parallel_scale(3:4)=4.,4.,
 display_bounds(:,3)=0.,6.283185307179586,0.,6.283185307179586,0.,6.283185307179586,
 display_bounds(:,4)=0.,6.283185307179586,0.,6.283185307179586,0.,6.283185307179586,
 color_range(:,3)=-1.,1.,color_range(:,4)=-1.,1.
/
""")
        path.write_text(content)
    if a.profile=='m12':
        candidates=['layered64','layered256']
        content="&streamline_views\n candidate_ids='layered64','layered256',velocity='instantaneous','instantaneous',\n"
        content+="seeds='bl-layered64','bl-layered256',color_field='speed','speed',line_width=2.,2.,\n"
        for i in (1,2):
            content+=f'camera_position(:,{i})=180.,100.,180.,focal_point(:,{i})=55.,4.,45.,view_up(:,{i})=0.,1.,0.,\n'
            content+=f'parallel_scale({i})=55.,display_bounds(:,{i})=20.,90.,0.,8.,0.,90.,color_range(:,{i})=0.,1.1,\n'
        (root/'datin/views.nml').write_text(content+'/\n')
    (root/'input.products.jsonl').write_text('{"catalog_schema":"astr-insitu-products-v1"}\n')
    env={k:v for k,v in os.environ.items() if not k.startswith(('ASTR_','OMPI_MCA_','UCX_'))}
    topology=[1,1,1];topology['xyz'.index(a.axis)]=a.ranks
    env.update(ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_SYNC_MODE='explicit',ASTR_GPU_HALO_TRANSPORT='pinned',
        ASTR_INSITU_CONFIG='datin/input.insitu',ASTR_INSITU_TEST_PREVIEW_ISOLATION='1',ASTR_INSITU_WRITE_EPS='0',
        OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',OMP_NUM_THREADS='1')
    for key in ('DISPLAY','PYTHONPATH','CATALYST_IMPLEMENTATION_PREFER_ENV','VTK_EGL_DEVICE_INDEX'):env.pop(key,None)
    if a.transport=='device-aware':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1',UCX_MEMTYPE_CACHE='n')
    before=snapshot(source_root)
    command=[str(a.mpiexec.resolve()),'--bind-to','none','-n',str(a.ranks)]
    if a.memcheck:command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer',
        '--tool','memcheck','--error-exitcode','99','--log-file',str(root/'memcheck.%p.log')]
    command +=[str(binary),'insitu-preview','datin/input.case']
    with (root/'run.log').open('x') as log:
        completed=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    if snapshot(source_root)!=before:raise AssertionError('preview changed source file content/size/mtime')
    (root/'source_snapshot.json').write_text(json.dumps(before,indent=2)+'\n')
    if completed.returncode:raise AssertionError(('preview exit',completed.returncode,(root/'run.log').read_text()[-4000:]))
    text=(root/'run.log').read_text()
    if 'advances=0' not in text or 'ASTR_TIMESTEP complete_step=' in text:raise AssertionError('preview advanced CFD')
    if list(root.rglob('*.h5')) or list(root.rglob('flowstate.dat')) or list(root.rglob('*.eps')):
        raise AssertionError('preview wrote flow/checkpoint/EPS instead of requested JPEGs')
    output=root/'outdat/render';pictures={};receipts=[]
    for name in candidates:
        picture=output/f'preview.{name}.step{a.step:08d}.jpeg'
        pixels=np.asarray(Image.open(picture).convert('RGB'))
        if pixels.shape!=(600,800,3) or pixels.std()<1:raise AssertionError('preview raster shape/blank')
        pictures[name]=pixels
        for rank in range(a.ranks):
            row=json.loads((output/f'preview.{name}.step{a.step:08d}.rank{rank:08d}.json').read_text())
            if row['source']['checkpoint']!=str(bundle) or row['step']!=a.step or row['geometry_host_bytes']!=0:
                raise AssertionError('preview source/step/geometry provenance')
            if name in ('reynolds','favre') and row['statistics']['statistics_duration']<=0:
                raise AssertionError('preview mean has no saved coverage')
            receipts.append(row)
    for rank in range(a.ranks):
        cache=re.findall(r'ASTR_INSITU_PREVIEW_CACHE rank='+str(rank)+r' .*?integrated=(\d+)',text)
        expected=['1','1'] if a.profile=='m12' else ['1','0']+(['1','1'] if a.means else [])
        if cache!=expected:raise AssertionError(('preview integration cache',rank,cache))
        if not re.search(r'ASTR_INSITU_PREVIEW_ISOLATION rank='+str(rank)+r' flow_mismatches=0 statistics_mismatches=0(?:\s|$)',text):
            raise AssertionError(('preview state audit missing',rank))
        if 'ASTR_GPU_CPP preview_state=read-only' in text and not re.search(
            r'ASTR_INSITU_PREVIEW_ISOLATION rank='+str(rank)+r' .*?rng_mismatches=0(?:\s|$)',text):
            raise AssertionError(('preview RNG audit missing',rank))
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),source=str(bundle),
        source_unchanged=True,advances=0,ranks=a.ranks,axis=a.axis,step=a.step,profile=a.profile,
        pipeline=a.pipeline,transport=a.transport,means=a.means,
        receipts=receipts,memcheck=a.memcheck)
    if a.memcheck:
        logs=list(root.glob('memcheck.*.log'))
        if len(logs)!=a.ranks or any('ERROR SUMMARY: 0 errors' not in f.read_text() for f in logs):raise AssertionError('preview memory check')
    if a.reference:
        report['pixel_mean_absolute_difference']={name:float(np.mean(abs(pixels.astype(float)-np.asarray(
            Image.open(a.reference/f'outdat/render/preview.{name}.step{a.step:08d}.jpeg').convert('RGB')).astype(float))))
            for name,pixels in pictures.items()}
        if a.profile=='m12':report['logged_trajectories']=compare_m12_trajectories(text,a.reference,a.ranks)
    (root/('reference.json' if a.freeze_only else 'report.json')).write_text(json.dumps(report,indent=2)+'\n')
    if a.freeze_only:
        for path in root.rglob('*'):path.chmod(0o555 if path.is_dir() else 0o444)
        root.chmod(0o555)
    print(f'read-only {a.profile} preview complete; source unchanged; flow/statistics and trajectory reuse checked')


if __name__=='__main__':main()
