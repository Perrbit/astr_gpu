"""IM8 opt-in VTK mapper component, not a production ASTR backend."""
import json
import os
from pathlib import Path
import sqlite3

import pytest
from insitu_resource_monitor import run_monitored

ROOT=Path(__file__).resolve().parents[2]
PROBE=Path(os.environ.get('ASTR_INSITU_STANDARD_DEVICE_PROBE',
    ROOT/'build_m12_device_render/bin/insitu_standard_device_probe'))
MPI=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpirun')


def launch(directory,device,mode,memcheck=False):
    directory.mkdir()
    env=dict(os.environ,ASTR_INSITU_IM8_PIXELS=str(directory/'pixels.rgba'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    env.pop('DISPLAY',None)
    command=[str(MPI),'--mca','pml','ob1','--mca','btl','self,tcp','--mca','osc','pt2pt',
        '--mca','coll_hcoll_enable','0','--mca','coll_ucc_enable','0',
        '--mca','opal_cuda_support','0','-np','1']
    if memcheck:
        command+=['compute-sanitizer','--tool','memcheck','--error-exitcode','99',
                  '--log-file',str(directory/'memcheck.log')]
    command+=[str(PROBE),str(device),'--im8-'+mode]
    with (directory/'run.log').open('w') as stream:
        resources=run_monitored(command,directory,env,stream,directory/'resources.sampled.json',
            timeout_seconds=300)
    output=(directory/'run.log').read_text()
    assert resources['host_rss_peak_bytes']<4*1024**3
    assert all(row['peak_bytes']<2*1024**3 for row in resources['devices'].values())
    records=[json.loads(line.split('ASTR_IM8_GENERATED ',1)[1]) for line in output.splitlines()
        if line.startswith('ASTR_IM8_GENERATED ')]
    assert len(records)==1,output
    record=records[0]
    assert record['device']==device and record['frames']==(3 if mode=='failure' else 20)
    assert record['pixels_passed'] and record['geometry_host_bytes']==0
    assert record['device_additional_peak_bytes']<2*1024**3
    if memcheck:
        assert 'ERROR SUMMARY: 0 errors' in (directory/'memcheck.log').read_text()
    assert sum(p.stat().st_size for p in directory.iterdir() if p.is_file())<256*1024**2
    return record


@pytest.mark.parametrize('device',[0,1])
def test_generated_display_matches_staging(tmp_path,device):
    records={}
    for mode in ('staged','mapped'):
        records[mode]=launch(tmp_path/mode,device,mode)
    assert (tmp_path/'staged/pixels.rgba').read_bytes()==(tmp_path/'mapped/pixels.rgba').read_bytes()
    assert (tmp_path/'mapped/pixels.rgba').stat().st_size==20*64*64*4
    assert records['staged']['source_bytes']==records['mapped']['source_bytes']==360000*104
    assert records['staged']['display_staging_bytes']==14400000
    assert records['staged']['display_copy_bytes']==216000000
    assert records['mapped']['display_staging_bytes']==records['mapped']['display_copy_bytes']==0
    (tmp_path/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')


def test_generated_display_memcheck_and_exception_release(tmp_path):
    for mode in ('mapped','failure'):
        record=launch(tmp_path/mode,0,mode,memcheck=True)
        assert record['failure_released']==(mode=='failure')


def captured_transfers(path):
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        tables={row[0] for row in db.execute('select name from sqlite_master')}
        copies=db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind').fetchall() if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
        assert db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()==(38,)
        api=dict((name.split('_v')[0],count) for name,count in db.execute(
            'select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r '
            'join StringIds s on s.id=r.nameId group by s.value'))
        events=db.execute('select address,bytes,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS '
            'where memKind=2 order by start').fetchall()
        live={};peak=0
        for address,size,operation in events:
            if operation==0:
                assert address not in live
                live[address]=size
            else:
                assert operation==1 and live.pop(address)==size
            peak=max(peak,sum(live.values()))
        assert not live
        return dict(copies=copies,runtime_counts=api,live_allocation_peak_bytes=peak,
            allocation_events=len(events))


def test_generated_display_captured_transfers(tmp_path):
    name=os.environ.get('ASTR_INSITU_IM8_SQLITE')
    if not name:
        pytest.skip('Explicit Nsight Systems capture required')
    mapped=captured_transfers(Path(name))
    staged=captured_transfers(Path(os.environ['ASTR_INSITU_IM8_STAGED_SQLITE']))
    assert mapped['copies']==[]
    assert len(staged['copies'])==57
    assert {kind for kind,_ in staged['copies']}=={'CUDA_MEMCPY_KIND_DTOD'}
    assert sum(size for _,size in staged['copies'])==216000000
    for record in (mapped,staged):
        api=record['runtime_counts']
        assert api['cudaGraphicsGLRegisterBuffer']==api['cudaGraphicsUnregisterResource']==6
        assert api['cudaGraphicsMapResources']==api['cudaGraphicsUnmapResources']==57
    assert mapped['runtime_counts']['cudaEventRecord']==mapped['runtime_counts']['cudaEventSynchronize']==19
    assert mapped['runtime_counts']['cudaMalloc']==mapped['runtime_counts']['cudaFree']==2
    assert staged['runtime_counts']['cudaMalloc']==staged['runtime_counts']['cudaFree']==5
    assert staged['live_allocation_peak_bytes']-mapped['live_allocation_peak_bytes']==14400000
    (tmp_path/'transfers.json').write_text(json.dumps(dict(mapped=mapped,staged=staged),indent=2)+'\n')
