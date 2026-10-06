"""Component probes only; not a device visualization or MPI acceptance test."""
import os
import json
from pathlib import Path
import subprocess
import sqlite3
import pytest

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get("ASTR_INSITU_DEVICE_PROBE_BIN", ROOT / "build_insitu_device_probes/bin"))


def test_fp64_cuda_compiler():
    result = subprocess.run([str(BIN / "insitu_device_compiler_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CUDA compiler probe FP64 max_error=" in result.stdout


def test_device_rk45_matches_original_vtk():
    result = subprocess.run([str(BIN / "insitu_device_rk45_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU RK45 versus original VTK matched=26" in result.stdout


def test_accepted_length_reference_regression():
    original = subprocess.run([str(BIN / "insitu_streamtracer_length_probe"),
                               "--require-accepted"], capture_output=True,
                              text=True, timeout=60)
    assert original.returncode == 2, original.stdout + original.stderr
    fixed = subprocess.run([str(BIN / "insitu_streamtracer_length_fixed_probe"),
                            "--require-accepted"], capture_output=True,
                           text=True, timeout=60)
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    assert "StreamTracer accepted-length audit" in fixed.stdout


def test_device_accepted_length_and_state_resume():
    result = subprocess.run([str(BIN / "insitu_device_trace_probe")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exact_state_resume=1 ownership_resume=1" in result.stdout
    assert "subminimum_stop=1 unrelated_errors_rejected=1" in result.stdout


def test_cuda_egl_graphics_interop():
    for device in (0, 1):
        result = subprocess.run([str(BIN / 'insitu_graphics_interop_probe'), str(device)],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        records = [json.loads(line.split('ASTR_INSITU_GRAPHICS_PROBE ', 1)[1])
                   for line in result.stdout.splitlines()
                   if line.startswith('ASTR_INSITU_GRAPHICS_PROBE ')]
        assert len(records) == 1
        record = records[0]
        assert record['cuda_device'] == device and record['gl_cuda_device'] == device
        assert record['frames'] == 3 and record['registered_buffers'] == 1
        assert record['geometry_host_bytes'] == 0
        assert record['image_host_bytes'] == 3 * 64 * 64 * 4
        assert record['released'] and record['pixels_passed']


@pytest.mark.parametrize('pipeline',['arrays','conduit','direct'])
def test_standard_vtk_device_arrays_render_without_host_geometry(pipeline,record_property):
    name=os.environ.get('ASTR_INSITU_STANDARD_DEVICE_PROBE')
    if not name:
        pytest.skip('Explicitly build/select the private standard-device rendering probe')
    prefix=os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Select the Open MPI installation matching the FP64 Viskores build'
    for device in (0,1):
        result=subprocess.run([str(Path(prefix)/'bin/mpirun'),'--prefix',prefix,
            '--mca','pml','ob1','--mca','btl','self,tcp','--mca','osc','pt2pt',
            '--mca','coll_hcoll_enable','0','--mca','coll_ucc_enable','0',
            '--mca','opal_cuda_support','0','-np','1',name,str(device)]+(['--'+pipeline] if pipeline!='arrays' else []),
            capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
        records=[json.loads(line.split('ASTR_INSITU_STANDARD_DEVICE_PROBE ',1)[1])
                 for line in result.stdout.splitlines()
                 if line.startswith('ASTR_INSITU_STANDARD_DEVICE_PROBE ')]
        assert len(records)==1
        record=records[0]
        assert record['cuda_device']==record['gl_cuda_device']==device
        assert record['frames']==3 and record['geometry_host_bytes']==0
        assert record['image_host_bytes']==3*64*64*4
        assert record['host_access_refused'] and record['pixels_passed']
        assert record['projection_max_pixels']<=1.
        assert record['conduit']==(pipeline=='conduit')
        assert record['direct']==(pipeline=='direct')
        record_property(f'{pipeline}_device{device}',record)


def test_standard_device_render_transfer_and_lifecycle_trace():
    name=os.environ.get('ASTR_INSITU_STANDARD_DEVICE_TRACE')
    if not name:
        pytest.skip('Explicitly capture/export the standard VTK rendering component')
    with sqlite3.connect(Path(name).resolve().as_uri()+'?mode=ro',uri=True) as db:
        copies=db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind').fetchall()
        assert len(copies)==9,copies
        assert all(kind=='CUDA_MEMCPY_KIND_DTOD' for kind,_ in copies),copies
        assert sorted(size for _,size in copies)==[24]*6+[72]*3,copies
        api=dict((name.split('_v')[0],count) for name,count in db.execute(
            'select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r '
            'join StringIds s on s.id=r.nameId group by s.value'))
        assert api['cudaGraphicsGLRegisterBuffer']==3
        assert api['cudaGraphicsMapResources']==api['cudaGraphicsUnmapResources']==9
        assert api['cudaGraphicsUnregisterResource']==3
        gl=dict(db.execute('select s.value,count(*) from OPENGL_API g '
            'join StringIds s on s.id=g.nameId group by s.value'))
        assert gl['glDrawRangeElements']==gl['glReadPixels']==3
        assert gl['eglCreateContext']==gl['eglDestroyContext']==1
