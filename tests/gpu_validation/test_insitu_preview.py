"""Read-only preview isolation; real device rendering is an explicit opt-in."""
import ctypes
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from time import monotonic

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('streamline_preview', ROOT/'scripts/insitu/streamline_preview.py')
PREVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREVIEW)


def compact_receipt():
    return dict(step=4, time=.004, source=dict(step=4, time=.004, checkpoint='/sealed/checkpoint'),
        geometry_host_bytes=0, rendering=dict(image='published'),
        candidate=dict(id='front', velocity='instantaneous', seeds='line16', color='u',
            values=[0.,0.,10.,0.,0.,0.,0.,1.,0.,2.,-2.,2.,-2.,2.,-2.,2.,-1.,1.,2.],
            seed_sites=[[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]))


def test_seed_projection_is_not_visibility():
    report=PREVIEW.seed_projection(compact_receipt()['candidate'])
    assert report['seed_sites_total']==report['seed_sites_in_display_band']==report['seed_sites_in_viewport']==3
    assert report['projected_nearest_spacing_pixels']==[150.,150.]
    assert report['projected_coincident_pairs']==0
    assert report['seed_layers_y']==[dict(y=0.,count=2),dict(y=1.,count=1)]
    assert 'not a count' in report['interpretation']


def test_display_resources_do_not_grow_per_frame():
    from types import SimpleNamespace
    def proxy(identifier):
        return SimpleNamespace(SMProxy=SimpleNamespace(GetGlobalIDAsString=lambda:identifier))
    item=dict(legend=proxy('legend'),lut=proxy('lut'))
    for _ in range(20):
        resources=PREVIEW.retain_display_resources(item)
        item.update(legend=proxy('legend'),lut=proxy('lut'))
    assert set(resources)=={'legend','lut'}


def test_trace_endpoint_domain_guard(tmp_path):
    executable=tmp_path/'trace_endpoint'
    subprocess.run(['c++','-std=c++11','-I',str(ROOT/'src_gpu'),
        str(ROOT/'tests/gpu_validation/insitu_trace_endpoint_probe.cpp'),'-o',str(executable)],
        check=True,timeout=30)
    result=subprocess.run([str(executable)],capture_output=True,text=True,timeout=10)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('fault',[None,'missing_source','phase','missing_image','mean_coverage','bad_id','overwrite'])
def test_selected_configuration_export(tmp_path,fault):
    record=compact_receipt()
    if fault=='missing_source': record.pop('source')
    if fault=='phase': record['time']=1.
    if fault=='missing_image': record['rendering']['image']=dict(status='missing')
    if fault=='mean_coverage': record['candidate']['velocity']='favre'
    if fault=='bad_id': record['candidate']['id']='../../bad'
    receipt=tmp_path/'receipt.json'; receipt.write_text(json.dumps(record))
    destination=tmp_path/'selected.nml'
    if fault=='overwrite': destination.write_text('keep')
    if fault:
        with pytest.raises((ValueError,FileExistsError)):
            PREVIEW.export_selection(receipt,destination)
        assert not destination.exists() or destination.read_text()=='keep'
    else:
        PREVIEW.export_selection(receipt,destination)
        assert destination.read_text().count("candidate_ids=")==1
        assert "candidate_ids='front'" in destination.read_text()
        assert json.loads(destination.with_suffix('.nml.json').read_text())['source']==record['source']


@pytest.fixture(scope="module")
def preview_output(tmp_path_factory):
    library = tmp_path_factory.mktemp("preview_filesystem") / "filesystem.so"
    subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                    str(ROOT / "src/checkpoint_filesystem.c"), "-o", str(library)],
                   check=True, timeout=30)
    handle = ctypes.CDLL(str(library))
    function = handle.astr_checkpoint_preview_output
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize("fault", [None, "inside_source", "source_cwd", "nonempty", "symlink", "missing"])
def test_preview_output_is_separate_and_unused(tmp_path, monkeypatch, preview_output, fault):
    source = tmp_path / "source/checkpoints/step000000000004"
    source.mkdir(parents=True)
    (source / "COMPLETE").write_text("sealed source\n")
    output = tmp_path / "preview"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    if fault == "inside_source":
        output = source.parent.parent / "preview"
        output.mkdir()
    elif fault == "source_cwd":
        monkeypatch.chdir(source.parent.parent)
    elif fault == "nonempty":
        (output / "keep.jpeg").write_bytes(b"previous image")
    elif fault == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        output = alias
    elif fault == "missing":
        output = tmp_path / "missing"
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    status = preview_output(str(source).encode(), str(output).encode())
    assert (status == 0) == (fault is None)
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def source_snapshot(root):
    result = {}
    for path in root.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            result[str(path.relative_to(root))] = checksum, path.stat().st_mtime_ns
    return result


def prepare_native_preview(tmp_path, ranks, means, views=None):
    sources = Path(os.environ.get("ASTR_INSITU_PREVIEW_SOURCE_ROOT", ROOT /
        "tests/gpu_validation/out/insitu_is8_matched_qualified_20261006")).resolve(strict=True)
    source_case = sources / f"gpu_np{ranks}_timing_pinned"
    source_root = source_case / "outdat/new"
    bundle = source_root / "checkpoints/step000000000004"
    assert (bundle / "COMPLETE").is_file(), "An existing sealed checkpoint is required; no CFD is generated"
    executable = Path(os.environ.get("ASTR_OUTPUT_INSITU_EXE", ROOT /
        "build_insitu_device_render/bin/astr")).resolve(strict=True)
    launcher = Path(os.environ.get("ASTR_OUTPUT_MPIEXEC",
        "/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec")).absolute()
    backend = Path(os.environ.get("ASTR_OUTPUT_INSITU_BACKEND", ROOT.parent /
        "astr_dependencies/install/paraview-6.1.1-hpcx-gcc13/lib/catalyst")).resolve(strict=True)
    (tmp_path / "datin").mkdir()
    output = tmp_path / "outdat/render"
    output.mkdir(parents=True)
    shutil.copyfile(source_root / "resources/input.txt", tmp_path / "datin/input.tgv")
    shutil.copyfile(source_case / "datin/controller", tmp_path / "datin/controller")
    (tmp_path / "datin/input.output").write_text(f"""&output
 directory='outdat/new', restore_directory='{bundle}',
 host_budget_bytes=67108864, device_budget_bytes=67108864, buffer_bytes=67108864
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
    mean_option = "t" if means else "f"
    (tmp_path / "datin/input.insitu").write_text(f"""&insitu_run
 enabled=t, statistics={mean_option}, render=t, mean_streamline_render={mean_option},
 statistics_window=0.003,0.004, output_directory='outdat/render',
 derivative_backend='gpu', processing_backend='device', products='streamlines',
 rendering_pipeline='standard-device', postprocess_transport='pinned',
 schedule_mode='steps', step_interval=1, initial_frame=f, final_frame=f,
 host_budget_bytes=4294967296, device_budget_bytes=2147483648, device_reserve_bytes=1073741824,
 implementation_path='{backend}', pipeline_file='{ROOT / 'scripts/insitu/device_render_pipeline.py'}'
/
""")
    if views is not None:
        config = tmp_path / 'datin/input.insitu'
        config.write_text(config.read_text().replace("products='streamlines',",
            "products='streamlines',streamline_views_file='datin/views.nml',"))
        (tmp_path / 'datin/views.nml').write_text(views)
    env = {key: value for key, value in os.environ.items() if not key.startswith("ASTR_")}
    for key in ("DISPLAY", "PYTHONPATH", "CATALYST_IMPLEMENTATION_PREFER_ENV", "VTK_EGL_DEVICE_INDEX"):
        env.pop(key, None)
    env.update(ASTR_FORCE_MPI_TOPOLOGY=f"{ranks},1,1", ASTR_GPU_SYNC_MODE="explicit",
        ASTR_INSITU_WRITE_EPS="1",  # Historical preview regressions check both formats explicitly.
        ASTR_GPU_HALO_TRANSPORT="pinned", ASTR_GPU_PRECISION_MODE="fp64",
        ASTR_GPU_FILTER_WORKSPACE="scalar", ASTR_INSITU_CONFIG="datin/input.insitu",
        ASTR_INSITU_TEST_PREVIEW_ISOLATION="1",
        ASTR_INSITU_RESIDENT_AUDIT="1",
        ASTR_INSITU_TIMING="1", OMPI_MCA_pml="ob1", OMPI_MCA_btl="self,tcp",
        OMPI_MCA_osc="pt2pt", OMPI_MCA_opal_cuda_support="false", OMPI_MCA_coll_ucc_enable="0")
    return source_root, bundle, executable, launcher, env, output


@pytest.mark.parametrize("ranks", [1, 2])
@pytest.mark.parametrize("means", [False, True])
def test_native_readonly_preview(tmp_path, ranks, means, record_property):
    if os.environ.get("ASTR_INSITU_PREVIEW_CANDIDATE") != "1":
        pytest.skip("Requires explicit approval for the bounded PV GPU matrix")
    from PIL import Image
    import numpy as np

    source_root, bundle, executable, launcher, env, output = prepare_native_preview(tmp_path,ranks,means)
    before = source_snapshot(source_root)
    started = monotonic()
    try:
        with (tmp_path / "run.log").open("w") as log:
            result = subprocess.run([str(launcher), "-np", str(ranks), str(executable),
                "insitu-preview", "datin/input.tgv"], cwd=tmp_path, env=env,
                stdout=log, stderr=subprocess.STDOUT, timeout=300)
        assert result.returncode == 0, (tmp_path / "run.log").read_text()[-6000:]
    finally:
        assert source_snapshot(source_root) == before, "Preview changed its source bundle/resources"
    elapsed = monotonic() - started
    log = (tmp_path / "run.log").read_text()
    restored = re.search(r"ASTR_INSITU_PREVIEW restored_step=(\d+) time=\s*(\S+) advances=0", log)
    assert restored and int(restored[1]) == 4 and float(restored[2]) == 0.004, log
    assert not re.search(r"ASTR_INSITU_STAGE_TIMING (advance_inclusive|completed_window|insitu_sample_inclusive) ", log)
    assert not list(tmp_path.rglob("*.h5"))
    assert not list(output.rglob("*.vtp")) and not list(output.rglob("*.bin"))
    assert not (tmp_path / "outdat/new").exists()
    expected = {"instantaneous_streamlines", "crossing_streamlines"}
    if means:
        expected |= {"mean_reynolds_streamlines", "mean_favre_streamlines"}
    for rank in range(ranks):
        lifecycle = json.loads((output / f"lifecycle_rank{rank}.json").read_text())
        assert lifecycle["finalized"] is True and [frame[0] for frame in lifecycle["frames"]] == [4]
        receipt = json.loads((output / f"mesh_step00000004_rank{rank}.json").read_text())
        assert receipt["step"] == 4 and receipt["time"] == 0.004
        assert receipt["geometry_host_bytes"] == 0 and set(receipt["products"]) == expected
        with (output / f"resources.rank{rank:08d}.csv").open() as stream:
            next(stream)
            resources = list(csv.DictReader(stream))
        assert resources and resources[0]["stage"] == "baseline"
        assert resources[-1]["stage"] == "session_released"
        assert "statistics_exported" not in {row["stage"] for row in resources}
        for row in resources:
            assert int(row["host_increment_bytes"]) <= 4 * 1024**3
            assert int(row["device_increment_bytes"]) <= 2 * 1024**3
            assert int(row["device_free_bytes"]) >= 1024**3
    counts = re.findall(r"ASTR_INSITU_GPU_FLOW rank=\d+ frame_downloads=(\d+)", log)
    assert len(counts) == ranks and all(int(value) == 0 for value in counts)
    statistics = re.findall(r"ASTR_INSITU_GPU_STATS rank=\d+ samples=(\d+) full_output_downloads=(\d+)", log)
    assert len(statistics) == (ranks if means else 0)
    assert all(int(samples) == int(downloads) == 0 for samples, downloads in statistics)
    assert len(re.findall('flow_mismatches=0 statistics_mismatches=0',log))==ranks
    pictures = sorted(output.glob("*.jpeg"))
    assert len(pictures) == len(expected)
    for path in pictures:
        assert path.with_suffix(".eps").is_file()
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"))
            assert pixels.shape == (600, 800, 3) and np.any(pixels < 245), path
    size = sum(path.stat().st_size for path in tmp_path.rglob("*") if path.is_file())
    assert size <= 256 * 1024**2
    record_property("directory_bytes", size)
    record_property("elapsed_seconds", elapsed)
    record_property("source_bundle", str(bundle))


def candidate_views():
    candidates = [
        ('front','instantaneous','line16','u',2.),
        ('oblique','instantaneous','line16','u',3.),
        ('speed','instantaneous','line16','speed',2.),
        ('stratified','instantaneous','tgv-stratified','u',2.),
        ('reynolds','reynolds','line16','u',2.),
        ('favre','favre','line16','u',2.),
    ]
    text = '&streamline_views\n'
    for field, column in [('candidate_ids',0),('velocity',1),('seeds',2),('color_field',3)]:
        text += field+'='+','.join(repr(c[column]) for c in candidates)+',\n'
    text += 'line_width='+','.join(str(c[4]) for c in candidates)+',\n'
    for i, c in enumerate(candidates,1):
        text += f"camera_position(:,{i})="+('11.,9.,10.' if i == 2 else '3.14,3.14,12.')+',\n'
        text += f"focal_point(:,{i})=3.14,3.14,3.14, view_up(:,{i})=0.,1.,0., parallel_scale({i})=2.,\n"
        text += f"display_bounds(:,{i})=0.,6.283185307179586,0.,6.283185307179586,0.,6.283185307179586,\n"
        text += f"color_range(:,{i})="+('0.,1.' if c[3] == 'speed' else '-1.,1.')+',\n'
    return text+'/\n'


@pytest.mark.parametrize('ranks',[1,2])
def test_native_candidate_batch(tmp_path,ranks,record_property):
    if os.environ.get('ASTR_INSITU_PREVIEW_CANDIDATE') != '1':
        pytest.skip('Requires explicit approval for the bounded PV GPU matrix')
    from PIL import Image
    import numpy as np
    source_root,bundle,executable,launcher,env,output = prepare_native_preview(tmp_path,ranks,True,candidate_views())
    before=source_snapshot(source_root)
    started=monotonic()
    try:
        with (tmp_path/'run.log').open('w') as log:
            result=subprocess.run([str(launcher),'-np',str(ranks),str(executable),'insitu-preview','datin/input.tgv'],
                cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode==0,(tmp_path/'run.log').read_text()[-6000:]
    finally:
        assert source_snapshot(source_root)==before
    text=(tmp_path/'run.log').read_text()
    assert 'advances=0' in text
    assert not re.search(r'ASTR_INSITU_STAGE_TIMING (advance_inclusive|completed_window|insitu_sample_inclusive) ',text)
    assert not list(tmp_path.rglob('*.h5')) and not list(output.glob('*.vtp'))
    cache=re.findall(r'ASTR_INSITU_PREVIEW_CACHE rank=(\d+) step=4 candidate=(\w+) velocity=(\w+) seeds=(\S+) integrated=(\d+)',text)
    assert len(cache)==6*ranks
    for rank in range(ranks):
        rows=[row for row in cache if int(row[0])==rank]
        assert {row[1]:int(row[4]) for row in rows}==dict(front=1,oblique=0,speed=0,stratified=1,reynolds=1,favre=1)
        records={p.name.split('.')[1]:json.loads(p.read_text()) for p in output.glob(f'preview.*.rank{rank:08d}.json')}
        assert set(records)=={'front','oblique','speed','stratified','reynolds','favre'}
        assert len({(records[k]['rendering']['local_points'],records[k]['rendering']['local_cells'])
                    for k in ('front','oblique','speed')})==1
        assert len({records[k]['trajectory_storage_id'] for k in ('front','oblique','speed')})==1
        assert records['front']['trajectory_identity']==records['oblique']['trajectory_identity']
        assert records['front']['trajectory_identity']==records['speed']['trajectory_identity']
        assert records['front']['trajectory_identity']!=records['stratified']['trajectory_identity']
        for record in records.values():
            assert record['geometry_host_bytes']==0 and record['step']==4 and record['time']==.004
            assert record['rendering']['image']=='published'
            assert record['visible_line_count'] is None
            assert record['source']['checkpoint']==str(bundle)
            assert record['source']['ranks']==ranks
            assert record['projected_nearest_spacing_pixels'] is not None
            assert record['actual_color_range'][0]<=record['actual_color_range'][1]
            count=2*record['seed_sites_total']
            assert sum(record['termination_counts'].values())==count
            assert sum(sum(row.values()) for row in record['termination_by_direction'].values())==count
            assert sum(sum(sum(row.values()) for row in layer.values()) for layer in record['termination_by_layer'])==count
            assert record['trajectory_identity']['integration']['step_limit']==100000
        for kind in ('reynolds','favre'):
            assert records[kind]['statistics']['statistics_duration']==.001
        lifecycle=json.loads((output/f'lifecycle_rank{rank}.json').read_text())
        assert lifecycle['finalized'] and len(lifecycle['frames'])==6
        with (output/f'resources.rank{rank:08d}.csv').open() as stream:
            next(stream); resources=list(csv.DictReader(stream))
        assert resources[-1]['stage']=='session_released'
        for row in resources:
            assert int(row['host_increment_bytes'])<=4*1024**3
            assert int(row['device_increment_bytes'])<=2*1024**3
            assert int(row['device_free_bytes'])>=1024**3
    assert len(re.findall('samples=0 full_output_downloads=0',text))==ranks
    assert len(re.findall('frame_downloads=0',text))==ranks
    assert len(re.findall('flow_mismatches=0 statistics_mismatches=0',text))==ranks
    pictures=sorted(output.glob('preview.*.jpeg'))
    assert len(pictures)==6
    for picture in pictures:
        assert picture.with_suffix('.eps').is_file()
        with Image.open(picture) as image:
            pixels=np.asarray(image.convert('RGB'))
            assert pixels.shape==(600,800,3) and np.any(pixels<245)
    size=sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())
    assert size<=256*1024**2
    record_property('elapsed_seconds',monotonic()-started)
    record_property('directory_bytes',size)


@pytest.mark.parametrize('ranks',[1,2])
def test_native_selected_candidate(tmp_path,ranks,record_property):
    if os.environ.get('ASTR_INSITU_PREVIEW_CANDIDATE')!='1':
        pytest.skip('Requires explicit PV GPU approval')
    from PIL import Image
    import numpy as np
    outputs=[]
    for phase in ('batch','selected'):
        case=tmp_path/phase;case.mkdir()
        views=candidate_views() if phase=='batch' else selected.read_text()
        source,bundle,exe,launcher,env,output=prepare_native_preview(case,ranks,True,views)
        before=source_snapshot(source)
        with (case/'run.log').open('w') as log:
            result=subprocess.run([str(launcher),'-np',str(ranks),str(exe),'insitu-preview','datin/input.tgv'],
                cwd=case,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert source_snapshot(source)==before
        assert result.returncode==0,(case/'run.log').read_text()[-6000:]
        outputs.append(output)
        if phase=='batch':
            receipt=output/'preview.oblique.step00000004.rank00000000.json'
            selected=PREVIEW.export_selection(receipt,tmp_path/'selected.nml')
    for rank in range(ranks):
        records=[json.loads((output/f'preview.oblique.step00000004.rank{rank:08d}.json').read_text()) for output in outputs]
        for field in ('candidate','trajectory_identity','statistics','actual_color_range','termination_counts'):
            assert records[0].get(field)==records[1].get(field)
    pictures=[output/'preview.oblique.step00000004.jpeg' for output in outputs]
    with Image.open(pictures[0]) as a,Image.open(pictures[1]) as b:
        assert np.array_equal(np.asarray(a.convert('RGB')),np.asarray(b.convert('RGB')))
    assert len(list(outputs[1].glob('*.jpeg')))==1
    assert sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())<=256*1024**2
    record_property('selected_configuration',str(selected))


@pytest.mark.parametrize('fault',['coverage','frozen_input','topology','device_budget','absent_bundle'])
def test_native_preview_rejection(tmp_path,fault):
    if os.environ.get('ASTR_INSITU_PREVIEW_CANDIDATE')!='1':
        pytest.skip('Requires explicit PV GPU approval')
    source,bundle,exe,launcher,env,output=prepare_native_preview(tmp_path,2,True,candidate_views())
    config=tmp_path/'datin/input.insitu'; native=tmp_path/'datin/input.output'
    if fault=='coverage': config.write_text(config.read_text().replace('0.003,0.004','0.002,0.004'))
    if fault=='frozen_input':
        frozen=tmp_path/'datin/input.tgv'; frozen.write_bytes(frozen.read_bytes()+b'\n')
    if fault=='topology': env['ASTR_FORCE_MPI_TOPOLOGY']='1,2,1'
    if fault=='device_budget': config.write_text(config.read_text().replace('device_budget_bytes=2147483648','device_budget_bytes=1024'))
    if fault=='absent_bundle': native.write_text(native.read_text().replace(str(bundle),str(bundle.parent/'absent')))
    before=source_snapshot(source)
    with (tmp_path/'run.log').open('w') as log:
        result=subprocess.run([str(launcher),'-np','2',str(exe),'insitu-preview','datin/input.tgv'],cwd=tmp_path,
            env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert source_snapshot(source)==before
    assert result.returncode!=0
    assert not list(output.glob('*.jpeg')) and not list(tmp_path.rglob('*.h5'))
    assert 'ASTR_INSITU_STAGE_TIMING advance_inclusive ' not in (tmp_path/'run.log').read_text()


def test_native_preview_memcheck(tmp_path):
    if os.environ.get('ASTR_INSITU_PREVIEW_CANDIDATE')!='1':
        pytest.skip('Requires explicit PV GPU approval')
    source,bundle,exe,launcher,env,output=prepare_native_preview(tmp_path,2,True,candidate_views())
    sanitizer=shutil.which('compute-sanitizer')
    assert sanitizer,'Compute Sanitizer is required for the PV memory gate'
    before=source_snapshot(source)
    with (tmp_path/'run.log').open('w') as log:
        result=subprocess.run([str(launcher),'-np','2',sanitizer,'--tool','memcheck','--target-processes','all',
            '--error-exitcode','99','--log-file',str(tmp_path/'memcheck.%p.log'),
            str(exe),'insitu-preview','datin/input.tgv'],cwd=tmp_path,env=env,
            stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert source_snapshot(source)==before
    assert result.returncode==0,(tmp_path/'run.log').read_text()[-6000:]
    reports=list(tmp_path.glob('memcheck.*.log'))
    assert len(reports)==2
    for report in reports:
        assert 'ERROR SUMMARY: 0 errors' in report.read_text(),report.read_text()[-6000:]


@pytest.mark.parametrize('color',['v','w'])
def test_native_preview_vector_colors(tmp_path,color):
    if os.environ.get('ASTR_INSITU_PREVIEW_CANDIDATE')!='1':
        pytest.skip('Requires explicit PV GPU approval')
    record=compact_receipt()
    record['candidate'].update(id=color,color=color)
    record['candidate']['values']=[3.14,3.14,12.,3.14,3.14,3.14,0.,1.,0.,2.,
        0.,2.*3.141592653589793,0.,2.*3.141592653589793,0.,2.*3.141592653589793,-1.,1.,2.]
    source,bundle,exe,launcher,env,output=prepare_native_preview(tmp_path,2,False,PREVIEW.selected_namelist(record))
    before=source_snapshot(source)
    with (tmp_path/'run.log').open('w') as log:
        result=subprocess.run([str(launcher),'-np','2',str(exe),'insitu-preview','datin/input.tgv'],
            cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert source_snapshot(source)==before
    assert result.returncode==0,(tmp_path/'run.log').read_text()[-6000:]
    for rank in (0,1):
        receipt=json.loads((output/f'preview.{color}.step00000004.rank{rank:08d}.json').read_text())
        assert receipt['rendering']['color_field']==color
        assert all(math.isfinite(x) for x in receipt['actual_color_range'])
    rows=re.findall(r'ASTR_INSITU_RESIDENT_AUDIT rank=\d+ step=4 product=instantaneous_streamlines field_maxabs=(\S+) display_maxabs=(\S+)',
        (tmp_path/'run.log').read_text())
    assert len(rows)==2 and all(float(field)<=2e-10 and float(display)==0. for field,display in rows)


def test_m12_native_preview(tmp_path,record_property):
    if os.environ.get('ASTR_INSITU_PREVIEW_M12')!='1':
        pytest.skip('Requires explicit M12 read-only representative-source approval')
    from PIL import Image
    import h5py
    import numpy as np
    source,bundle,exe,launcher,env,output=prepare_native_preview(tmp_path,2,False)
    env.pop('ASTR_INSITU_RESIDENT_AUDIT',None)
    source_case=ROOT/'tests/gpu_validation/out/m12_extrapolation_mp_positive_10000_20261010/run_0_10000'
    source=source_case/'outdat/new';bundle=source/'checkpoints/step000000010000'
    assert (bundle/'COMPLETE').is_file(),'Do not generate a new CFD source'
    # The frozen case also guards exact isolation of unused NaN halo storage.
    with h5py.File(bundle/'state.h5','r') as state:
        extra_nan_count=sum(int(np.isnan(state['rank_extras'][i:i+262144]).sum())
            for i in range(0,state['rank_extras'].size,262144))
        assert extra_nan_count>0
        for component in range(1,12):
            field=state[f'q{component:04d}']
            for k in range(0,field.shape[0],8):
                values=field[k:k+8,:,:]
                assert np.isfinite(values).all()
                if component in (1,6,11):
                    assert (values>0.).all()
    record_property('source_extra_nan_count',extra_nan_count)
    shutil.copyfile(source/'resources/input.txt',tmp_path/'datin/input.tgv')
    shutil.copyfile(source_case/'datin/controller',tmp_path/'datin/controller')
    native=tmp_path/'datin/input.output'
    text=native.read_text()
    text=re.sub(r"restore_directory='[^']+'",f"restore_directory='{bundle}'",text)
    text=text.replace('host_budget_bytes=67108864','host_budget_bytes=1073741824')
    native.write_text(text)
    config=tmp_path/'datin/input.insitu'
    # Approved M12-only exception; keep the common TGV and production budgets.
    config.write_text(config.read_text()
        .replace("products='streamlines',", "products='streamlines',streamline_views_file='datin/views.nml',")
        .replace('device_budget_bytes=2147483648','device_budget_bytes=3221225472'))
    text="&streamline_views\n candidate_ids='layered64','layered256',velocity='instantaneous','instantaneous',\n"
    text+=" seeds='bl-layered64','bl-layered256',color_field='speed','speed',line_width=2.,2.,\n"
    for i in (1,2):
        text+=f'camera_position(:,{i})=180.,100.,180.,focal_point(:,{i})=55.,4.,45.,view_up(:,{i})=0.,1.,0.,\n'
        text+=f'parallel_scale({i})=55.,display_bounds(:,{i})=20.,90.,0.,8.,0.,90.,color_range(:,{i})=0.,1.1,\n'
    (tmp_path/'datin/views.nml').write_text(text+'/\n')
    before=source_snapshot(source); started=monotonic()
    with (tmp_path/'run.log').open('w') as log:
        result=subprocess.run([str(launcher),'-np','2',str(exe),'insitu-preview','datin/input.tgv'],
            cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    assert source_snapshot(source)==before
    assert result.returncode==0,(tmp_path/'run.log').read_text()[-8000:]
    log=(tmp_path/'run.log').read_text()
    assert 'restored_step=10000' in log and 'advances=0' in log
    assert len(re.findall('flow_mismatches=0 statistics_mismatches=0',log))==2
    assert len(re.findall('frame_downloads=0',log))==2
    assert not list(tmp_path.rglob('*.h5')) and not list(output.glob('*.vtp'))
    assert len(list(output.glob('*.jpeg')))==len(list(output.glob('*.eps')))==2
    for picture in output.glob('*.jpeg'):
        with Image.open(picture) as image:
            pixels=np.asarray(image.convert('RGB'))
            assert pixels.shape==(600,800,3) and np.any(pixels<245),picture
    for rank in (0,1):
        lifecycle=json.loads((output/f'lifecycle_rank{rank}.json').read_text())
        assert lifecycle['finalized'] and len(lifecycle['frames'])==2
        records=[json.loads(p.read_text()) for p in output.glob(f'preview.*.rank{rank:08d}.json')]
        assert len(records)==2
        for record in records:
            assert record['source']['checkpoint']==str(bundle)
            assert record['step']==10000 and abs(record['time']-200.)<1e-9
            assert record['geometry_host_bytes']==0 and record['trajectory_identity']['integration']['step_limit']==4096
            assert record['trajectory_identity']['integration']['initial_step']==.5
            assert sum(record['termination_counts'].values())==2*record['seed_sites_total']
        with (output/f'resources.rank{rank:08d}.csv').open() as stream:
            next(stream);rows=list(csv.DictReader(stream))
        assert rows[-1]['stage']=='session_released'
        for row in rows:
            assert int(row['host_increment_bytes'])<=4*1024**3
            assert int(row['device_increment_bytes'])<=3*1024**3
            assert int(row['device_free_bytes'])>=1024**3
    assert sum(p.stat().st_size for p in tmp_path.rglob('*') if p.is_file())<=256*1024**2
    record_property('elapsed_seconds',monotonic()-started)
    record_property('source_checkpoint',str(bundle))
