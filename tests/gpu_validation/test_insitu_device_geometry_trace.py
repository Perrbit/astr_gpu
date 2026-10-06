"""Read-only attribution of a bounded geometry component's Nsight capture."""
import os
from pathlib import Path
import sqlite3
import json
import re
from collections import Counter

import pytest

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'tests/gpu_validation/out/insitu_is8_transport_20261005/geometry_ranges.sqlite'


def test_geometry_extraction_has_only_scalar_d2h():
    path=Path(os.environ.get('ASTR_INSITU_GEOMETRY_TRACE',DEFAULT))
    if not path.is_file():
        pytest.skip('Explicitly capture and export the geometry NVTX probe first')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        ranges=dict((text,(start,end)) for start,end,text in db.execute(
            "select start,end,text from NVTX_EVENTS where text like 'ASTR_IS8_%'"))
        assert set(ranges)=={'ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION','ASTR_IS8_COMPACT_GEOMETRY_READ'}
        begin,end=ranges['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION']
        copies=db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.end<=?',
            (begin,end)).fetchall()
        downloads=[size for name,size in copies if name=='CUDA_MEMCPY_KIND_DTOH']
        assert sorted(downloads)==[4,4,8],copies
        assert not any(name=='CUDA_MEMCPY_KIND_UVM_DTOH' for name,_ in copies),copies
        assert all(size<=264 for name,size in copies if name=='CUDA_MEMCPY_KIND_HTOD'),copies
        kernels=[name for name, in db.execute('select s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
            'join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=?',(begin,end))]
        for operator in ('astr_insitu::IndexPlane','astr_insitu::SelectQAndU','flying_edges::ComputePass1',
                         'flying_edges::ComputePass4','flying_edges::ComputePass5'):
            assert any(operator in name for name in kernels),operator
        begin,end=ranges['ASTR_IS8_COMPACT_GEOMETRY_READ']
        pages=db.execute('select sum(m.bytes) from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
            "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH' and m.start>=? and m.end<=?",(begin,end)).fetchone()[0]
        assert pages and pages>0,'Final compact geometry must remain counted, not called zero-D2H'


def test_device_view_has_only_bounded_metadata_readback():
    name=os.environ.get('ASTR_INSITU_DEVICE_VIEW_TRACE')
    if not name:
        pytest.skip('Capture/export the strict device-view component first')
    with sqlite3.connect(Path(name).resolve().as_uri()+'?mode=ro',uri=True) as db:
        ranges=dict((text,(start,end)) for start,end,text in db.execute(
            "select start,end,text from NVTX_EVENTS where text like 'ASTR_IS8_%'"))
        assert set(ranges)=={'ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION','ASTR_IS8_DEVICE_SURFACE_VIEW'}
        begin,end=ranges['ASTR_IS8_DEVICE_SURFACE_VIEW']
        copies=db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.end<=?',
            (begin,end)).fetchall()
        assert sorted(size for kind,size in copies if kind=='CUDA_MEMCPY_KIND_DTOH')==[8,16,72],copies
        assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
            "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
        kernels=[name for name, in db.execute('select s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
            'join StringIds s on s.id=k.demangledName where k.start>=? and k.end<=?',(begin,end))]
        for operator in ('astr_insitu::PackSurfaceDisplay','astr_insitu::PackSurfaceIndex',
                         'inspect_surface','inspect_display_indices'):
            assert any(operator in kernel for kernel in kernels),operator


def test_native_resident_q_trace():
    root = os.environ.get('ASTR_INSITU_RESIDENT_TRACE_ROOT')
    if not root:
        pytest.skip('Select the completed native Q trace matrix')
    cases = [p for p in Path(root).glob('test_*/gpu_*') if not p.parent.is_symlink()]
    assert len(cases) == 8
    traces = 0
    for case in cases:
        for path in case.glob('trace.rank*.sqlite'):
            traces += 1
            rank = int(path.name.split('rank', 1)[1].split('.', 1)[0])
            with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
                ranges = db.execute("select start,end,text from NVTX_EVENTS where text in "
                    "('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION','ASTR_IS8_RESIDENT_RENDER') order by start").fetchall()
                assert [r[2] for r in ranges] == ['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION',
                    'ASTR_IS8_RESIDENT_RENDER'] * 2
                assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                    'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                    "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
                for index, (begin, end, name) in enumerate(ranges):
                    copies = db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                        'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.end<=?',
                        (begin, end)).fetchall()
                    if name == 'ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION':
                        assert sorted(size for kind,size in copies if kind=='CUDA_MEMCPY_KIND_DTOH') == [4,4,8,8,72]
                    else:
                        record = json.loads((case / 'outdat/render' /
                            f'mesh_step{index//2+1:08d}_rank{rank}.json').read_text())
                        assert all(kind=='CUDA_MEMCPY_KIND_DTOD' for kind,_ in copies), copies
                        assert sorted(size for _,size in copies) == sorted([
                            record['local_points']*12, record['local_points']*4, record['local_cells']*12])
                api = dict((name.split('_v')[0],count) for name,count in db.execute(
                    'select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r '
                    'join StringIds s on s.id=r.nameId group by s.value'))
                assert api['cudaGraphicsGLRegisterBuffer'] == api['cudaGraphicsUnregisterResource'] == 3
                assert api['cudaGraphicsMapResources'] == api['cudaGraphicsUnmapResources'] == 6
    assert traces == 14


def test_resident_streamline_component_trace():
    root = os.environ.get('ASTR_INSITU_RESIDENT_STREAMLINE_TRACE_ROOT')
    if not root:
        pytest.skip('Select the completed resident streamline component trace')
    paths = sorted(Path(root).glob('component.rank*.sqlite'))
    assert len(paths) == 2
    for path in paths:
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            ranges = db.execute("select start,end,text from NVTX_EVENTS where text like "
                "'ASTR_IS8_%' order by start").fetchall()
            names = [r[2] for r in ranges]
            assert 'ASTR_IS8_COMPACT_TRACE_READ' not in names
            assert names.count('ASTR_IS8_TRACE_CONTROL_READ') == 2
            assert names.count('ASTR_IS8_RESIDENT_TRACE_CONCATENATE') == 1
            for begin,end,name in ranges:
                copies = db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                    'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.start>=? and m.end<=?',
                    (begin,end)).fetchall()
                downloads = [size for kind,size in copies if kind=='CUDA_MEMCPY_KIND_DTOH']
                if name=='ASTR_IS8_TRACE_CONTROL_READ':
                    assert downloads == [1024]
                elif name=='ASTR_IS8_RESIDENT_TRACE_CONCATENATE':
                    assert copies and all(kind=='CUDA_MEMCPY_KIND_DTOD' for kind,_ in copies)
                else:
                    assert all(size<=8 for size in downloads),copies
            assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()


def test_native_resident_full_transfer_ledger(record_property):
    root=os.environ.get('ASTR_INSITU_RESIDENT_FULL_TRACE_ROOT')
    if not root:
        pytest.skip('Select the completed four-case full-product native capture')
    from test_insitu_device_native_trace import copies
    cases=sorted(p for p in Path(root).glob('test_resident_full*/gpu_*')
                 if not p.parent.is_symlink())
    assert len(cases)==4
    ledger=[]
    identities=set()
    for case in cases:
        log=(case/'run.log').read_text()
        pixels=[dict(re.findall(r'(\w+)=([^ ]+)',row)) for row in
                re.findall(r'ASTR_INSITU_PIXEL_READ ([^\n]+)',log)]
        assert pixels,'The native GL readback observer must be present'
        for rank in (0,1):
            record=json.loads((case/f'outdat/render/mesh_step00000001_rank{rank}.json').read_text())
            pipeline=record['rendering_pipeline']
            mode=re.search(rf'ASTR_INSITU_DEVICE_FRAME rank={rank} step=2 transport=(\S+)',log)[1]
            identities.add((pipeline,mode))
            face=re.search(rf'ASTR_INSITU_DEVICE_FRAME rank={rank} step=2 transport=\S+ '
                          r'face_d2h_bytes=(\d+) face_h2d_bytes=(\d+) face_exchanges=(\d+)',log)
            assert face and int(face[3])==16
            with sqlite3.connect((case/f'trace.rank{rank}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
                counts=dict(db.execute("select text,count(*) from NVTX_EVENTS "
                    "where text like 'ASTR_IS8_%' group by text"))
                assert 'ASTR_IS8_COMPACT_GEOMETRY_READ' not in counts
                assert 'ASTR_IS8_COMPACT_TRACE_READ' not in counts
                assert counts['ASTR_IS8_DEVICE_SAMPLE']==counts['ASTR_IS8_DEVICE_MEAN_SUPPLY']==2
                assert counts['ASTR_IS8_RESIDENT_RENDER']==2
                assert counts['ASTR_IS8_RESIDENT_STREAMLINES']==8
                assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
                    'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
                    "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
                faces=copies(db,'ASTR_IS8_DEVICE_SAMPLE')+copies(db,'ASTR_IS8_DEVICE_MEAN_SUPPLY')
                sizes=sorted([34848]*2+[109512]*12+[52272]*2)
                if mode=='pinned':
                    for kind,column in (('CUDA_MEMCPY_KIND_DTOH',1),('CUDA_MEMCPY_KIND_HTOD',2)):
                        assert sorted(n for k,n in faces if k==kind and n>144)==sizes
                        assert int(face[column])==sum(sizes)
                    assert not any(k=='CUDA_MEMCPY_KIND_PTOP' for k,_ in faces)
                else:
                    assert int(face[1])==int(face[2])==0
                    assert all(n<=8 for k,n in faces if k=='CUDA_MEMCPY_KIND_DTOH')
                    assert all(n<=144 for k,n in faces if k=='CUDA_MEMCPY_KIND_HTOD')
                    assert sorted(n for k,n in faces if k=='CUDA_MEMCPY_KIND_PTOP')==sizes
                for name,allowed in (('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION',{4,8,72}),
                        ('ASTR_IS8_DEVICE_TRACE_AND_COMPACT',{8}),
                        ('ASTR_IS8_RESIDENT_TRACE_PACK',{4,8}),
                        ('ASTR_IS8_TRACE_CONTROL_READ',{1024,2048})):
                    downloads=[n for k,n in copies(db,name) if k=='CUDA_MEMCPY_KIND_DTOH']
                    assert downloads and set(downloads)<=allowed,(name,downloads)
                concat=copies(db,'ASTR_IS8_RESIDENT_TRACE_CONCATENATE')
                assert concat and all(k=='CUDA_MEMCPY_KIND_DTOD' for k,_ in concat)
                render=copies(db,'ASTR_IS8_RESIDENT_RENDER')
                expected=[]
                nonempty=set()
                for step in (1,2):
                    products=json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())['products']
                    for name,product in products.items():
                        if product['local_points']:
                            nonempty.add(name)
                        expected.extend((product['local_points']*12,product['local_points']*4,
                                         product['local_cells']*(8 if 'streamlines' in name else 12)))
                assert render and all(k=='CUDA_MEMCPY_KIND_DTOD' for k,_ in render)
                assert sorted(n for _,n in render)==sorted(n for n in expected if n)
                api=dict((name.split('_v')[0],count) for name,count in db.execute(
                    'select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r '
                    'join StringIds s on s.id=r.nameId group by s.value'))
                assert api['cudaGraphicsGLRegisterBuffer']==api['cudaGraphicsUnregisterResource']
                assert api['cudaGraphicsMapResources']==api['cudaGraphicsUnmapResources']==sum(bool(n) for n in expected)
                assert 3*len(nonempty)<=api['cudaGraphicsGLRegisterBuffer']<=sum(bool(n) for n in expected)
                mpi=db.execute('select s.value,count(*),sum(m.size) from MPI_P2P_EVENTS m '
                    'join StringIds s on s.id=m.textId where exists(select 1 from NVTX_EVENTS n '
                    "where n.text='ASTR_IS8_RESIDENT_RENDER' and m.start>=n.start and m.end<=n.end) "
                    'group by s.value').fetchall()
                assert mpi
                control_bytes=sum(n for k,n in copies(db,'ASTR_IS8_TRACE_CONTROL_READ')
                                  if k=='CUDA_MEMCPY_KIND_DTOH')
                kernels=[v for v, in db.execute('select s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
                    'join StringIds s on s.id=k.demangledName where exists(select 1 from NVTX_EVENTS n '
                    "where n.text like 'ASTR_IS8_%' and k.start>=n.start and k.end<=n.end)")]
                for operator in ('flying_edges::ComputePass5','astr_insitu::RK45TraceWorklet',
                        'astr_insitu::PackSurfaceDisplay','astr_insitu::MapDisplayColor',
                        'insitu_statistics_gpu_mean_vectors_kernel_'):
                    assert any(operator in k for k in kernels),operator
            reads=[p for p in pixels if int(p['rank'])==rank]
            for step in (1,2):
                for name in record['products']:
                    group=[p for p in reads if int(p['step'])==step and p['product']==name]
                    formats=Counter((int(p['format']),int(p['type'])) for p in group)
                    assert formats[(6407,5121)]==1
                    assert formats[(6408,5121)]==formats[(6402,5126)]
                    assert formats[(6408,5121)] in (0,2)
                    assert set(formats)<={(6407,5121),(6408,5121),(6402,5126)}
            for p in reads:
                assert int(p['width'])==800 and int(p['height'])==600
                assert int(p['pack_buffer'])==0 and int(p['row_length'])==0
                assert int(p['payload_bytes'])==800*600*(3 if int(p['format'])==6407 else 4)
                assert float(p['seconds'])>=0
            ledger.append({'pipeline':pipeline,'transport':mode,'rank':rank,
                'face_bytes_per_direction':sum(sizes),'face_d2h_bytes':int(face[1]),
                'trace_control_d2h_bytes':control_bytes,
                'geometry_render_d2h_bytes':0,'display_dtod_bytes':sum(expected),
                'pixel_reads':len(reads),'pixel_payload_bytes':sum(int(p['payload_bytes']) for p in reads),
                'pixel_read_seconds':sum(float(p['seconds']) for p in reads),
                'render_mpi_p2p_calls_bytes':mpi})
    assert identities=={(p,m) for p in ('standard-device','direct-device') for m in ('pinned','device-aware')}
    record_property('actual_transfer_ledger',ledger)


@pytest.mark.parametrize('pipeline',['compatible','standard-device','direct-device'])
@pytest.mark.parametrize('rank',[0,1])
def test_native_256_render_transfer_ledger(pipeline,rank,record_property):
    root=os.environ.get('ASTR_INSITU_256_TRACE_ROOT')
    if not root:
        pytest.skip('Select the completed 256-cubed two-step attribution')
    from test_insitu_device_native_trace import copies
    case=Path(root)/('trace_'+pipeline)
    with sqlite3.connect((case/f'native.{rank}.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        counts=dict(db.execute("select text,count(*) from NVTX_EVENTS where text like 'ASTR_IS8_%' group by text"))
        assert counts['ASTR_IS8_DEVICE_SAMPLE']==counts['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION']==2
        assert counts['ASTR_IS8_DEVICE_TRACE_AND_COMPACT']>0
        reads=copies(db,'ASTR_IS8_COMPACT_GEOMETRY_READ')+copies(db,'ASTR_IS8_COMPACT_TRACE_READ')
        geometry_reads=sum(n for k,n in reads if k in ('CUDA_MEMCPY_KIND_DTOH','CUDA_MEMCPY_KIND_UVM_DTOH'))
        if pipeline=='compatible':
            assert counts['ASTR_IS8_COMPACT_GEOMETRY_READ']==2 and geometry_reads>0
            record_property('compact_geometry_readback_bytes',geometry_reads)
            return
        assert 'ASTR_IS8_COMPACT_GEOMETRY_READ' not in counts and 'ASTR_IS8_COMPACT_TRACE_READ' not in counts
        assert counts['ASTR_IS8_RESIDENT_RENDER']==counts['ASTR_IS8_RESIDENT_STREAMLINES']==2
        assert not db.execute('select m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
            "where e.name='CUDA_MEMCPY_KIND_UVM_DTOH'").fetchall()
        groups=(('ASTR_IS8_DEVICE_SAMPLE',{4,8}),
            ('ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION',{4,8,72}),
            ('ASTR_IS8_DEVICE_TRACE_AND_COMPACT',{8}),
            ('ASTR_IS8_RESIDENT_TRACE_PACK',{4,8}),
            ('ASTR_IS8_TRACE_CONTROL_READ',{1024,2048}))
        ledger={}
        for name,allowed in groups:
            values=[n for k,n in copies(db,name) if k=='CUDA_MEMCPY_KIND_DTOH']
            assert set(values)<=allowed,(name,values)
            ledger[name]=sum(values)
        assert ledger['ASTR_IS8_TRACE_CONTROL_READ']>0
        expected=[]
        for step in (1,2):
            record=json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())
            assert record['geometry_host_bytes']==0
            for name,item in record['products'].items():
                assert item['color_field']=='speed' and item['color_range']==[0.,1.]
                expected.extend((item['local_points']*12,item['local_points']*4,
                    item['local_cells']*(8 if 'streamlines' in name else 12)))
        render=copies(db,'ASTR_IS8_RESIDENT_RENDER')
        assert render and all(k=='CUDA_MEMCPY_KIND_DTOD' for k,_ in render)
        assert sorted(n for _,n in render)==sorted(n for n in expected if n)
        api=dict((name.split('_v')[0],count) for name,count in db.execute(
            'select s.value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r '
            'join StringIds s on s.id=r.nameId group by s.value'))
        assert api['cudaGraphicsGLRegisterBuffer']==api['cudaGraphicsUnregisterResource']
        assert api['cudaGraphicsMapResources']==api['cudaGraphicsUnmapResources']==sum(bool(n) for n in expected)
        record_property('geometry_render_d2h_bytes',geometry_reads)
        record_property('display_dtod_bytes',sum(expected))
        record_property('bounded_control_reads',ledger)
