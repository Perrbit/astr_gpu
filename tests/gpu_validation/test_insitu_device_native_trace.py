"""Read-only per-object attribution of the actual solver's IS8 captures."""
import os
from pathlib import Path
import sqlite3

import pytest

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'tests/gpu_validation/out/insitu_is8_native_trace_20261006'


def copies(db,name):
    return db.execute('select e.name,m.bytes from CUPTI_ACTIVITY_KIND_MEMCPY m '
        'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind '
        'where exists(select 1 from NVTX_EVENTS n where n.text=? and m.start>=n.start and m.end<=n.end)',
        (name,)).fetchall()


@pytest.mark.parametrize('mode',['pinned','device-aware'])
@pytest.mark.parametrize('rank',[0,1])
def test_native_device_transfer_attribution(mode,rank,record_property):
    root=Path(os.environ.get('ASTR_IS8_NATIVE_TRACE',DEFAULT))
    path=root/f'gpu_np2_trace_{mode}/trace.rank{rank}.sqlite'
    if not path.is_file():
        pytest.skip('Capture the explicit two-backend native trace first')
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as db:
        counts=dict(db.execute("select text,count(*) from NVTX_EVENTS where text like 'ASTR_IS8_%' group by text"))
        assert counts['ASTR_IS8_DEVICE_SAMPLE']==counts['ASTR_IS8_DEVICE_MEAN_SUPPLY']==2
        assert counts['ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION']==counts['ASTR_IS8_COMPACT_GEOMETRY_READ']==2
        assert counts['ASTR_IS8_DEVICE_INSTANT_LINES']==counts['ASTR_IS8_DEVICE_CROSSING_LINES']==2
        assert counts['ASTR_IS8_DEVICE_REYNOLDS_LINES']==counts['ASTR_IS8_DEVICE_FAVRE_LINES']==1
        assert counts['ASTR_IS8_DEVICE_TRACE_AND_COMPACT']==counts['ASTR_IS8_COMPACT_TRACE_READ']>0
        sample=copies(db,'ASTR_IS8_DEVICE_SAMPLE')
        means=copies(db,'ASTR_IS8_DEVICE_MEAN_SUPPLY')
        for group in (sample,means):
            assert not any(kind=='CUDA_MEMCPY_KIND_UVM_DTOH' for kind,_ in group)
        face_sizes=[34848]*2+[109512]*8+[52272]
        combined=sample+means
        if mode=='pinned':
            for kind in ('CUDA_MEMCPY_KIND_DTOH','CUDA_MEMCPY_KIND_HTOD'):
                assert sorted(n for k,n in combined if k==kind and n>144)==sorted(face_sizes)
            assert not any(k=='CUDA_MEMCPY_KIND_PTOP' for k,_ in combined)
        else:
            assert all(n<=8 for k,n in combined if k=='CUDA_MEMCPY_KIND_DTOH')
            assert all(n<=144 for k,n in combined if k=='CUDA_MEMCPY_KIND_HTOD')
            assert sorted(n for k,n in combined if k=='CUDA_MEMCPY_KIND_PTOP')==sorted(face_sizes)
        extraction=copies(db,'ASTR_IS8_DEVICE_GEOMETRY_EXTRACTION')
        assert sorted(n for k,n in extraction if k=='CUDA_MEMCPY_KIND_DTOH')==[4,4,4,4,8,8]
        assert not any(k=='CUDA_MEMCPY_KIND_UVM_DTOH' for k,_ in extraction)
        trace=copies(db,'ASTR_IS8_DEVICE_TRACE_AND_COMPACT')
        assert sorted(n for k,n in trace if k=='CUDA_MEMCPY_KIND_DTOH')==[8]*counts['ASTR_IS8_DEVICE_TRACE_AND_COMPACT']
        assert not any(k=='CUDA_MEMCPY_KIND_UVM_DTOH' for k,_ in trace)
        assert all(n in (1024,2048) for k,n in trace if k=='CUDA_MEMCPY_KIND_HTOD')
        # Managed first-touch/workspace H2D remains counted, not called tiny metadata.
        record_property('trace_managed_h2d_bytes',sum(n for k,n in trace if k=='CUDA_MEMCPY_KIND_UVM_HTOD'))
        allowed=copies(db,'ASTR_IS8_COMPACT_GEOMETRY_READ')+copies(db,'ASTR_IS8_COMPACT_TRACE_READ')
        managed=sum(n for k,n in allowed if k=='CUDA_MEMCPY_KIND_UVM_DTOH')
        total=db.execute('select coalesce(sum(m.bytes),0) from CUPTI_ACTIVITY_KIND_MEMCPY m '
            'join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where e.name=\'CUDA_MEMCPY_KIND_UVM_DTOH\'').fetchone()[0]
        assert managed==total and managed>0
        record_property('final_compact_managed_d2h_bytes',managed)
        record_property('face_payload_bytes_per_direction',sum(face_sizes))
        kernels=[s for s, in db.execute('select s.value from CUPTI_ACTIVITY_KIND_KERNEL k '
            'join StringIds s on s.id=k.demangledName where exists(select 1 from NVTX_EVENTS n '
            "where n.text like 'ASTR_IS8_%' and k.start>=n.start and k.end<=n.end)")]
        for operator in ('insitu_device_velocity_derive_velocity_kernel_',
                'insitu_statistics_gpu_mean_vectors_kernel_','astr_insitu::IndexPlane','astr_insitu::SelectQAndU',
                'flying_edges::ComputePass1','flying_edges::ComputePass4','flying_edges::ComputePass5',
                'astr_insitu::RK45TraceWorklet','astr_insitu::SampleTraceVelocity'):
            assert any(operator in value for value in kernels),operator
        assert db.execute('select count(*) from MPI_P2P_EVENTS').fetchone()[0]>0
