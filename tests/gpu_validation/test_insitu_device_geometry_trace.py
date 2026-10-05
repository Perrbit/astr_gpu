"""Read-only attribution of a bounded geometry component's Nsight capture."""
import os
from pathlib import Path
import sqlite3

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
