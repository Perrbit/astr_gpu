"""Offline checks for the DCU batch reader/setup, not GPU validation."""
import struct

import pytest

from run_mb67_consumer_batch import archive_schedule,environment,snapshot


@pytest.mark.parametrize('precision,code',[('fp64','d'),('fp32','f')])
def test_snapshot_physical_domain_and_strict_energy(tmp_path,precision,code):
    path=tmp_path/'outdat/rhs.cpp_final.step00000004.rk01.rank00000000.bin'
    path.parent.mkdir()
    # Eight physical nodes, no halo; conserved components are separate blocks.
    values=[1.]*8+[0.]*24+[1.]*8
    path.write_bytes(struct.pack('<5i',1,1,1,0,5)+struct.pack('<40'+code,*values))
    _,metrics=snapshot(tmp_path,0,precision)
    assert metrics==dict(density_min=1.,internal_energy_min=1.)
    values[-1]=0.
    path.write_bytes(struct.pack('<5i',1,1,1,0,5)+struct.pack('<40'+code,*values))
    with pytest.raises(AssertionError,match='internal energy'):snapshot(tmp_path,0,precision)


def test_archive_schedule_excludes_only_segment_receipts(tmp_path):
    path=tmp_path/'archives.bin';payload=b'ASTROA03'+b'current clocks'
    path.write_bytes(payload+struct.pack('<6q',-1,-1,0,0,0,0))
    assert archive_schedule(path)==payload
    path.write_bytes(payload+struct.pack('<6q',4,-1,123,0,0,0))
    assert archive_schedule(path)==payload
    path.write_bytes(payload[:-1]+b'X'+struct.pack('<6q',4,-1,123,0,0,0))
    assert archive_schedule(path)!=payload


def test_rocm_transport_does_not_enable_cuda_support(monkeypatch):
    monkeypatch.setenv('OMPI_MCA_opal_cuda_support','1')
    monkeypatch.setenv('ASTR_FORCE_MPI_TOPOLOGY','4,1,1')
    monkeypatch.setenv('UCX_TLS','cuda_copy')
    monkeypatch.setenv('UCX_MODULE_DIR','/qualified/private/ucx/lib/ucx')
    monkeypatch.setenv('TMPDIR','/bounded/weiph/tmp')
    for transport in ('pinned','device'):
        values=environment(transport,[2,2,1])
        assert values['OMPI_MCA_opal_cuda_support']=='0'
        assert values['OMPI_MCA_pml']==('ucx' if transport=='device' else 'ob1')
        assert values['UCX_MODULE_DIR']=='/qualified/private/ucx/lib/ucx'
        assert values['OMPI_MCA_orte_tmpdir_base']=='/bounded/weiph/tmp'
        assert 'UCX_TLS' not in values
        assert values['ASTR_FORCE_MPI_TOPOLOGY']=='2,2,1'
        assert values['ASTR_GPU_COMPACT_TRANSPORT']==('device-aware' if transport=='device' else 'pinned')
