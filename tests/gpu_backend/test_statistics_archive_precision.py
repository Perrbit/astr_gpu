"""Native packed statistics words must not be read as FP64 subnormals."""
import h5py
import numpy as np
import pytest

from run_cpp_statistics_gate import compare_statistics


def archive(path, precision, regional=-.125, value=1., payload=None):
    dtype=np.dtype('float32' if precision=='fp32' else 'float64')
    integers=np.int32 if precision=='fp32' else np.int64
    reals=np.full(34,regional,dtype=dtype)
    metadata=np.concatenate((np.array([1,1,4,0],dtype=np.int64),reals.view(integers).astype(np.int64)))
    with h5py.File(path,'w') as f:
        f['identity']=np.array([1,3,1,1,1,1,34,4],dtype=np.int64)
        f['metadata']=metadata
        f['numerical_kind_bytes']=[dtype.itemsize]
        f['partitions']=np.zeros(8,dtype=np.int64)
        f['rank_extras']=np.zeros(0,dtype=dtype)
        for component in range(1,35):
            f[f'q{component:04d}']=np.full((1,1,1),value,dtype=payload or dtype)


@pytest.mark.parametrize('precision',('fp32','fp64'))
def test_signed_regional_words_round_trip(tmp_path,precision):
    path=tmp_path/'self.h5';archive(path,precision)
    assert max(compare_statistics(path,path,1,precision).values())==0


def test_fp32_reports_actual_regional_difference(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5'
    archive(a,'fp32',regional=-.25);archive(b,'fp32',regional=-.125)
    assert compare_statistics(a,b,1,'fp32')['regional_state']==.125


def test_fp64_original_bound_is_retained(tmp_path):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5'
    archive(a,'fp64',value=1.001);archive(b,'fp64')
    with pytest.raises(AssertionError):compare_statistics(a,b,1)


def test_fp32_nonfinite_packed_regional_value_is_rejected(tmp_path):
    path=tmp_path/'nan.h5';archive(path,'fp32',regional=np.nan)
    with pytest.raises(AssertionError,match='nonfinite regional'):
        compare_statistics(path,path,1,'fp32')


def test_payload_kind_cannot_be_inferred_from_marker_alone(tmp_path):
    path=tmp_path/'wrong.h5';archive(path,'fp64',payload=np.float32)
    with pytest.raises(AssertionError,match='payload kind'):
        compare_statistics(path,path,1)
