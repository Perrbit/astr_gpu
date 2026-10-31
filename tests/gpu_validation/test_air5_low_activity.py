"""AP6 numeric checkpoint tail; no solver run in the format tests."""
import struct

import pytest

from air5_checkpoint_contract import FILTER_IDENTITY,read_air5_contract


def payload(threshold=None,filtered=False):
    result=b'ASTRA504'+b' '*1280+b'\0'*320+b'joint'.ljust(128,b' ')
    if threshold is not None or filtered:
        result+=struct.pack('<2q',0,0)
    if filtered:
        result+=b'ASTRF001'+struct.pack('<5q',*FILTER_IDENTITY)
    if threshold is not None:
        result+=b'ASTRACT1'+struct.pack('<qd',1,threshold)
    return result


@pytest.mark.parametrize('filtered',[False,True])
def test_absent_activity_tail_means_disabled(filtered):
    control=read_air5_contract(payload(filtered=filtered))
    assert control['activity_threshold']==0
    assert control['activity_version']==1
    assert control['filter_identity']==(FILTER_IDENTITY if filtered else (0,)*5)


@pytest.mark.parametrize('filtered',[False,True])
@pytest.mark.parametrize('threshold',[1e-4,1e-3,1e-2,.1,1.])
def test_activity_tail_roundtrips_numeric_threshold(filtered,threshold):
    control=read_air5_contract(payload(threshold,filtered))
    assert control['activity_threshold']==threshold
    assert control['activity_version']==1
    assert control['filter_identity']==(FILTER_IDENTITY if filtered else (0,)*5)


@pytest.mark.parametrize('tail',[b'ASTRACT1',b'ASTRACT1'+struct.pack('<qd',2,.1),
    b'ASTRACT1'+struct.pack('<qd',1,float('nan')),
    b'ASTRACT1'+struct.pack('<qd',1,-.1)])
def test_activity_tail_rejects_incomplete_or_wrong_contract(tail):
    base=payload(.1)[:-24]
    with pytest.raises(ValueError):
        read_air5_contract(base+tail)


def test_activity_tail_is_not_allowed_to_override_itself():
    with pytest.raises(ValueError,match='duplicate'):
        read_air5_contract(payload(.1)+b'ASTRACT1'+struct.pack('<qd',1,.2))
