"""Known two-dimensional acoustic signals, not another PDE solver."""
import numpy as np
import pytest

from analyze_keep_open_acoustic import compare_histories


def histories():
    step=np.arange(41);t=step*.01
    x,z=np.meshgrid([4.5,5.,5.5],[0.,.125,.25])
    coordinates=np.stack([x,np.full_like(x,.875),z],axis=-1)
    shape=(len(step),*x.shape)
    incident=np.zeros(len(step));returned=np.zeros(len(step))
    incident[2:11]=np.sin(np.pi*(step[2:11]-2)/8)**2
    returned[22:31]=np.sin(np.pi*(step[22:31]-22)/8)**2
    plus=np.broadcast_to(incident[:,None,None],shape)
    bulk=np.broadcast_to(.012*returned[:,None,None],shape)
    extra=np.broadcast_to(.03*returned[:,None,None],shape)

    def sample(offset,wp,wm):
        base_p=np.broadcast_to((2+offset+.02*t)[:,None,None],shape).copy()
        base_v=np.broadcast_to((offset+.1*t)[:,None,None],shape).copy()
        common=dict(steps=step,times=t,coordinates=coordinates,units={'pressure':'1','velocity_y':'1'})
        baseline=dict(**common,pressure=base_p,velocity=base_v)
        pulse=dict(**common,pressure=base_p+(wp+wm)/2,velocity=base_v+(wp-wm)/(2*(10/3)))
        return baseline,pulse

    return (*sample(0.,plus,bulk+extra),*sample(.7,plus,bulk))


def test_known_return_with_background_and_bulk_cancellation():
    metrics,_=compare_histories(*histories(),1.,10/3,(2,10),(22,30))
    assert abs(metrics['return_ratio']-.03)<2e-13
    assert metrics['incident_relative_difference']<2e-13
    assert metrics['physical_probe_area']==.25


def test_unmatched_reference_clock_is_not_interpolated():
    records=list(histories())
    records[3]=dict(records[3],times=records[3]['times']+1e-4)
    with pytest.raises(ValueError,match='times do not match'):
        compare_histories(*records,1.,10/3,(2,10),(22,30))
