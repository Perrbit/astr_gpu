"""Fixture selection and ownership checks without advancing a flow solver."""

import numpy as np

from prepare_m12_local_case import selected_nodes


def test_cp6_subset_keeps_distinct_original_nodes_at_stretched_outlet():
    x = np.concatenate((np.linspace(0,1062,2249),[1081,1100]))
    ix,iy,selected = selected_nodes({'x':x[None,:]},(96,48,32))
    assert len(ix)==97 and len(iy)==49
    assert (ix[0],ix[-1])==(0,2250)
    assert (iy[0],iy[-1])==(0,260)
    assert np.all(np.diff(ix)>0) and np.all(np.diff(iy)>0)
    assert np.array_equal(selected,x[ix])


def test_original_interface_selection_is_unchanged():
    x = np.linspace(0,1100,2251)
    ix,iy,_ = selected_nodes({'x':x[None,:]})
    targets = np.linspace(x[0],x[-1],65)
    assert np.array_equal(ix,np.abs(x[:,None]-targets[None,:]).argmin(axis=0))
    assert np.array_equal(iy,np.rint(np.linspace(0,260,33)).astype(int))
