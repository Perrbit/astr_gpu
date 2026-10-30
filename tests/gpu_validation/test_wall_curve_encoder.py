"""Existing wall-curve encoder and explicit runtime chart dimensions."""
import importlib.util
import io
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]


def module():
    import sys
    sys.path.insert(0,str(ROOT/'scripts/insitu'))
    spec=importlib.util.spec_from_file_location('wall_encoder_under_test',ROOT/'scripts/insitu/sbli_products.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_explicit_size_dpi_and_no_implicit_eps():
    encoder=module()
    jpeg,eps=encoder.curve_image_bytes([0.,1.],[.5,1.],r'$p_w$',image_size=(1000,500),dpi=200,
        jpeg_quality=95,x_range=(0.,1.),y_range=(0.,2.),line_width=2.)
    assert eps is None
    with Image.open(io.BytesIO(jpeg)) as image:
        assert image.size==(1000,500) and image.info['dpi']==(200,200)


def test_masked_out_nodes_do_not_create_a_spurious_connecting_segment(monkeypatch):
    from matplotlib.axes import Axes
    original=Axes.plot;observed=[]
    def plot(axis,x,y,*args,**kwargs):
        observed.append((np.asarray(x),np.asarray(y)))
        return original(axis,x,y,*args,**kwargs)
    monkeypatch.setattr(Axes,'plot',plot)
    module().curve_image_bytes([0.,1.,3.,4.],[1.,2.,3.,4.],r'$p_w$',image_size=(800,600),dpi=300,
        jpeg_quality=95,x_range=(0.,4.),y_range=(0.,5.),break_before=[0,0,1,0])
    x,y=observed[0]
    assert np.isnan(x[2]) and np.isnan(y[2])
    np.testing.assert_array_equal(x[np.isfinite(x)],[0.,1.,3.,4.])
    np.testing.assert_array_equal(y[np.isfinite(y)],[1.,2.,3.,4.])


def test_existing_sbli_wrapper_keeps_its_curve_csv_and_requested_formats(tmp_path):
    encoder=module()
    data=np.array([[0.,1.,.01,.02],[1.,1.1,.02,.03]])
    encoder.encode_wall_curve('skin_friction_curve',data,2,.002,tmp_path,True)
    base=tmp_path/'skin_friction_curve.step00000002'
    jpeg=base.with_suffix(base.suffix+'.jpeg');eps=base.with_suffix(base.suffix+'.eps')
    with Image.open(jpeg) as image: assert image.size==(3840,1440)
    assert eps.read_bytes().startswith(b'%!PS-Adobe')
    table=np.loadtxt(base.with_suffix(base.suffix+'.csv'),delimiter=',')
    np.testing.assert_array_equal(table,data[:,[0,2]])
