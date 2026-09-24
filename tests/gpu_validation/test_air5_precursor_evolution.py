import numpy as np
import pytest

from check_air5_precursor_evolution import wall_response


class ConstantTransport:
    def transport_properties(self,*args):
        return 2.,3.,4.,None


@pytest.mark.parametrize('stencil_points',[3,5,7])
def test_linear_wall_fields_have_exact_response_and_sign(stencil_points):
    y = np.linspace(0.,2.,9)[None,:,None]
    shape = (3,9,5)
    fields = {key:np.full(shape,value) for key,value in
        dict(ro=1.,p=20000.,u2=0.,u3=0.,sp001=.767,sp002=.233,
             sp003=0.,sp004=0.,sp005=0.).items()}
    fields.update(u1=np.broadcast_to(5*y,shape).copy(),
        t=np.broadcast_to(3000.-100*y,shape).copy(),
        tv=np.broadcast_to(3000.-50*y,shape).copy())
    result = wall_response(fields,[1.,2.,1.],ConstantTransport(),1.,10.,[1,3],stencil_points)
    for row in result:
        assert row['skin_friction'] == pytest.approx(.2)
        assert row['heat_flux_tr'] == pytest.approx(300.)
        assert row['heat_flux_v'] == pytest.approx(200.)
        assert row['heat_flux_total'] == pytest.approx(500.)
        assert row['tau_wall'] == pytest.approx(10.)
        assert row['first_node_distance'] == .25
        assert row['u_tau'] == pytest.approx(np.sqrt(10.))
        assert row['yplus_first'] == pytest.approx(.25*np.sqrt(10.)/2.)
    fields['u1'] *= -1.
    reverse = wall_response(fields,[1.,2.,1.],ConstantTransport(),1.,10.,[1],stencil_points)[0]
    assert reverse['tau_wall'] == pytest.approx(-10.)
    assert reverse['yplus_first'] == pytest.approx(.25*np.sqrt(10.)/2.)
    fields['u1'][:,0,:] = 1.
    with pytest.raises(ValueError):
        wall_response(fields,[1.,2.,1.],ConstantTransport(),1.,10.,[1])
