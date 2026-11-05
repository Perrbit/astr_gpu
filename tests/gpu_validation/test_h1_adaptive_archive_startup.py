"""Cold automatic archives need a real dt; exact restore must retain its state."""
import h5py
import numpy as np
import pytest

from run_adaptive_tgv_gate import STEP
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw
from test_h1_hcw_inlet_compatible import state

pytestmark=pytest.mark.skipif(not EXE or H1_VARIANT!='inlet-compatible',
    reason='Select the approved root executable and H1 candidate')


@pytest.mark.parametrize('backend',('cpu','gpu'))
def test_automatic_initial_archive_and_exact_resume(tmp_path,backend):
    full,part,tail=[tmp_path/name for name in ('full','part','tail')]
    options=dict(automatic_cfl=.5,checkpoint_interval=2,volume_time_interval=1e-4)
    launch_hcw(full,backend,steps=4,**options)
    launch_hcw(part,backend,steps=2,**options)
    launch_hcw(tail,backend,steps=4,restore=part/'outdat/new/checkpoints/step000000000002',**options)
    np.testing.assert_array_equal(state(full,4),state(tail,4))
    first=next(STEP.finditer((full/'run.log').read_text()))
    path=next((full/'outdat/new/fields').glob('segment*/step000000000000/data.h5'))
    with h5py.File(path) as file:
        metadata=file['metadata'][...]
        assert int(metadata[1])==0
        np.testing.assert_array_equal(metadata[2:5].view(np.float64),[0.,0.,float(first[4])])
    a=list(STEP.finditer((full/'run.log').read_text()))[2:]
    b=list(STEP.finditer((tail/'run.log').read_text()))
    assert [match.groups() for match in a]==[match.groups() for match in b]
    frames=list((tail/'outdat/new/fields').glob('segment*/step*/data.h5'))
    assert frames
    for path in frames:
        with h5py.File(path) as file:
            assert float(file['metadata'][4:5].view(np.float64)[0])>0
