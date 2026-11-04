"""The launch-boundary converter must retain physics and fixed-step intent."""
from pathlib import Path

import pytest

from named_fixture import prepare_named_run
from convert_legacy_input import convert_controller,namelist

ROOT=Path(__file__).resolve().parents[2]


def test_launch_conversion_preserves_model_and_execution_controls(tmp_path):
    datin=tmp_path/'datin';datin.mkdir()
    text=(ROOT/'examples/GPU_Quickstart/tgv/datin/input.dat').read_text()
    (datin/'legacy').write_text(text)
    controller='f,f,t,t\n9,3,999,999,0,2\n2.d-5\n'
    (datin/'controller').write_text(controller)
    environment=dict(ASTR_GPU_PRECISION_MODE='fp64',ASTR_NSCBC_FARFIELD_MODE='incoming_only',
        ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1')
    primary=prepare_named_run(tmp_path,'datin/legacy',environment)
    assert primary=='datin/input.case'
    named=(tmp_path/primary).read_text()
    assert "gpu_precision_mode = 'fp64'" in named
    assert "nscbc_farfield_mode = 'incoming_only'" in named
    assert (datin/'input.run').read_text()==namelist(convert_controller(controller))
    assert environment==dict(ASTR_GPU_HALO_TRANSPORT='pinned',OMPI_MCA_pml='ob1')
    assert (datin/'legacy').read_text()==text
    assert (datin/'controller').read_text()==controller


def test_bad_fixture_is_not_launched_or_partly_converted(tmp_path):
    datin=tmp_path/'datin';datin.mkdir()
    (datin/'legacy').write_text('incomplete\n')
    (datin/'controller').write_text('f,f,f,f\n9,1,999,999,0,1\n1.d-3\n')
    environment={'ASTR_GPU_PRECISION_MODE':'fp64'}
    with pytest.raises(ValueError,match='incomplete positional input'):
        prepare_named_run(tmp_path,'datin/legacy',environment)
    assert not (datin/'input.case').exists()
    assert not (datin/'input.run').exists()
    assert environment=={'ASTR_GPU_PRECISION_MODE':'fp64'}
