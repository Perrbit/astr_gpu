"""Control-only product isolation; no ParaView initialization or solver advance."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture
def dispatch(monkeypatch):
    spec=importlib.util.spec_from_file_location('isolated_dispatch',ROOT/'scripts/insitu/product_dispatch.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module._native=SimpleNamespace(); module._disable=Mock(return_value=0)

    class Array:
        def InsertNextValue(self,value): self.value=value
        def SetNumberOfTuples(self,count): self.value=0
        def GetValue(self,index): return self.value

    monkeypatch.setitem(sys.modules,'vtkmodules.vtkCommonCore',SimpleNamespace(vtkIntArray=Array))
    monkeypatch.setitem(sys.modules,'vtkmodules.vtkParallelCore',
        SimpleNamespace(vtkCommunicator=SimpleNamespace(MAX_OP=0)))
    return module


def controller(remote_failure=False):
    def reduce(local,total,op): total.value=int(bool(local.value or remote_failure))
    return SimpleNamespace(AllReduce=reduce)


def test_one_product_failure_never_becomes_fake_success(dispatch):
    operation=Mock(side_effect=dispatch.ProductFailure('encoding failed'))
    render=dispatch.isolate_image(operation)
    result=render('q_surface',{},2,.002,10,5)
    assert result['image']['status']=='disabled'
    render('q_surface',{},4,.004,10,5)
    assert operation.call_count==1
    assert not dispatch.due('q_surface','image')
    assert dispatch.due('q_surface','geometry') and dispatch.scene_due('velocity_slice')
    dispatch._disable.assert_called_once_with(b'q_surface.image')


def test_local_exception_becomes_collective_product_failure(dispatch):
    with pytest.raises(dispatch.ProductFailure,match='ValueError: bad geometry'):
        dispatch.collective_call(controller(),lambda: (_ for _ in ()).throw(ValueError('bad geometry')))


def test_remote_failure_disables_healthy_rank_too(dispatch):
    # The first reduction sees the remote product failure; CUDA health is clean.
    calls=0
    def reduce(local,total,op):
        nonlocal calls
        calls+=1; total.value=int(calls==1)
    with pytest.raises(dispatch.ProductFailure,match='another rank'):
        dispatch.collective_call(SimpleNamespace(AllReduce=reduce),lambda: 42)


def test_cuda_failure_is_not_swallowed(dispatch):
    dispatch._native.astr_insitu_product_device_health=lambda: 700
    @dispatch.isolate_image
    def render(*args):
        return dispatch.collective_call(controller(),lambda: (_ for _ in ()).throw(ValueError('failed')))
    with pytest.raises(RuntimeError,match='Unrecoverable CUDA'):
        render('q_surface',{},2,.002,1,1)
    assert not dispatch._disable.called


def test_process_control_signals_are_not_caught(dispatch):
    with pytest.raises(KeyboardInterrupt):
        dispatch.collective_call(controller(),lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
