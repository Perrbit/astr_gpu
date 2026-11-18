"""Clock analysis uses active windows, not their completed phase."""
import numpy as np

import run_cpp_adaptive_images as gate


def observations():
    time=np.r_[0.,np.cumsum(np.full(12,.001))]
    rows=np.column_stack([np.arange(13),time,np.zeros(13)])
    flags=[[['T' if 1<=step<6 else 'F'],[0 if step<3 else 1 if step<7 else 2]]
        for step in range(13)]
    return rows,flags


def test_completed_window_returns_to_normal_period(monkeypatch):
    monkeypatch.setattr(gate,'monitor',lambda case:observations())
    assert gate.expected_steps(None,gate.definitions()[2])==[0,3,5,11]


def test_event_entry_is_immediate_then_exits_dense_period(monkeypatch):
    monkeypatch.setattr(gate,'monitor',lambda case:observations())
    assert gate.expected_steps(None,gate.definitions()[0])==[0,1,3,5,11]
    assert gate.expected_steps(None,gate.definitions()[1])==[0,1,3,5]
