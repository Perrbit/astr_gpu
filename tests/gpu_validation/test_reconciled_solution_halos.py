from types import SimpleNamespace

import numpy as np
import pytest

import check_reconciled_solution_halos as checker


@pytest.mark.parametrize('numq', [5, 11])
def test_periodic_face_donor_contract(monkeypatch, tmp_path, numq):
    hm, dim = 2, 4
    rng = np.random.default_rng(12)
    q = rng.random((dim+2*hm+1,)*3+(numq,))
    q[hm:hm+dim+1, hm:hm+dim+1, :hm] = q[hm:hm+dim+1, hm:hm+dim+1, dim:dim+hm]
    q[hm:hm+dim+1, hm:hm+dim+1, hm+dim+1:] = q[hm:hm+dim+1, hm:hm+dim+1, hm+1:2*hm+1]
    layout = SimpleNamespace(i0=0, j0=0, k0=0)
    monkeypatch.setattr(checker, '_load_parallel_layout', lambda _: {0: layout})
    monkeypatch.setattr(checker, '_phase_files', lambda *args: {0: tmp_path/'q.bin'})
    monkeypatch.setattr(checker, 'read_q_snapshot', lambda _: SimpleNamespace(
        header=(dim, dim, dim, hm, numq), values=q.ravel(order='F')))
    assert checker.check(tmp_path, 0, 1)['passed']
    # A later transverse reconciliation changed a donor after its halo was sent.
    q[hm, hm, dim, -1] += .01
    report = checker.check(tmp_path, 0, 1)
    assert not report['passed']
    assert not report['faces'][0]['equal']
