"""Host-only checks for staged G2 execution without changing native defaults."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import run_air5_flow_control_gate as gate


@pytest.mark.parametrize('selected', [None, ['gpu_np1', 'gpu_np2']])
@pytest.mark.parametrize('cpu_timeout', [None, 1800, 14400])
def test_control_matrix_records_selected_backends_and_separate_cpu_budget(tmp_path, monkeypatch,
                                                                         selected, cpu_timeout):
    executable = tmp_path / 'unused'
    executable.touch()
    args = SimpleNamespace(output=tmp_path / 'matrix', executable=executable, mpiexec=executable,
        engineering_only=False, reference_only=False, references=tmp_path / 'qualified.json')
    if cpu_timeout is not None:
        args.cpu_timeout_seconds = cpu_timeout
    if selected is not None:
        args.cases = selected
    references = dict(references={str(level): dict(state=str(tmp_path / f'ref{level}.h5'))
                                 for level in (11, 12)},
                      initial=str(tmp_path / 'initial.h5'), freestream={})
    calls = []

    def run_case(options, backend, ranks, label):
        calls.append((backend, ranks, options.chemistry_tolerance, options.step_history,
                      options.runtime_timeout_seconds))
        case = args.output / f'{backend}_np{ranks}_{label}'
        case.mkdir()
        return dict(case=str(case), state=str(case / 'unused.h5'))

    monkeypatch.setattr(gate, 'load_references', lambda *args: references)
    monkeypatch.setattr(gate, 'run_case', run_case)
    monkeypatch.setattr(gate, 'same_initial', lambda *args: None)
    monkeypatch.setattr(gate, 'compare_checkpoints', lambda *args: dict(
        within_reference_margin=True, within_candidate_error=True))
    gate.run(args)
    expected = [('cpu', 1), ('gpu', 1), ('gpu', 2)] if selected is None else [('gpu', 1), ('gpu', 2)]
    expected_cpu_timeout = 14400 if cpu_timeout is None else cpu_timeout
    assert calls == [(backend, ranks, tolerance, history, expected_cpu_timeout if backend == 'cpu' else 1800)
                     for tolerance, history in gate.CONTROLS for backend, ranks in expected]
    report = json.loads((args.output / 'flow_control.json').read_text())
    assert report['selected_cases'] == (['cpu_np1', 'gpu_np1', 'gpu_np2'] if selected is None else selected)
    assert report['cpu_runtime_timeout_seconds'] == expected_cpu_timeout
    assert report['reference_runtime_timeout_seconds'] == 14400
    assert report['status'] == ('local-flow-control-error-passed-not-performance' if selected is None else
                                'local-selected-flow-control-error-passed-not-performance')


def test_failed_reference_margin_retains_component_evidence(tmp_path, monkeypatch):
    binary = tmp_path / 'unused'
    binary.touch()
    args = SimpleNamespace(output=tmp_path, mpiexec=binary, reference11_executable=binary,
        reference12_executable=binary, reference11_restore=None)

    def run_case(options, backend, ranks, label, restore=None):
        case = tmp_path / label
        batch = case / gate.FINAL
        batch.mkdir(parents=True)
        level = int(label.removeprefix('reference'))
        prefix = b'ASTRA502'+bytes(1632)+b'joint'.ljust(128, b' ')
        (batch / 'air5_config.bin').write_bytes(prefix+np.array([level, 0], dtype='<i8').tobytes())
        return dict(case=str(case), state=str(batch / 'unused.h5'))

    margin = dict(within_reference_margin=False, within_candidate_error=True,
                  reference_pair={'q': {'max_normalized_error': [1.12]}})
    monkeypatch.setattr(gate, 'run_case', run_case)
    monkeypatch.setattr(gate, 'same_initial', lambda *args: None)
    monkeypatch.setattr(gate, 'frozen_freestream', lambda *args: {'rho': .05})
    monkeypatch.setattr(gate, 'batch_identity', lambda *args: {'state.h5': 'host-only-test'})
    monkeypatch.setattr(gate, 'compare_checkpoints', lambda *args: margin)
    with pytest.raises(ValueError, match='reference margin'):
        gate.collect_references(args, None)
    report = json.loads((tmp_path / 'reference_summary.json').read_text())
    assert report['status'] == 'failed'
    assert report['margin'] == margin
    assert Path(report['initial']).name == 'state.h5'
