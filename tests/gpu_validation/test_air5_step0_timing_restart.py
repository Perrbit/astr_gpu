"""Exact step-0 restore/timer gate, reusing qualified 64-cubed engineering fields."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_air5_flow_control_gate import arguments, FINAL, INITIAL
from run_air5_performance_diagnosis import batch_identity, digest, window_summary
from run_output_air5_restart_validation import launch, state_sanity, ROOT
from run_output_restart_validation import compare_fields


@pytest.mark.parametrize('ranks', [1, 2])
def test_step0_restore_and_complete_timer_preserve_24_step_state(tmp_path, ranks):
    source = os.environ.get('ASTR_AIR5_ENGINEERING_SOURCE')
    if source is None:
        pytest.skip('requires an explicit immutable G2 engineering result')
    report = json.loads(Path(source).read_text())
    assert report['status'] == 'engineering-exact-field-passed-not-performance'
    row = next(row for row in report['cases'] if row['ranks'] == ranks)['baseline']
    baseline = Path(row['case'])
    assert batch_identity(baseline / FINAL) == row['batch_files_sha256']
    executable = Path(os.environ.get('ASTR_OUTPUT_RUNTIME_EXE', ROOT / 'build_gpu_probe/bin/astr'))
    assert digest(executable) == report['executable_sha256']
    seed = baseline / INITIAL
    seed_identity = batch_identity(seed)
    args = arguments(SimpleNamespace(output=tmp_path,
        mpiexec=Path(os.environ['ASTR_OUTPUT_MPIEXEC'])), executable)
    resumed, disk = launch(args, 'gpu', ranks, 'step0_timing', 24, restore=seed,
        interval=24, override=True, complete_step_timing=True, host_budget_bytes=256*1024**2,
        device_budget_bytes=256*1024**2, directory_budget_bytes=1024**3)
    exact = compare_fields(baseline / FINAL / 'state.h5', resumed / FINAL / 'state.h5')
    assert (baseline / 'flowstate.dat').read_bytes() == (resumed / 'flowstate.dat').read_bytes()
    timing = window_summary((resumed / 'run.log').read_text(), ranks, 0, 24, 4)
    assert batch_identity(seed) == seed_identity
    evidence = dict(status='exact-step0-timing-restart-passed', ranks=ranks, exact_datasets=exact,
        complete_steps=24, warmup=4, measured_steps=20, timing_start_step=0,
        timing_observation=timing, timing_is_performance_evidence=False,
        reason='CPU reference active; numerical/timer semantics only',
        executable_sha256=digest(executable), source_checkpoint_sha256=seed_identity,
        sanity=state_sanity(resumed / FINAL / 'state.h5'), directory_bytes=disk)
    (tmp_path / 'step0_timing_restart.json').write_text(json.dumps(evidence, indent=2, allow_nan=False)+'\n')
