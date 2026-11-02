"""Non-exact five-variable CPU/GPU restart with frozen wall RNG state."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import h5py
import numpy as np

from run_native_precision_m12_restart_gate import launch
from run_native_precision_representative_cross_restart_gate import check_probe, wall_rng_packet
from run_native_precision_restart_gate import check_state


EXE = os.environ.get('ASTR_OUTPUT_RUNTIME_EXE')
MPIEXEC = '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'
pytestmark = pytest.mark.skipif(not EXE, reason='set ASTR_OUTPUT_RUNTIME_EXE to rebuilt CUDA solver')


@pytest.mark.parametrize('source_backend,target_backend', [('gpu', 'cpu'), ('cpu', 'gpu')])
@pytest.mark.parametrize('farfield_mode', ['compatibility', 'incoming_only'])
def test_nonexact_m12_cross_backend(tmp_path, source_backend, target_backend, farfield_mode):
    args = SimpleNamespace(output=tmp_path, source=Path('/home/dell/workspace/M12_C13'),
                           cpu=Path(EXE), gpu=Path(EXE), mpiexec=Path(MPIEXEC),
                           diffusion='643e', filter=False, farfield_mode=farfield_mode)
    source, _ = launch(args, source_backend, 2, 'seed', 5, exact=False)
    batch = source / 'outdat/new/checkpoints/step000000000005'
    original = (batch / 'control.bin').read_bytes()
    target, _ = launch(args, target_backend, 2, 'resumed', 12, restore=batch, exact=False, restore_probe=True)
    check_probe(batch, target, 'fp64', (False, False, True))
    _, saved = wall_rng_packet(batch / 'control.bin', 'fp64')
    _, restored = wall_rng_packet(target / 'outdat/restore_control_probe.bin', 'fp64')
    assert saved.tobytes() == restored.tobytes()
    assert (batch / 'control.bin').read_bytes() == original
    assert 'ASTR_OUTPUT non-exact backend switch' in (target / 'run.log').read_text()
    check_state(target / 'outdat/new/checkpoints/step000000000012/state.h5', 'fp64')


@pytest.mark.parametrize('farfield_mode,backflow_mach,ranks',
    [(mode, mach, 2) for mode in ('compatibility', 'incoming_only') for mach in (None, .5, 1.5)] +
    [('compatibility', None, 4)])
def test_m12_outlet_cpu_gpu_fields(tmp_path, farfield_mode, backflow_mach, ranks):
    args = SimpleNamespace(output=tmp_path, source=Path('/home/dell/workspace/M12_C13'),
                           cpu=Path(EXE), gpu=Path(EXE), mpiexec=Path(MPIEXEC),
                           diffusion='643e', filter=False, farfield_mode=farfield_mode,
                           backflow_mach=backflow_mach)
    cpu, _ = launch(args, 'cpu', ranks, 'reference', 12)
    gpu, _ = launch(args, 'gpu', ranks, 'reference', 12)
    name = 'outdat/new/checkpoints/step000000000012/state.h5'
    with h5py.File(cpu / name) as a, h5py.File(gpu / name) as b:
        for component in range(1, 6):
            key = f'q{component:04d}'
            expected, actual = a[key][...], b[key][...]
            error = np.max(np.abs(actual - expected) / (1. + np.abs(expected)))
            assert error <= 2e-10, (key, error)


def test_exact_m12_cross_backend_remains_rejected(tmp_path):
    args = SimpleNamespace(output=tmp_path, source=Path('/home/dell/workspace/M12_C13'),
                           cpu=Path(EXE), gpu=Path(EXE), mpiexec=Path(MPIEXEC),
                           diffusion='643e', filter=False)
    source, _ = launch(args, 'gpu', 2, 'seed', 5)
    with pytest.raises(RuntimeError):
        launch(args, 'cpu', 2, 'rejected', 12,
               restore=source / 'outdat/new/checkpoints/step000000000005')
    assert 'numerical/controller contract mismatch' in (tmp_path / 'cpu_np2_rejected/run.log').read_text()


def test_cpu_first_repair_diagnostic(tmp_path):
    args = SimpleNamespace(output=tmp_path, source=Path('/home/dell/workspace/M12_C13'),
                           cpu=Path(EXE), gpu=Path(EXE), mpiexec=Path(MPIEXEC),
                           diffusion='643e', filter=False, farfield_mode='compatibility',
                           diagnostic_stop=True, wall_amplitude=1000.)
    with pytest.raises(RuntimeError):
        launch(args, 'cpu', 2, 'diagnostic', 12, exact=False)
    case = tmp_path / 'cpu_np2_diagnostic'
    assert 'ASTR_CPU_FIRST_REPAIR_STOP' in (case / 'run.log').read_text()
    reports = [p.read_text() for p in case.glob('cpu_first_repair*.log') if p.stat().st_size]
    assert reports and all('global_ijk:' in text and 'full_rhs:' in text for text in reports)


def test_cpu_restore_without_field_outputs(tmp_path):
    args = SimpleNamespace(output=tmp_path, source=Path('/home/dell/workspace/M12_C13'),
                           cpu=Path(EXE), gpu=Path(EXE), mpiexec=Path(MPIEXEC),
                           diffusion='643e', filter=False, farfield_mode='compatibility')
    source, _ = launch(args, 'gpu', 2, 'seed', 5, exact=False)
    args.no_field_io = True
    args.diagnostic_stop = True
    args.restart_output = 'override'
    batch = source / 'outdat/new/checkpoints/step000000000005'
    target, _ = launch(args, 'cpu', 2, 'diagnostic', 12, restore=batch, exact=False, restore_probe=True)
    check_probe(batch, target, 'fp64', (False, False, True))
    assert not (target / 'datin/grid.h5').exists()
    assert not list((target / 'outdat/new/checkpoints').glob('step*'))
