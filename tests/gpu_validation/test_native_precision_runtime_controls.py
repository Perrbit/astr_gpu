"""Statistics sampling and deferred timing must not alter GPU advancement."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from run_compact_execution import window_times
from run_output_restart_validation import compare_fields, run_case
from compare_q_validation_snapshots import read_q_snapshot


ROOT = Path(__file__).resolve().parents[2]
FINAL = 'outdat/new/checkpoints/step000000000004/state.h5'


@pytest.fixture(scope='module', params=[(kind, scheme) for kind in ('fp32', 'fp64')
                                        for scheme in ('643e', '643c')])
def reference(request, tmp_path_factory):
    kind, scheme = request.param
    executable = Path(os.environ.get('ASTR_NATIVE_'+kind.upper()+'_CONTROL_EXE',
                                    ROOT/f'build_native_fp0_{kind}_gpu/bin/astr'))
    args = SimpleNamespace(output=tmp_path_factory.mktemp('native_controls_'+kind+'_'+scheme),
        executable=executable.resolve(), mpiexec=Path(
            '/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'),
        case='tgv', mode='steps', restart_step=3, initial_dimension=0,
        legacy_statistics=False, statistics=False, initial_restart=False,
        filter_workspace='scalar', force='feedback', axis='x', no_samples=True, wall_samples=False,
        directory_budget_bytes=256*1024**2)
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    case, _ = run_case(args, ROOT, 'gpu', 2, 'reference', 4, grid='32,32,32',
                       scheme=scheme, checkpoint_interval=1, gpu_precision_mode='')
    assert f'ASTR_GPU_PRECISION_MODE={kind}' in (case/'run.log').read_text()
    return args, case, scheme, kind


@pytest.mark.parametrize('interval,buffered', [(0, True), (2, False)])
def test_statistics_and_buffered_timing_preserve_state(reference, interval, buffered):
    args, expected, scheme, precision = reference
    case, _ = run_case(args, ROOT, 'gpu', 2, f'interval{interval}', 4, grid='32,32,32',
        scheme=scheme, checkpoint_interval=1, gpu_precision_mode='',
        gpu_statistics_interval=interval, benchmark_buffer_output=buffered,
        tgv_conservation=interval==2, rhs_snapshot_step=2 if interval==2 else None)
    compare_fields(case/FINAL, expected/FINAL)
    flows = list(case.rglob('flowstate.dat'))
    if interval == 0:
        assert not flows
    else:
        assert len(flows) == 1
        from compare_tgv_flowstate import read_flowstate
        _, values = read_flowstate(case)
        assert [row[0] for row in values] == [2]
        dtype = np.dtype(precision.replace('fp', 'float'))
        totals = []
        for rank in range(2):
            snapshot = read_q_snapshot(case / f'outdat/rhs.pre_rhs.step00000002.rk01.rank{rank:08d}.bin',
                                       precision=precision)
            im, jm, km, halo, components = snapshot.header
            storage = snapshot.values.reshape((im+2*halo+1, jm+2*halo+1, km+2*halo+1, components), order='F')
            physical = storage[halo+1:halo+im+1, halo+1:halo+jm+1, halo+1:halo+km+1]
            local = []
            for component in (0, 4):
                first = physical[..., component].ravel(order='F').reshape(-1, 256).copy()
                for stride in (128, 64, 32, 16, 8, 4, 2, 1):
                    first[:, :stride] += first[:, stride:2*stride]
                second = np.zeros(256, dtype=dtype)
                second[:first.shape[0]] = first[:, 0]
                for stride in (128, 64, 32, 16, 8, 4, 2, 1):
                    second[:stride] += second[stride:2*stride]
                local.append(second[0])
            totals.append(np.array(local, dtype=dtype))
        expected_means = (totals[0]+totals[1])/dtype.type(32**3)
        path, = case.rglob('tgv_conservation.dat')
        rows = np.loadtxt(path, skiprows=1, ndmin=2)
        assert rows[:, 0].tolist() == [0, 2]
        np.testing.assert_array_equal(rows[-1, 2:].astype(dtype), expected_means)
        assert 'ASTR_TGV_FINAL_CLOCK 4 ' in (case/'run.log').read_text()
    if buffered:
        text = (case/'run.log').read_text()
        window_times(text, 2, 4, 0)
        assert text.index('ASTR_GPU_RANK_RK_TIMING') > text.rindex('ASTR_CFL complete_step=')
