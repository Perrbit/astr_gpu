"""Native FP32 never overlays the legacy FP64 mixed-workspace selector."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_output_restart_validation import run_case, compare_fields


ROOT = Path(__file__).resolve().parents[2]
MPIEXEC = Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
def test_legacy_selectors_do_not_change_native32_state(tmp_path, backend):
    os.environ.update(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                      OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    args = SimpleNamespace(output=tmp_path,
        executable=ROOT / f'build_native_fp0_fp32_{backend}/bin/astr', mpiexec=MPIEXEC,
        case='tgv', mode='steps', restart_step=5, initial_dimension=0, legacy_statistics=False,
        statistics=False, initial_restart=False, filter_workspace='scalar', force='feedback',
        axis='x', no_samples=True, wall_samples=False)
    reference = None
    final = 'outdat/new/checkpoints/step000000000002/state.h5'
    for value in ('', 'fp64', 'mixed_workspace'):
        case, _ = run_case(args, ROOT, backend, 2, value or 'unset', 2, gpu_precision_mode=value)
        if reference is None:
            reference = case
        else:
            compare_fields(reference / final, case / final)
        if backend == 'gpu':
            log = (case / 'run.log').read_text()
            assert 'ASTR_GPU_PRECISION_MODE=fp32' in log
            assert 'ASTR_GPU_ACTIVE_MIXED_WORKSPACE=none' in log
            assert 'ASTR_GPU_MIXED_WORKSPACE_BYTES=0' in log
            assert log.count('is ignored by this native FP32 build') == int(bool(value))
