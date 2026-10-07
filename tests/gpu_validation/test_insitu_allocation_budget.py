"""Overflow-safe admission arithmetic, without CUDA or a physical run."""
from pathlib import Path
import subprocess
import os
import re
import pytest


def test_admission_arithmetic(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / 'probe.cpp'
    source.write_text(r'''
#include "insitu_allocation_budget.h"
#include <cassert>
#include <limits>
int main() {
  using astr_insitu::admit_device_allocation;
  assert(admit_device_allocation(200, 900, 1000, 300, 200));
  assert(!admit_device_allocation(201, 900, 1000, 300, 200));
  assert(!admit_device_allocation(1, 699, 1000, 300, 200));
  assert(!admit_device_allocation(1, 199, 1000, 2000, 200));
  assert(admit_device_allocation(300, 1200, 1000, 300, 200));
  auto max=std::numeric_limits<std::uint64_t>::max();
  assert(!admit_device_allocation(max, 900, 1000, max, 200));
  assert(admit_device_allocation(max-100, max, max, max, 100));
  assert(!admit_device_allocation(max-99, max, max, max, 100));
}''')
    binary = tmp_path / 'probe'
    subprocess.run(['g++', '-std=c++17', '-I', str(root/'src_gpu'), str(source), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.skipif('ASTR_INSITU_DEVICE_PROBE_BIN' not in os.environ, reason='Set the root-built CUDA probe directory')
def test_device_array_and_sort_preflight():
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set the matching MPI prefix; do not launch this MPI-enabled Viskores probe directly'
    command = [str(Path(prefix)/'bin/mpirun'), '--prefix', prefix,
               '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
               '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
               '--mca', 'opal_cuda_support', '0', '-np', '1',
               str(Path(os.environ['ASTR_INSITU_DEVICE_PROBE_BIN'])/'insitu_allocation_guard_probe')]
    result = subprocess.run(command,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'array_rejected_before_malloc=1' in result.stdout
    assert 'scratch_rejection=1' in result.stdout


@pytest.mark.skipif('ASTR_OUTPUT_INSITU_EXE' not in os.environ,
                    reason='Select the root-built strict renderer for the native refusal gate')
def test_native_frame_preflight_refusal(tmp_path):
    from test_insitu_curve_demo import scale_configuration
    from test_insitu_device_curve_wall_fields import current_arguments
    from run_output_restart_validation import run_case

    args = current_arguments(tmp_path, samples=False)
    config = scale_configuration('standard-device', 'pinned').replace(
        'device_budget_bytes=6442450944', 'device_budget_bytes=314572800')
    case, _ = run_case(args, Path(__file__).resolve().parents[2], 'gpu', 2,
        'allocation_refusal', 2, enabled=False, checkpoint_enabled=False,
        grid='32,32,32', tgv_mapping='periodic', insitu_config=config,
        postprocess_transport='pinned', reject='ASTR INSITU ALLOCATION REFUSED',
        failure_after_start=True)
    log = (case/'run.log').read_text()
    refused = re.findall(r'ASTR INSITU ALLOCATION REFUSED source=(.*?) requested=(\d+)', log)
    assert refused and all(source in ('Viskores array', 'Viskores unmanaged/Thrust temporary',
                                      'OpenGL display buffer') and int(size) > 0
                           for source, size in refused), log
    assert 'device allocation would exceed budget/reserve' in log, log
    assert not list((case/'outdat/render').glob('*.jpeg'))
    assert not list((case/'outdat/new').rglob('COMPLETE'))
