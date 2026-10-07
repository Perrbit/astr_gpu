"""GPU-only bulk plane extraction with bounded test-only geometric oracles."""
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(os.environ.get('ASTR_INSITU_STRUCTURED_PROBE',
                            ROOT / 'build_insitu_device_render/bin/insitu_device_structured_probe'))
PREFIX = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')


@pytest.mark.parametrize('ranks,axis', [(1, 0), (2, 0), (2, 1), (2, 2)])
def test_bulk_plane_geometry(ranks, axis):
    assert PREFIX, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX to the matching MPI installation'
    env = dict(os.environ, OMPI_MCA_coll_hcoll_enable='0', OMPI_MCA_coll_ucc_enable='0')
    result = subprocess.run([str(Path(PREFIX) / 'bin/mpirun'), '--prefix', PREFIX,
                             '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp',
                             '--mca', 'osc', 'pt2pt', '--mca', 'opal_cuda_support', '0',
                             '-np', str(ranks), str(PROBE), str(axis)],
                            env=env, capture_output=True, text=True, timeout=90)
    output = result.stdout + result.stderr
    print(output)
    assert result.returncode == 0, output
    assert 'PASS structured device scan, key welding, topology and immutable source' in output
    assert output.count('zero_normal=1 budget=1 inverted_cell=1 nonfinite_field=1') == ranks
    assert output.count('device_only_before_oracle=1 test_geometry_readback=1 runtime_admission=0') == 6
    values = re.findall(r'max_error=(\S+)', output)
    assert len(values) == 4 and all(float(value) <= 2e-10 for value in values), output
