"""Compact device geometry; independent final-coordinate interpolation checks."""
import math
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
BIN = Path(os.environ.get('ASTR_INSITU_DEVICE_GEOMETRY_PROBE',
    ROOT / 'build_insitu_device_probes/bin/insitu_device_geometry_probe'))


def launch(ranks, axis, empty, directory):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX to the matching Open MPI installation'
    result = subprocess.run([str(Path(prefix) / 'bin/mpirun'), '--prefix', prefix,
        '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
        '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
        '--mca', 'opal_cuda_support', '0', '-np', str(ranks), str(BIN), str(axis),
        'empty' if empty else 'surface',str(directory/'geometry')], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert 'input_host_mirror=0' in output and 'source_errors=0' in output, output
    assert output.count('exact=1')==ranks,output
    assert len(list(directory.glob('*.vtp')))==ranks*2
    assert sum(p.stat().st_size for p in directory.glob('*.vtp'))<=256*1024*1024
    fields = dict(re.findall(r'(max_error|slice_area|surface_area|slice_cells|surface_cells)=([^ ]+)', result.stdout))
    assert all(math.isfinite(float(v)) for v in fields.values()), output
    assert float(fields['max_error']) <= 2e-10, output
    assert int(fields['slice_cells']) == 1024
    assert abs(float(fields['slice_area']) - 4 * math.pi**2) <= 2e-10
    assert (int(fields['surface_cells']) == 0) == empty
    return fields


@pytest.mark.parametrize('empty', [False, True])
def test_geometry_partition_invariance(empty,tmp_path):
    directory=tmp_path/'np1';directory.mkdir()
    reference = launch(1, 0, empty,directory)
    for axis in range(3):
        directory=tmp_path/f'np2_{axis}';directory.mkdir()
        actual = launch(2, axis, empty,directory)
        assert actual['surface_cells'] == reference['surface_cells']
        assert abs(float(actual['surface_area']) - float(reference['surface_area'])) <= 2e-10


@pytest.mark.parametrize('empty', [False, True])
def test_device_surface_view_without_geometry_readback(empty, tmp_path):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX'
    for ranks, axis in ((1, 0), (2, 0), (2, 1), (2, 2)):
        result = subprocess.run([str(Path(prefix) / 'bin/mpirun'), '--prefix', prefix,
            '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
            '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
            '--mca', 'opal_cuda_support', '0', '-np', str(ranks), str(BIN), str(axis),
            'empty' if empty else 'surface', 'device-view'], cwd=tmp_path,
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count('IS8_DEVICE_SURFACE_VIEW ')==ranks
        assert result.stdout.count('geometry_host_bytes=0')==ranks
        fields = dict(re.findall(r'(max_error|display_error|source_errors)=([^\s]+)', result.stdout))
        assert float(fields['max_error']) <= 2e-10
        assert float(fields['display_error']) == 0
        assert int(fields['source_errors']) == 0
        assert not list(tmp_path.iterdir()), 'Strict component must not write geometry files'


@pytest.mark.parametrize('fault,expected', [
    ('nonfinite', 'Nonfinite device surface'),
    ('bad-index', 'Out-of-range device surface connectivity'),
])
def test_device_surface_view_rejects_invalid_gpu_data(fault, expected, tmp_path):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX'
    result = subprocess.run([str(Path(prefix)/'bin/mpirun'), '--prefix', prefix,
        '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
        '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
        '--mca', 'opal_cuda_support', '0', '-np', '1', str(BIN), '0', 'surface',
        f'device-view-{fault}'], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert expected in result.stdout+result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('empty', [False, True])
def test_curve_contour_empty_and_device_only_views(empty, tmp_path):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX'
    reference = None
    for ranks, axis in ((1, 0), (2, 0), (2, 1), (2, 2)):
        result = subprocess.run([str(Path(prefix)/'bin/mpirun'), '--prefix', prefix,
            '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
            '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
            '--mca', 'opal_cuda_support', '0', '-np', str(ranks), str(BIN), str(axis),
            'curve-empty' if empty else 'curve', 'device-view'], cwd=tmp_path,
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+result.stderr
        fields = dict(re.findall(r'(max_error|display_error|source_errors|surface_cells)=([^\s]+)', result.stdout))
        assert float(fields['max_error']) <= 2e-10 and float(fields['display_error']) == 0.
        assert int(fields['source_errors']) == 0
        assert (int(fields['surface_cells']) == 0) == empty
        if reference is None:
            reference = fields['surface_cells']
        assert fields['surface_cells'] == reference
        assert result.stdout.count('geometry_host_bytes=0') == ranks
    assert not list(tmp_path.iterdir())


def test_curve_nodal_threshold_rejects_collapsed_triangles(tmp_path):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set ASTR_INSITU_DEVICE_MPI_PREFIX'
    result = subprocess.run([str(Path(prefix)/'bin/mpirun'), '--prefix', prefix,
        '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
        '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
        '--mca', 'opal_cuda_support', '0', '-np', '1', str(BIN), '0', 'curve-nodal', 'device-view'],
        cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert 'CURVE contour has nonfinite or zero-area triangles' in result.stdout+result.stderr
    assert not list(tmp_path.iterdir())
