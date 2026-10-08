"""Bounded compact-geometry driver: device halo remains device-only."""
import math
import os
from pathlib import Path
import re
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[2]
BIN=Path(os.environ.get('ASTR_INSITU_STREAMLINES_PROBE',
    ROOT/'build_insitu_device_probes/bin/insitu_device_streamlines_probe'))
INVERSE_BIN=Path(os.environ.get('ASTR_INSITU_CURVE_INVERSE_PROBE',
    ROOT/'build_insitu_device_probes/bin/insitu_curve_inverse_probe'))


@pytest.mark.parametrize('ranks,axis', [(1, 0), (2, 0), (2, 1), (2, 2)])
def test_stratified_tgv_device_streamlines(ranks, axis, record_property):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set the matching Open MPI prefix'
    result = subprocess.run([str(Path(prefix)/'bin/mpirun'), '--prefix', prefix,
        '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
        '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
        '--mca', 'opal_cuda_support', '0', '-np', str(ranks), str(BIN), str(axis),
        'tgv', 'both', 'resident', '', 'tgv-stratified'], capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'STRATIFIED_SEEDS layout=tgv-stratified seeds=256 unique_direction_starts=512' in result.stdout
    assert 'particles=512 resident=1' in result.stdout
    assert 'input_host_mirror=0 source_errors=0' in result.stdout
    error = float(re.search(r'max_error=(\S+)', result.stdout)[1])
    assert math.isfinite(error) and error <= 2e-10
    rounds = int(re.search(r'rounds=(\d+)', result.stdout)[1])
    assert int(re.search(r'control_read_bytes=(\d+)', result.stdout)[1]) <= rounds*32768
    assert int(re.search(r'owner_query_read_bytes=(\d+)', result.stdout)[1]) == 0
    record_property('stratified_tgv_component_output', result.stdout)


def test_curve_inverse_residual(record_property):
    result = subprocess.run([str(INVERSE_BIN)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = re.findall(r'mapping=(\d+) coarse_failures=(\d+) refined_failures=(\d+) '
        r'newton_ok=(\d+) newton_nonconverged=(\d+) residual=(\S+) parameter_error=(\S+)', result.stdout)
    assert len(rows) == 2, result.stdout
    assert {int(row[0]) for row in rows} == {1, 2}
    for row in rows:
        assert int(row[1]) == int(row[2]) == 0, result.stdout
        assert int(row[3]) + int(row[4]) == 32**3, result.stdout
        for value in row[5:]:
            assert math.isfinite(float(value)) and float(value) <= 2e-10, result.stdout
    record_property('physical_inverse_diagnostic', result.stdout)


@pytest.mark.parametrize('constant,forward',[(False,False),(True,False),(True,True)])
@pytest.mark.parametrize('ranks,axis',[(1,0),(2,0),(2,1),(2,2)])
@pytest.mark.parametrize('storage',['compact','resident'])
def test_compact_device_streamlines(ranks,axis,constant,forward,storage):
    prefix=os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix,'Set the matching Open MPI prefix'
    result=subprocess.run([str(Path(prefix)/'bin/mpirun'),'--prefix',prefix,
        '--mca','pml','ob1','--mca','btl','self,tcp','--mca','osc','pt2pt',
        '--mca','coll_hcoll_enable','0','--mca','coll_ucc_enable','0',
        '--mca','opal_cuda_support','0','-np',str(ranks),str(BIN),str(axis),
        'constant' if constant else 'tgv','forward' if forward else 'both',storage],
        capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    match=re.search(r'max_error=(\S+) vertices=(\d+) segments=(\d+) transfers=(\d+)',result.stdout)
    assert match,result.stdout
    assert math.isfinite(float(match[1])) and float(match[1])<=2e-10
    assert int(match[2])>0 and int(match[3])>=(16 if forward else 32)
    assert 'input_host_mirror=0 source_errors=0' in result.stdout
    if constant and ranks==2:
        assert int(match[4])>=16,result.stdout


@pytest.mark.parametrize('mapping', ['periodic', 'y-wavy'])
@pytest.mark.parametrize('ranks,axis', [(1, 0), (2, 0), (2, 1), (2, 2)])
def test_curve_resident_constant_streamlines(ranks, axis, mapping, record_property):
    prefix = os.environ.get('ASTR_INSITU_DEVICE_MPI_PREFIX')
    assert prefix, 'Set the matching Open MPI prefix'
    result = subprocess.run([str(Path(prefix) / 'bin/mpirun'), '--prefix', prefix,
        '--mca', 'pml', 'ob1', '--mca', 'btl', 'self,tcp', '--mca', 'osc', 'pt2pt',
        '--mca', 'coll_hcoll_enable', '0', '--mca', 'coll_ucc_enable', '0',
        '--mca', 'opal_cuda_support', '0', '-np', str(ranks), str(BIN), str(axis),
        'constant', 'forward', 'resident', mapping],
        capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'curve_grid=1 mapping={mapping}' in result.stdout, result.stdout
    match = re.search(r'max_error=(\S+) vertices=(\d+) segments=(\d+) transfers=(\d+)', result.stdout)
    assert match, result.stdout
    error = float(match[1])
    assert math.isfinite(error) and error <= 2e-10
    assert int(match[2]) > 0 and int(match[3]) >= 16
    assert 'input_host_mirror=0 source_errors=0' in result.stdout
    if ranks == 2:
        assert int(match[4]) >= 16, result.stdout
    ledger = re.search(r'rounds=(\d+).*control_read_bytes=(\d+).*owner_query_read_bytes=(\d+)', result.stdout)
    assert ledger, result.stdout
    assert int(ledger[2]) <= int(ledger[1]) * 2048
    assert int(ledger[3]) <= (int(ledger[1]) + 1) * 256
    record_property('constant_curve_maxabs', error)
    record_property('curve_component_output', result.stdout)
