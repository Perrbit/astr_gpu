"""Offline readback of retained real M12 slice evidence, not another CFD run."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('slice_archive', ROOT/'scripts/insitu/slice_archive.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


@pytest.fixture(scope='module')
def segment():
    if os.environ.get('ASTR_SLICE_OFFLINE_SEGMENT'):
        return Path(os.environ['ASTR_SLICE_OFFLINE_SEGMENT']).resolve(strict=True)
    evidence = ROOT/'tests/gpu_validation/out/index_slice_catalog_20261029/representative_evidence/m12'
    paths = sorted(evidence.glob('cpu*/outdat/new/slice_products/*/*/segment*'))
    if not paths:
        pytest.skip('Provide retained real M12 slice evidence first')
    return paths[0]


def test_native_data_geometry_and_profile(segment, tmp_path):
    frame = sorted(segment.glob('step*.h5'))[-1]
    data = TOOL.read_slice(frame)
    with h5py.File(frame) as raw, h5py.File(segment/'geometry.h5') as geometry:
        for name, values in data['fields'].items():
            assert values.dtype == raw[name].dtype
            np.testing.assert_array_equal(values, raw[name])
        np.testing.assert_array_equal(data['coordinates'], geometry['coordinates'])
    output = tmp_path/'profile.csv'
    TOOL.write_profile(frame, 'velocity_x', 0, 3, output)
    table = np.loadtxt(output, delimiter=',', skiprows=1)
    np.testing.assert_array_equal(table[:, :3], data['coordinates'][3, :, :])
    np.testing.assert_array_equal(table[:, 3], data['fields']['velocity_x'][3, :])


def test_move_and_explicit_join(segment, tmp_path):
    left, right = tmp_path/'first', tmp_path/'second'
    shutil.copytree(segment, left);shutil.copytree(segment, right)
    for path, selection in ((left, slice(0, 2)), (right, slice(2, None))):
        tree = ET.parse(path/'series.xdmf')
        collection = tree.find('./Domain/Grid')
        grids = list(collection)
        for grid in grids:
            collection.remove(grid)
        collection.extend(grids[selection])
        tree.write(path/'series.xdmf', encoding='utf-8', xml_declaration=True)
    output = tmp_path/'joined.xdmf'
    TOOL.join_segments([left, right], output)
    tree = ET.parse(output)
    times = [float(node.attrib['Value']) for node in tree.findall('.//Time')]
    assert times == [0., .08, .16]
    for item in tree.findall('.//DataItem[@Format="HDF"]'):
        file, _, dataset = item.text.partition(':/')
        with h5py.File(output.parent/file) as frame:
            assert dataset in frame
    copied = TOOL.read_slice(sorted(right.glob('step*.h5'))[-1])
    original = TOOL.read_slice(sorted(segment.glob('step*.h5'))[-1])
    np.testing.assert_array_equal(copied['fields']['Q_rs'], original['fields']['Q_rs'])


def test_replot_jpeg_and_local_window(segment, tmp_path):
    TOOL.plot_slice(sorted(segment.glob('step*.h5'))[-1], 'velocity_x', tmp_path/'velocity',
                    window=[0, 200, 0, 20])
    with Image.open(tmp_path/'velocity.jpeg') as image:
        assert image.width >= 2000
        assert np.asarray(image.convert('RGB')).std() > 5
    assert not (tmp_path/'velocity.eps').exists()


def test_paraview_xdmf3_values(segment, tmp_path):
    executable = os.environ.get('ASTR_SLICE_PVPYTHON', '/usr/bin/pvpython')
    paths = sorted(segment.glob('step*.h5'))
    payload = {}
    times = []
    for i, path in enumerate(paths):
        data = TOOL.read_slice(path)
        times.append(data['time'])
        for name, values in data['fields'].items():
            payload[f'{i}_{name}'] = values.reshape(-1)
    payload['coordinates'] = data['coordinates'].reshape(-1, 3)
    payload['names'] = np.array(list(data['fields']))
    payload['times'] = np.array(times)
    reference = tmp_path/'reference.npz'
    np.savez(reference, **payload)
    result = subprocess.run([executable, '--no-mpi',
        str(ROOT/'tests/gpu_validation/check_index_slice_paraview.py'), str(segment), str(reference)],
        capture_output=True, text=True)
    (tmp_path/'paraview.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'PARAVIEW_INDEX_SLICE_PASS' in result.stdout


def test_moderate_cost_outputs_and_transfer_accounting():
    report_path=ROOT/'tests/gpu_validation/out/index_slice_catalog_20261029/m12_cost_final/summary.json'
    if not report_path.is_file():
        pytest.skip('Prepare the local moderate-M12 cost evidence first')
    report=json.loads(report_path.read_text())
    metadata={'step','clock','field_ids','index_ranges','reference_state'}
    for case in report['cases']:
        root=Path(case['case'])
        if case['configuration']=='off':
            assert case['field_d2h_total_bytes']==0 and not (root/'outdat/new/slice_products').exists()
            continue
        definitions=[json.loads(line) for line in (root/'input.products.jsonl').read_text().splitlines()[1:]]
        payload=0
        for definition in definitions:
            segment=next((root/'outdat/new/slice_products'/definition['id']).glob('*/segment*'))
            paths=sorted(segment.glob('step*.h5'))
            assert [int(path.stem[4:]) for path in paths]==[0,2,4,6,8,10]
            xml=ET.parse(segment/'series.xdmf')
            clocks=[float(node.attrib['Value']) for node in xml.findall('.//Time')]
            for path,time in zip(paths,clocks,strict=True):
                with h5py.File(path) as frame:
                    assert set(frame)-metadata==set(definition['fields'])
                    assert float(frame['clock'][0])==time
                    for field in definition['fields']:
                        values=frame[field][...]
                        assert values.ndim==2 and values.dtype==np.dtype('float64') and np.isfinite(values).all()
                        payload+=values.nbytes
        assert payload==case['field_d2h_total_bytes']
        if case['configuration']=='two-products':
            archive=root/'outdat/new/slice_products'
            full=next((archive/'basic').glob('*/segment*'))
            region=next((archive/'region').glob('*/segment*'))
            for path in sorted(full.glob('step*.h5')):
                with h5py.File(path) as a,h5py.File(region/path.name) as b:
                    for field in ('velocity_x','pressure'):
                        np.testing.assert_array_equal(a[field][:17,24:49],b[field])
