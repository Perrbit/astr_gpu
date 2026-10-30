"""Short real-solver checks of public catalog preparation/verification tools."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_output_insitu_restart import BACKEND,MPIEXEC
from test_index_slice_runtime import ROOT

EXE=os.environ.get('ASTR_CATALOG_ENTRY_EXE')
pytestmark=pytest.mark.skipif(not EXE,reason='Select the native renderer for bounded public-entry checks')


def load(name,script):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts/insitu'/script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_standalone_32_preset_uses_its_explicit_catalog(tmp_path):
    case=tmp_path/'case'
    subprocess.run(['python3',str(ROOT/'scripts/insitu/start_tgv_acceptance.py'),
        '--output',str(case),'--library',str(BACKEND),'--executable',str(Path(EXE).resolve()),
        '--mpiexec',str(MPIEXEC),'--np','2','--processing-backend','device',
        '--postprocess-transport','pinned','--rendering-pipeline','standard-device'],check=True,timeout=180)
    state=json.loads((case/'outdat/new/products.status.json').read_text())
    assert len(state['products'])==6 and all(row['state']=='active' for row in state['products']),state
    assert len(list((case/'outdat/new/render_products').rglob('*.jpeg')))==14
    assert not list((case/'outdat/render').glob('*.jpeg'))
    assert sum(path.stat().st_size for path in case.rglob('*') if path.is_file())<256*1024**2


@pytest.mark.parametrize('pipeline',['standard-device','compatible'])
def test_visualization_benchmark_no_large_io_at_bounded_grid(tmp_path,pipeline):
    benchmark=load('catalog_benchmark','benchmark_tgv256_visualization.py')
    case=tmp_path/'case';benchmark.prepare(case,2,True,BACKEND,'pinned',pipeline,cells=32)
    config=case/'insitu.nml'
    config.write_text(config.read_text().replace('host_budget_bytes=17179869184','host_budget_bytes=4294967296').replace(
        'device_budget_bytes=6442450944','device_budget_bytes=2147483648').replace(
        'device_reserve_bytes=2147483648','device_reserve_bytes=1073741824'))
    command=[str(MPIEXEC),'--mca','coll_hcoll_enable','0','-np','2',str(Path(EXE).resolve()),'run','datin/input.tgv']
    with (case/'run.log').open('w') as log:
        subprocess.run(command,cwd=case,env=benchmark.environment(case,'pinned'),stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=180)
    result=benchmark.verify(case,2,True,pipeline)
    assert result['images']==6 and result['initial_state_frame']
    assert benchmark.timing(case,2)['pure_rk']['per_rank_events']=={'0':2,'1':2}
