"""AP5 copy-only halo oracle, including absent optional energy and signed bits."""
import os
from pathlib import Path
import subprocess

import pytest


MPI=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec')


@pytest.fixture(scope='module')
def executable():
    value=os.environ.get('ASTR_AIR5_FLUX_HALO_PROBE_EXE')
    if not value:
        pytest.skip('ASTR_AIR5_FLUX_HALO_PROBE_EXE is not set')
    path=Path(value).resolve(strict=True)
    return path


TOPOLOGIES=[(1,'x','periodic'),(1,'x','physical'),
    (2,'x','periodic'),(2,'y','periodic'),(2,'z','periodic'),
    (2,'x','physical'),(2,'y','physical'),(2,'z','physical'),
    (4,'xy','physical'),(4,'xz','periodic'),(4,'yz','physical'),(8,'xyz','periodic')]


@pytest.mark.parametrize('backend',['pinned','device-aware'])
@pytest.mark.parametrize('mode',['packed','packed-overlap'])
@pytest.mark.parametrize('ranks,topology,boundary',TOPOLOGIES)
def test_flux_halo_bits(tmp_path,executable,backend,mode,ranks,topology,boundary):
    run_probe(tmp_path,executable,backend,mode,ranks,topology,boundary)


@pytest.mark.parametrize('mode',['packed','packed-overlap'])
def test_pageable_flux_halo(tmp_path,executable,mode):
    run_probe(tmp_path,executable,'pageable',mode,4,'xy','physical')


def run_probe(output,executable,backend,mode,ranks,topology,boundary):
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_')}
    env.update(ASTR_GPU_HALO_TRANSPORT=backend,ASTR_AIR5_FLUX_HALO=mode,CUDA_VISIBLE_DEVICES='0,1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[str(MPI),'--mca','coll_hcoll_enable','0','--oversubscribe','-np',str(ranks),
             str(executable),topology,boundary]
    with (output/'run.log').open('w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert result.returncode==0,(output/'run.log').read_text()
    text=(output/'run.log').read_text()
    for count in (27,33):
        assert f'AIR5_FLUX_HALO_PROBE components={count} bit_mismatches=0' in text
