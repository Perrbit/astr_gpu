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
