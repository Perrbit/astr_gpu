"""The exact comparator must compare nested histories, not open groups as data."""
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np
import pytest

PROGRAM=os.environ.get('ASTR_BACKEND_COMPARE')
pytestmark=pytest.mark.skipif(not PROGRAM,reason='Select the root astr_backend_compare target')


def pair(tmp_path,dtype=np.float64):
    a,b=tmp_path/'a.h5',tmp_path/'b.h5'
    with h5py.File(a,'w') as f:
        f['identity']=np.arange(4,dtype=np.int64)
        f['q0001']=np.ones((2,2),dtype=dtype)
        f['inherited/window/q0001']=np.arange(6,dtype=dtype)
    shutil.copy2(a,b)
    return a,b


def run(a,b):
    return subprocess.run([str(Path(PROGRAM).resolve()),'exact',str(a),str(b)],capture_output=True,text=True)


@pytest.mark.parametrize('dtype',(np.float32,np.float64))
def test_nested_history_equal_words(tmp_path,dtype):
    result=run(*pair(tmp_path,dtype))
    assert result.returncode==0 and 'EXACT datasets=3' in result.stdout,result.stdout+result.stderr


@pytest.mark.parametrize('change',('value','missing','kind','extra-empty-group'))
def test_nested_history_differences_are_not_skipped(tmp_path,change):
    a,b=pair(tmp_path)
    with h5py.File(b,'r+') as f:
        if change=='value':f['inherited/window/q0001'][2]=99.
        elif change=='missing':del f['inherited/window/q0001']
        elif change=='kind':
            del f['inherited/window/q0001'];f.create_group('inherited/window/q0001')
        else:f.create_group('extra')
    result=run(a,b)
    assert result.returncode!=0 and 'COMPARE FAILED' in result.stderr,result.stdout+result.stderr
