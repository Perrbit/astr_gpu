"""IS4 test-only field rejection and preload arming, on private snapshots."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/insitu'))
base=runpy.run_path(str(ROOT/'scripts/insitu/tgv_pipeline.py'))
options=base['options']
fault=base['output'].resolve().parent.parent.name.partition('_fault_')[2]


def catalyst_execute(info):
    if fault=='missing_field' and int(info.timestep)==4:
        # UpdatePipeline may be collective: every rank must enter it once.
        base['source'].UpdatePipeline(float(info.time))
        if base['rank']==1:
            data=base['source'].GetClientSideObject().GetOutputDataObject(0)
            if data.IsA('vtkDataSet'):
                data.GetPointData().RemoveArray('Q_rs')
            else:
                iterator=data.NewIterator(); iterator.InitTraversal()
                while not iterator.IsDoneWithTraversal():
                    leaf=iterator.GetCurrentDataObject()
                    if leaf is not None and leaf.IsA('vtkDataSet'):
                        leaf.GetPointData().RemoveArray('Q_rs')
                    iterator.GoToNextItem()
    base['catalyst_execute'](info)
    if int(info.timestep)==4 and base['rank']==0 and os.environ.get('ASTR_IS4_TEST_PHASE','').startswith('before_'):
        source=Path(os.environ['ASTR_IS4_TEST_TARGET']).parent/'step000000000002'
        before={}
        for path in source.rglob('*'):
            if path.is_file():
                with path.open('rb') as stream:
                    before[str(path.relative_to(source))]=hashlib.file_digest(stream,'sha256').hexdigest()
        (base['output']/'is4_backup_before.json').write_text(json.dumps(before,indent=2))
    target=2 if os.environ.get('ASTR_IS4_TEST_PHASE')=='nonfinite_sample' else 4
    if int(info.timestep)==target and os.environ.get('ASTR_IS4_TEST_PHASE') in ('nonfinite_sample','mpi_failure'):
        arm=ctypes.CDLL(None).astr_is4_test_arm
        arm.argtypes=[]; arm.restype=None; arm()


catalyst_finalize=base['catalyst_finalize']
