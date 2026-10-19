"""CP8 execution changes must not alter the compact operator or exact state."""
import os

import pytest

from run_output_restart_validation import compare_fields
from test_compact_restart_runtime import launch,batch

pytestmark=pytest.mark.skipif(not os.environ.get('ASTR_COMPACT_CPU_EXE') or
    not os.environ.get('ASTR_COMPACT_GPU_EXE'),reason='set both compact executables')


@pytest.mark.parametrize('topology', [(2,1,1),(1,2,1),(1,1,2)])
@pytest.mark.parametrize('workspace', ['scalar','full'])
def test_dependency_batches_are_bitwise(tmp_path,topology,workspace):
    reference=launch(tmp_path,'gpu',2,'explicit',4,workspace=workspace,topology=topology)
    candidate=launch(tmp_path,'gpu',2,'dependency',4,workspace=workspace,topology=topology,
                     compact_sync='dependency',compact_batch_rhs=1024)
    compare_fields(batch(reference,4)/'state.h5',batch(candidate,4)/'state.h5')
    assert 'COMPACT_GPU_EXECUTION sync=dependency batch_rhs=1024' in (candidate/'run.log').read_text()


def test_execution_choice_preserves_exact_continuation(tmp_path):
    reference=launch(tmp_path,'gpu',2,'continuous',12)
    first=launch(tmp_path,'gpu',2,'first',5)
    resumed=launch(tmp_path,'gpu',2,'resumed',12,workspace='full',restore=batch(first,5),
                   compact_sync='dependency',compact_batch_rhs=1024)
    compare_fields(batch(reference,12)/'state.h5',batch(resumed,12)/'state.h5')


def test_dependency_batches_memcheck(tmp_path):
    candidate=launch(tmp_path,'gpu',2,'dependency',4,workspace='full',
                     compact_sync='dependency',compact_batch_rhs=1024,memcheck=True)
    assert batch(candidate,4).is_dir()


@pytest.mark.parametrize('mode,capacity', [('invalid',65535),('dependency',65536)])
def test_execution_configuration_is_explicit(tmp_path,mode,capacity):
    launch(tmp_path,'gpu',2,'invalid',1,compact_sync=mode,compact_batch_rhs=capacity,
           reject='compact execution requires consistent explicit/dependency sync and batch RHS in 1:65535')
