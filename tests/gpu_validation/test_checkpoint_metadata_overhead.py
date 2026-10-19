"""Partition validity must not require one collective per owner pair."""
from pathlib import Path
import json
import re
import pytest
from test_checkpoint_state import run

ROOT=Path(__file__).resolve().parents[2]


def test_partition_validation_is_local_until_one_global_gate():
    source=(ROOT/'src/checkpoint_state_io.F90').read_text()
    helper=source.split('subroutine validate_partitions_local(',1)[1].split('end subroutine',1)[0]
    assert 'MPI_' not in helper and 'call require(' not in helper
    for message in ('overlapping owners','incomplete ownership','extras count'):
        assert message in helper


@pytest.mark.parametrize('mode', ['write','exact'])
def test_partition_gate_has_constant_collective_count(tmp_path, mode):
    counts={}
    for ranks in (1,2,4):
        path=tmp_path/f'np{ranks}.h5'
        initial=run(path,'write',ranks)
        log=initial if mode=='write' else run(path,'exact',ranks)
        counts[ranks]=int(re.search(r'CHECKPOINT_ALLREDUCE_CALLS\s+(\d+)',log).group(1))
    (tmp_path/'counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    assert len(set(counts.values()))==1, counts
