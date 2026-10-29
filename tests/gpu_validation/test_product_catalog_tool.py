"""Atomic control-file tools; no solver, GPU, MPI or simulated flow."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('catalog_tool',
    Path(__file__).resolve().parents[2] / 'scripts/insitu/product_catalog.py')
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
HEADER = json.dumps({'catalog_schema': TOOL.SCHEMA}) + '\n'


def test_publish_keeps_bad_body_record_and_valid_later_record(tmp_path):
    formal, candidate = tmp_path / 'formal', tmp_path / 'candidate'
    formal.write_text(HEADER)
    content = HEADER + '{"id":"broken",\n' + '{"id":"good","enable_request":1}\n'
    candidate.write_text(content)
    warnings = TOOL.publish(candidate, formal)
    assert warnings and formal.read_text() == content
    assert not candidate.exists()


def test_bad_header_leaves_formal_unchanged(tmp_path):
    formal, candidate = tmp_path / 'formal', tmp_path / 'candidate'
    formal.write_text(HEADER)
    candidate.write_text('{"catalog_schema":"unknown"}\n')
    with pytest.raises(ValueError):
        TOOL.publish(candidate, formal)
    assert formal.read_text() == HEADER and candidate.exists()


def test_enable_preserves_other_broken_records_and_exact_integer(tmp_path):
    formal = tmp_path / 'formal'
    prefix = HEADER + '{"id":"broken",\n'
    formal.write_text(prefix + json.dumps({'id': 'good', 'enable_request': 9007199254740993}) + '\n')
    previous = formal.stat().st_ino
    number, warnings = TOOL.enable(formal, 'good')
    assert number == 9007199254740994 and warnings
    assert formal.read_text().startswith(prefix)
    assert json.loads(formal.read_text().splitlines()[-1])['enable_request'] == number
    assert formal.stat().st_ino != previous
    assert not list(tmp_path.glob('*.candidate.*'))


@pytest.mark.parametrize('body', [
    {'id': 'good', 'enable_request': True},
    {'id': 'good', 'enable_request': TOOL.MAX_REQUEST},
    {'id': 'other', 'enable_request': 1},
])
def test_bad_enable_does_not_modify_file(tmp_path, body):
    formal = tmp_path / 'formal'
    content = HEADER + json.dumps(body) + '\n'
    formal.write_text(content)
    with pytest.raises(ValueError):
        TOOL.enable(formal, 'good')
    assert formal.read_text() == content


def test_duplicate_id_is_not_first_or_last_wins(tmp_path):
    formal = tmp_path / 'formal'
    record = json.dumps({'id': 'good', 'enable_request': 1}) + '\n'
    formal.write_text(HEADER + record + record)
    with pytest.raises(ValueError):
        TOOL.enable(formal, 'good')
    assert formal.read_text() == HEADER + record + record


def test_explicit_empty_catalog_is_publishable(tmp_path):
    formal, candidate = tmp_path / 'formal', tmp_path / 'candidate'
    candidate.write_text(HEADER)
    assert TOOL.publish(candidate, formal) == []
    assert formal.read_text() == HEADER
