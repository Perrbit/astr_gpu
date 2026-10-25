"""Check the release export selection and provenance without changing Git."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    'release_package', ROOT/'scripts/release/package_source.py')
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


@pytest.fixture(scope='module')
def exported():
    return PACKAGE.payload('HEAD')


def test_production_dependencies_and_exclusions(exported):
    _, files = exported
    assert {'CMakeLists.txt', 'src/CMakeLists.txt', 'src_gpu/checkpoint_gpu.cuf',
            'LICENSE', 'USER_GUIDE.md', 'examples/GPU_Quickstart/build.sh'} <= files.keys()
    assert not PACKAGE.EXCLUDED & files.keys()
    assert not any(name.startswith(('tests/', 'chem/', '.git/', 'scripts/',
                                    'documents/presentations/')) for name in files)


def test_checksums_cover_every_payload(exported):
    _, files = exported
    listed = {}
    for line in files['SHA256SUMS'][1].decode().splitlines():
        digest, name = line.split('  ', 1)
        assert digest == hashlib.sha256(files[name][1]).hexdigest()
        listed[name] = digest
    assert listed.keys() == files.keys()-{'SHA256SUMS'}


def test_candidate_is_not_platform_release(exported):
    commit, files = exported
    metadata = json.loads(files['RELEASE.json'][1])
    assert metadata['source_commit'] == commit
    assert metadata['status'] == 'local-source-candidate'
    assert metadata['target_platform_qualified'] is False
    assert '-DBUILD_TESTING=OFF' in metadata['configure_required']
