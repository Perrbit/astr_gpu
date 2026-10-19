"""The ordinary archive writer must not re-read payloads for checksums."""
from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_archive_publication_uses_name_only_manifests():
    source=(ROOT/'src/output_archive.F90').read_text()
    writer=source.split('subroutine write_frame(',1)[1].split('end subroutine',1)[0]
    assert 'with_checksums=.true.' not in writer
    for call in ('write_checkpoint_resource_refs', 'seal_checkpoint_bundle', 'publish_checkpoint_bundle'):
        assert f'call {call}(' in writer
