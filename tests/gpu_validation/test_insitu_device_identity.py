import pytest

from insitu_device_identity import match_device


def test_mapping_uses_uuid_not_ordinal():
    cuda = {"cuda_ordinal": 0, "uuid": "second", "pci_bus_id": "73:00.0"}
    egl = [{"egl_index": 0, "uuid": "first"}, {"egl_index": 1, "uuid": "second"},
           {"egl_index": 2, "uuid": None}]
    assert match_device(cuda, egl)["egl_index"] == 1


@pytest.mark.parametrize("egl", [[], [{"egl_index": 0, "uuid": "wrong"}],
                                    [{"egl_index": 0, "uuid": "same"},
                                     {"egl_index": 1, "uuid": "same"}]])
def test_ambiguous_or_missing_mapping_rejected(egl):
    with pytest.raises(RuntimeError, match="Expected one EGL device"):
        match_device({"uuid": "same"}, egl)
