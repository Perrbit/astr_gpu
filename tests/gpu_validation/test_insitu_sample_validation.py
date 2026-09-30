import numpy as np
import pytest

from run_insitu_sample_validation import diagnostics, read_sample


@pytest.mark.parametrize('signature,canonical', [(b'ASTRIS01', True), (b'ASTRIC01', False)])
def test_raw_and_canonical_cannot_be_interchanged(tmp_path, signature, canonical):
    path = tmp_path/'sample.bin'
    path.write_bytes(signature)
    with pytest.raises(ValueError, match='Invalid sample signature'):
        read_sample(path, canonical=canonical)


def test_q_uses_rotation_strain_definition_in_compressible_field():
    gradient = np.eye(3)[None,None,None,...]
    result = diagnostics(gradient)
    assert result[...,9].item() == -1.5
    assert result[...,10].item() == 3.0
    np.testing.assert_array_equal(result[...,11:14], np.zeros((1,1,1,3)))
