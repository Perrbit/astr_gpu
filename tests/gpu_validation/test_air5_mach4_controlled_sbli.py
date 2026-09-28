import json

import numpy as np
import pytest

from prepare_air5_mach4_controlled_sbli import prepare_campaign


def test_campaign_contract(tmp_path):
    root = tmp_path/'campaign'
    manifest = prepare_campaign(root)
    assert manifest['prepared_only'] and not manifest['physical_acceptance']
    assert manifest['physics']['max_scaled_normal_flux_residual'] < 2e-12
    assert manifest['tau'] > 0
    reference = (root/'precursor/datin/air5_hbl_profile.dat').read_bytes()
    for spec in manifest['cases']:
        case = root/spec['name']
        env = json.loads((case/'environment.json').read_text())
        assert env['ASTR_AIR5_SOURCE_MODE'] == spec['source_mode']
        assert 'ASTR_AIR5_TOP_GPU_VALIDATION' not in env
        assert (case/'datin/air5_hbl_profile.dat').read_bytes() == reference
        assert (case/'datin/air5_incident_shock.dat').exists() == spec['incident']
        meta = json.loads((case/'mach4_case_metadata.json').read_text())
        assert meta['incident_shock_enabled'] == spec['incident']
        for path in (case/'datin').iterdir():
            if path.is_file() and path.name.startswith(('input', 'controller')):
                assert b'\r' not in path.read_bytes()
    a = np.loadtxt(root/'sbli_vt_only/datin/air5_incident_shock.dat', skiprows=2)
    b = np.loadtxt(root/'sbli_coupled/datin/air5_incident_shock.dat', skiprows=2)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(FileExistsError):
        prepare_campaign(root)
