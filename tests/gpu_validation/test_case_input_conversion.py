"""Input conversion tests; no flow, thermochemistry or reference solver."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/input'))
from convert_legacy_input import convert_case, convert_controller, namelist, values, literal_environment, MODEL_ENVIRONMENT


def tgv_text():
    return (ROOT/'examples/Taylor_Green_Vortex/datin/input.tgv').read_text()


def dimensional_text():
    return '''! Input-reader fixture, not a physical simulation
swbli
32,24,8
f,f,f
f,t,t,t,f,t,f,f,f
f
0.49d0,48
300.d0,600.d0,0.01d0,0.2d0
743e,643e,rk3
5,t,0.3d0,0.05d0
2,0.7d0,0.8d0
none,h
11,prof
22,0.9d0
411,0.1d0,300.d0
41,350.d0
421,0.1d0
51,0.6d0,30.d0
2
1,2,3,4,5,6,layer
datin/grid with space.h5
stl,datin/body.stl
'''


def test_conversion_preserves_literals_and_cpu_default():
    groups = convert_case(tgv_text())
    assert groups['case_options'] == dict(flowtype='tgv', ltimrpt=True, use_gpu=False, iomode='h')
    assert groups['mesh']['ia'] == 128
    assert groups['physics']['mach'] == '0.1d0'
    assert groups['numerics']['bfacmpld'] == '0.3d0'
    assert 'odetype' not in groups['physics']  # Ignored by the non-COMB reader.
    result = namelist(groups)
    assert "flowtype = 'tgv'" in result
    assert 'mach = 0.1d0' in result


def test_dimensional_species_boundary_and_resources():
    groups = convert_case(dimensional_text())
    assert groups['physics']['ref_vel'] == '600.d0'
    assert groups['species_transport']['schmidt'] == ['0.7d0', '0.8d0']
    assert groups['boundary']['bctype'] == [11, 22, 411, 41, 421, 51]
    assert groups['boundary']['twall'][2:4] == ['300.d0', '350.d0']
    assert groups['boundary']['xslip'] == '0.1d0'
    assert groups['boundary']['angshk'] == '30.d0'
    assert groups['initialization']['solidfile'] == 'datin/body.stl'
    assert groups['mesh']['gridfile'] == 'datin/grid with space.h5'
    assert list(groups['sponge'].values()) == [1, 2, 3, 4, 5, 6, 'layer']


def test_conflicting_shared_values_need_human_decision():
    with pytest.raises(ValueError, match='conflicting shared boundary parameter xslip'):
        convert_case(dimensional_text().replace('421,0.1d0', '421,0.2d0'))


def test_crlf_repetition_and_unknown_tail():
    assert convert_case(tgv_text().replace('\n', '\r\n')) == convert_case(tgv_text())
    assert values('6*0, "quoted path.h5" ! ignored') == ['0']*6+['quoted path.h5']
    with pytest.raises(ValueError, match='trailing input'):
        convert_case(tgv_text()+'unexpected-resource.dat\n')


def test_controller_fixed_dt_and_legacy_extra_complete_step():
    text = 'f,f,t,t\n10,4,99,88,2,3\n1.D-3\n'
    groups = convert_controller(text)
    assert groups['run_control'] == {'max_steps': 11, 'reload_steps': 4}
    assert groups['timestep'] == {'mode': 'fixed', 'fixed_dt': '1.D-3'}
    assert groups['repair']['crashfix']
    assert groups['statistics'] == dict(enabled=True, sample_clock='steps', sample_steps=3, write_steps=0)


def test_comb_parameters_remain_distinct():
    text = tgv_text().replace('t,t,t,f,f,f,t,f', 't,t,t,f,f,f,t,t')+'datin/mechanism.yaml\n'
    groups = convert_case(text, comb=True)
    assert groups['physics']['chemfile'] == 'datin/mechanism.yaml'
    assert groups['physics']['odetype'] == 'rk3'
    assert groups['physics']['lcomb']


def test_model_environment_is_explicit_literal_input():
    environment = literal_environment('''export ASTR_PERFECT_GAS_PRANDTL=0.71d0
ASTR_NSCBC_FARFIELD_MODE=incoming_only
export ASTR_AIR5_LOW_ACTIVITY=off
export ASTR_GPU_HALO_TRANSPORT=device-aware
''')
    groups = convert_case(tgv_text(), model_environment=environment)
    assert groups['transport_options']['perfect_gas_prandtl'] == '0.71d0'
    assert groups['boundary_options']['nscbc_farfield_mode'] == 'incoming_only'
    assert groups['chemistry']['low_activity_threshold'] == '0'
    assert 'gpu_halo_transport' not in namelist(groups)
    with pytest.raises(ValueError, match='resolve shell expressions'):
        literal_environment('export ASTR_AIR5_TOP_TAU=$TAU')


def test_model_option_registry_matches_the_solver():
    import re
    source = (ROOT/'src/case_model_options.F90').read_text()
    keys = re.findall("'([^']*)'", source.split('keys(count)=')[1].split(']')[0])
    retired = re.findall("'([^']*)'", source.split('retired(count)=')[1].split(']')[0])
    assert dict(zip(retired,keys,strict=True)) == {k:v[1] for k,v in MODEL_ENVIRONMENT.items()}
