"""Independent automatic CFL=0.3 window, not a relabelled CFL=0.5 pass."""
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from compare_flowstate import read_flowstate
from run_h1_hcw_span_grid import analyze_case,clock
from test_high_order_boundary_runtime import EXE,H1_VARIANT,launch_hcw
from test_h1_hcw_long_window import state_report

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/'tests/gpu_validation/out/h1_span_grid_20261105/window_after_archive_fix/span2mm_n16'
pytestmark=pytest.mark.skipif(not EXE or H1_VARIANT!='inlet-compatible',
    reason='Select the approved root executable and H1 candidate')


def test_independent_cfl03_physical_window(tmp_path):
    target=float(clock(REFERENCE)[-1,2])
    fields={};clocks={};statistics={};report=dict(cfl_target=.3,growth_factor=1.1,
        scope='64x48x5 NP2x independent auto, same physical target; not a two-flowthrough validation',
        target_time=target,clock_tolerance=1e-12,field_threshold=2e-10,cases={})
    for backend in ('gpu','cpu'):
        case=tmp_path/backend
        launch_hcw(case,backend,steps=420,automatic_cfl=.3,stop_time=target,
                   checkpoint_interval=1000000,volume_time_interval=target/10)
        clocks[backend]=clock(case)
        steps=int(clocks[backend][-1,0])
        fields[backend],_=state_report(case,steps)
        statistics[backend]=read_flowstate(case)
        summary=analyze_case(case)
        assert summary['last_time_start']<target<=summary['final_time']
        upper=[float(row.split()[-1]) for row in re.findall(
            r'ASTR_CFL directional/local_sum/upper_bound=([^\n]+)',(case/'run.log').read_text())]
        summary['observed_cfl_upper_range']=[min(upper),max(upper)]
        report['cases'][backend]=summary
        (tmp_path/'cfl03.json').write_text(json.dumps(report,indent=2)+'\n')
    assert clocks['cpu'].shape==clocks['gpu'].shape,report['cases']
    np.testing.assert_array_equal(clocks['cpu'][:,0],clocks['gpu'][:,0])
    clock_error=float(abs(clocks['cpu'][:,1:]-clocks['gpu'][:,1:]).max())
    errors={name:float((abs(fields['gpu'][name]-value)/(1+abs(value))).max())
            for name,value in fields['cpu'].items()}
    names,cpu=statistics['cpu'];other,gpu=statistics['gpu']
    assert names==other
    np.testing.assert_array_equal(cpu[:,0],gpu[:,0])
    statistics_error=float(abs(cpu[:,2:]-gpu[:,2:]).max())
    samples=[]
    files={backend:sorted((tmp_path/backend/'outdat/new/fields').glob('segment*/step*/data.h5'))
           for backend in ('cpu','gpu')}
    assert [file.parent.name for file in files['cpu']]==[file.parent.name for file in files['gpu']]
    for first,second in zip(files['cpu'],files['gpu']):
        with h5py.File(first) as a,h5py.File(second) as b:
            time_error=float(abs(a['metadata'][2:3].view(np.float64)[0]-b['metadata'][2:3].view(np.float64)[0]))
            error=max(float((abs(b[name][...]-a[name][...])/(1+abs(a[name][...]))).max())
                      for name in ('density','velocity','pressure','temperature'))
        samples.append(dict(step=first.parent.name,time_difference=time_error,scaled_primitive_error=error))
    report.update(maximum_backend_clock_difference=clock_error,clocks_within_gate=clock_error<=1e-12,
        scaled_final_field_errors=errors,maximum_statistics_absolute_difference=statistics_error,
        statistics_time_text_difference=float(abs(cpu[:,1]-gpu[:,1]).max()),samples=samples)
    (tmp_path/'cfl03.json').write_text(json.dumps(report,indent=2)+'\n')
    assert clock_error<=1e-12,report
    assert max(errors.values())<=2e-10,report
    assert max(sample['scaled_primitive_error'] for sample in samples)<=2e-10,report
    assert statistics_error<=2e-10,report
