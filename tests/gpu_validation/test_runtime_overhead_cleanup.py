"""Bounded old/new ASTR checks for the approved runtime-overhead fixes."""
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from run_output_restart_validation import run_case,compare_fields
from run_output_archive_validation import frames,groups,check_series
from test_output_repartition_runtime import arguments
from test_checkpoint_bundle import fault_library

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/output'))
CAMPAIGN=ROOT/'tests/gpu_validation/out/runtime_overhead_cleanup_20261018'
FROZEN=CAMPAIGN/'frozen/astr_gpu'
CURRENT=Path(os.environ.get('ASTR_OUTPUT_RUNTIME_EXE',ROOT/'build_gpu_probe/bin/astr'))
CPU_CURRENT=Path(os.environ.get('ASTR_OUTPUT_CPU_EXE',ROOT/'build_cpu_probe/bin/astr'))
FINAL='outdat/new/checkpoints/step000000000004'


@pytest.mark.parametrize('backend', ['cpu','gpu'])
@pytest.mark.parametrize('ranks', [1,2])
def test_statistics_and_exact_restart_are_unchanged(tmp_path,backend,ranks):
    args=arguments(tmp_path)
    old=SimpleNamespace(**vars(args))
    old.executable=FROZEN if backend=='gpu' else CAMPAIGN/'frozen/astr_cpu'
    reference,_=run_case(old,ROOT,backend,ranks,'frozen',4,checkpoint_interval=2)
    args.executable=CURRENT if backend=='gpu' else CPU_CURRENT
    actual,_=run_case(args,ROOT,backend,ranks,'current',4,checkpoint_interval=2)
    for member in ('state.h5','statistics.h5'):
        compare_fields(reference/FINAL/member,actual/FINAL/member)
    names=['flowstate.dat']
    if backend=='gpu':
        names+=['gpu_kenergy.dat','gpu_enstophy.dat','gpu_dissipation.dat']
    for name in names:
        assert (reference/name).read_bytes()==(actual/name).read_bytes(),name
    restored,_=run_case(args,ROOT,backend,ranks,'restored',4,
        restore=actual/'outdat/new/checkpoints/step000000000002',checkpoint_interval=2)
    for member in ('state.h5','statistics.h5'):
        compare_fields(actual/FINAL/member,restored/FINAL/member)


def test_archive_payloads_and_no_checksum_readback(tmp_path):
    from repair_series import repair_segment
    args=arguments(tmp_path)
    references={}
    counts={}
    for name,executable in [('frozen',FROZEN),('current',CURRENT)]:
        trace=tmp_path/(name+'.strace')
        wrapper=tmp_path/(name+'.sh')
        wrapper.write_text('#!/bin/sh\nexec strace -f -e trace=openat -o '+
            shlex.quote(str(trace))+' '+shlex.quote(str(executable))+' "$@"\n')
        wrapper.chmod(0o755)
        args.executable=wrapper
        case,_=run_case(args,ROOT,'cpu',1,name,4,checkpoint_interval=2,archive_groups=groups('steps'))
        assert trace.stat().st_size>0
        reads=[]
        for path,flags in re.findall(r'openat\([^,]+, "([^"]+)", ([^)]*)\)',trace.read_text()):
            if path.endswith('/data.h5') and any('/'+p+'/' in path for p in ('fields','slices')) and 'O_RDONLY' in flags:
                reads.append(path)
        counts[name]=reads
        references[name]=case
    assert counts['frozen'] and not counts['current'],counts
    for product in ('fields','slices'):
        old,new=(frames(references[name],product) for name in ('frozen','current'))
        assert old.keys()==new.keys()
        for step,path in new.items():
            compare_fields(old[step]/'data.h5',path/'data.h5')
            assert (path/'COMPLETE').read_text()=='ASTR_COMPLETE_2\n'
            assert (path/'MANIFEST').read_text().startswith('ASTR_CHECKPOINT_BUNDLE 2\n')
        check_series(references['current'],product)
        report=repair_segment(references['current']/'outdat/new'/product/'segment00000000')
        assert report['frames']==len(new)
    (tmp_path/'archive_reads.json').write_text(json.dumps(counts,indent=2)+'\n')


def test_tgv_statistics_kernel_calls_are_halved(tmp_path):
    args=arguments(tmp_path); args.statistics=False
    args.directory_budget_bytes=256*1024**2
    args.nsys_trace_domains='cuda,nvtx'
    records={}
    for name,executable in [('frozen',FROZEN),('current',CURRENT)]:
        args.executable=executable
        case,_=run_case(args,ROOT,'gpu',1,name,3,enabled=False,no_field_io=True,
                        nsys_trace=True,checkpoint_enabled=False)
        with sqlite3.connect((case/'trace.rank0.sqlite').as_uri()+'?mode=ro',uri=True) as db:
            names=[row[0] for row in db.execute('SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '+
                'JOIN StringIds s ON s.id=k.demangledName')]
        records[name]={kernel:sum(kernel in value for value in names) for kernel in (
            'kenergy_partial_kernel','enstophy_partial_kernel','dissipation_partial_kernel')}
    assert all(records['current'][key]>0 and records['frozen'][key]==2*records['current'][key]
               for key in records['current']),records
    (tmp_path/'kernel_counts.json').write_text(json.dumps(records,indent=2)+'\n')


@pytest.mark.parametrize('phase', ['batch_create','batch_rename'])
def test_archive_io_failures_do_not_publish_a_frame(tmp_path,fault_library,phase):
    args=arguments(tmp_path); args.statistics=False
    case=tmp_path/'cpu_np1_failed'
    wrapper=tmp_path/'archive_fault.sh'
    controls=dict(LD_PRELOAD=str(fault_library),ASTR_CHECKPOINT_TEST_FAULT=phase,
        ASTR_CHECKPOINT_TEST_ROOT=str(case/'outdat/new/fields/segment00000000'),
        ASTR_CHECKPOINT_TEST_TARGET='step000000000002')
    wrapper.write_text('#!/bin/sh\n'+''.join('export '+key+'='+shlex.quote(value)+'\n'
        for key,value in controls.items())+'exec '+shlex.quote(str(CURRENT))+' "$@"\n')
    wrapper.chmod(0o755); args.executable=wrapper
    message='cannot create exclusive frame candidate' if phase=='batch_create' else 'cannot publish frame'
    run_case(args,ROOT,'cpu',1,'failed',4,enabled=False,checkpoint_enabled=False,
             archive_groups=groups('steps',slices=False),reject=message,failure_after_start=True)
    segment=case/'outdat/new/fields/segment00000000'
    assert (segment/'LATEST').read_text()=='step000000000000\n'
    assert (segment/'step000000000000/COMPLETE').is_file()
    assert not (segment/'step000000000002').exists()
    assert not (segment/'step000000000004').exists()
