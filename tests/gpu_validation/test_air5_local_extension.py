"""LE1-LE3 native checks; run explicitly within the approved local campaign."""
import json
import os
import subprocess
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from run_air5_performance_diagnosis import digest
import run_air5_performance_diagnosis as performance
from run_output_air5_restart_validation import launch as native_launch, state_sanity
from run_output_restart_validation import compare_fields
from test_output_air5_conservation import arguments, FINAL
from test_air5_integration_control_restart import history_summary
from run_air5_chemistry_performance_gate import chemistry_failure_hook
from run_air5_layered_stress_gate import check_shared_face_payloads
from run_air5_flow_control_gate import load_references
import run_air5_local_development as development

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=Path(os.environ.get('ASTR_AIR5_EXTENSION_REFERENCE',
    ROOT/'tests/gpu_validation/out/air5_g2_64_20261013/local_mach4_development_20261017_r4/astr'))


def launch(*args,**kwargs):
    kwargs.setdefault('host_budget_bytes',256*1024**2)
    kwargs.setdefault('device_budget_bytes',256*1024**2)
    kwargs.setdefault('directory_budget_bytes',1024**3)
    return native_launch(*args,**kwargs)


def engineering(output, topology=(1, 1, 1), workspace='scalar', history='off'):
    args = arguments(output, case='sbli', workspace=workspace)
    args.topology = topology
    args.air5_filter_mode = 'conservative-flux'
    args.source_evaluation = 'source-only'
    args.chemistry_block_size = 32
    args.report_interval = 4
    args.step_history = history
    args.runtime_timeout_seconds = 300
    return args


def field_errors(left, right):
    errors = {}
    with h5py.File(left) as a, h5py.File(right) as b:
        names = [name for name in a if name.startswith('q')]
        assert names == [name for name in b if name.startswith('q')]
        for name in names:
            errors[name] = float(np.max(np.abs(a[name][:] - b[name][:])))
            assert errors[name] <= 2e-10, (name, errors[name])
    return errors


def density_padding(path):
    checked=0
    with h5py.File(path) as state:
        header=state['identity'][:]
        extras=state['rank_extras'][:]
        offset=0
        for partition in state['partitions'][:].reshape(-1,8):
            origin,cells,halo=partition[:3],partition[3:6],int(partition[6])
            owned=cells+(origin+cells==header[3:6]-1)
            z,y,x=np.indices(tuple((cells+2*halo+1)[::-1]))
            coordinates=np.stack((x,y,z),axis=-1).reshape(-1,3)
            coordinates=coordinates[~np.all((coordinates>=halo)&(coordinates<halo+owned),axis=-1)]-halo
            padding=np.count_nonzero((coordinates<0)|(coordinates>cells),axis=-1)>=2
            count=int(partition[7])
            values=extras[offset:offset+count].reshape(int(header[6]),-1)[22]
            assert np.all(values[padding].view(np.uint64)==0), 'noncanonical AIR5 density edge/corner cache'
            checked+=int(np.count_nonzero(padding))
            offset+=count
    return checked


@pytest.mark.parametrize('case', ['air5','fluid'])
def test_density_padding_allocation(case):
    executable=arguments(ROOT).executable.parent/'air5_density_padding_probe'
    environment=dict(os.environ,MALLOC_PERTURB_='165')
    result=subprocess.run([str(executable),case],env=environment,text=True,capture_output=True,check=True)
    expected='T 3400 2414' if case=='air5' else 'F 0 5814'
    assert f'DENSITY_PADDING_PASS air5={expected}' in result.stdout


@pytest.mark.parametrize('ranks', [1, 2])
def test_reuse_and_packed_exact_restart(tmp_path, ranks):
    args = engineering(tmp_path, (ranks, 1, 1))
    reference = None
    evidence = []
    for reuse, reductions in [('off', 'baseline'), ('chemistry', 'baseline'),
                              ('off', 'packed'), ('chemistry', 'packed')]:
        args.primitive_reuse, args.chemistry_reductions = reuse, reductions
        label = f'{reuse}_{reductions}'
        whole, _ = launch(args, 'gpu', ranks, label, 12, chemistry_diagnostics='report-steps')
        first, _ = launch(args, 'gpu', ranks, label+'_first', 5, chemistry_diagnostics='report-steps')
        seed = first/'outdat/new/checkpoints/step000000000005'
        resumed, _ = launch(args, 'gpu', ranks, label+'_resumed', 12, restore=seed,
                            chemistry_diagnostics='report-steps')
        for name in ('state.h5', 'statistics.h5'):
            compare_fields(whole/FINAL/name, resumed/FINAL/name)
        if reference is None:
            reference = whole
        errors = field_errors(reference/FINAL/'state.h5', whole/FINAL/'state.h5')
        reports = lambda case: [line for line in (case/'run.log').read_text().splitlines()
                                if line.startswith('AIR5_CHEMISTRY half=')]
        assert reports(reference) == reports(whole)
        assert reports(first)+reports(resumed) == reports(whole)
        evidence.append(dict(reuse=reuse, reductions=reductions, errors=errors,
                             sanity=state_sanity(whole/FINAL/'state.h5')))
    (tmp_path/'le1.json').write_text(json.dumps(dict(executable_sha256=digest(args.executable),
        topology=list(args.topology), cases=evidence), indent=2, allow_nan=False)+'\n')


@pytest.mark.parametrize('topology', [(1,1,1),(2,1,1),(1,2,1),(1,1,2)])
def test_private_transport_matches_frozen_program(tmp_path,topology):
    ranks=int(np.prod(topology))
    args=engineering(tmp_path,topology)
    old=SimpleNamespace(**vars(args))
    old.executable=REFERENCE.resolve(strict=True)
    reference,_=launch(old,'gpu',ranks,'frozen',12,chemistry_diagnostics='report-steps')
    records={}
    for transport in ('pinned','device-aware'):
        args.halo_transport=transport
        case,_=launch(args,'gpu',ranks,transport,12,chemistry_diagnostics='report-steps')
        records[transport]={name:compare_fields(reference/FINAL/name,case/FINAL/name)
                            for name in ('state.h5','statistics.h5')}
        assert (reference/'flowstate.dat').read_bytes()==(case/'flowstate.dat').read_bytes()
    (tmp_path/'le2.json').write_text(json.dumps(dict(topology=topology,
        old_executable_sha256=digest(old.executable),executable_sha256=digest(args.executable),
        bitwise_datasets=records),indent=2)+'\n')


@pytest.mark.parametrize('topology,grid', [((1,1,1),16),((2,1,1),16),((1,2,1),16),
    ((1,1,2),16),((2,2,1),32),((2,1,2),32),((1,2,2),32),((2,2,2),32)])
def test_integrated_topologies(tmp_path,topology,grid):
    ranks=int(np.prod(topology))
    args=engineering(tmp_path,topology,history='previous')
    args.grid=(grid,)*3
    args.share_gpus=ranks>2
    records,cases={},{}
    for backend in ('cpu','gpu'):
        args.source_evaluation='joint' if backend=='cpu' else 'source-only'
        args.chemistry_block_size=128 if backend=='cpu' else 32
        args.primitive_reuse='off' if backend=='cpu' else 'chemistry'
        args.chemistry_reductions='baseline' if backend=='cpu' else 'packed'
        args.filter_workspace='scalar'
        whole,size=launch(args,backend,ranks,'whole',12,chemistry_diagnostics='report-steps')
        padding=density_padding(whole/FINAL/'state.h5')
        first,_=launch(args,backend,ranks,'first',5,chemistry_diagnostics='report-steps')
        seed=first/'outdat/new/checkpoints/step000000000005'
        history=history_summary(seed/'state.h5')
        # A transport change does not change the exact numerical restart contract.
        args.halo_transport='device-aware' if backend=='gpu' else 'pinned'
        resumed,_=launch(args,backend,ranks,'resumed',12,restore=seed,
                         chemistry_diagnostics='report-steps')
        exact={name:compare_fields(whole/FINAL/name,resumed/FINAL/name)
               for name in ('state.h5','statistics.h5')}
        args.filter_workspace='full'
        full,_=launch(args,backend,ranks,'full',12,chemistry_diagnostics='report-steps')
        for name in ('state.h5','statistics.h5'):
            compare_fields(whole/FINAL/name,full/FINAL/name)
        cases[backend]=whole
        records[backend]=dict(history=history,exact_datasets=exact,
            sanity=state_sanity(whole/FINAL/'state.h5'),directory_bytes=size,density_padding_nodes=padding)
    errors=field_errors(cases['cpu']/FINAL/'state.h5',cases['gpu']/FINAL/'state.h5')
    (tmp_path/'le3.json').write_text(json.dumps(dict(executable_sha256=digest(args.executable),
        topology=topology,grid_intervals=args.grid,shared_gpu_correctness_only=ranks>2,
        backends=records,cpu_gpu_errors=errors),indent=2,allow_nan=False)+'\n')


@pytest.mark.parametrize('transport,topology', [('pinned',(2,2,2)),('device-aware',(2,1,1)),
    ('device-aware',(1,2,1)),('device-aware',(1,1,2))])
def test_private_transport_memcheck(tmp_path,transport,topology):
    ranks=int(np.prod(topology))
    args=engineering(tmp_path,topology,history='previous')
    args.grid=(32,)*3 if ranks>2 else (16,)*3
    args.halo_transport=transport
    args.share_gpus=ranks>2
    args.mean_statistics=False
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    case,_=launch(args,'gpu',ranks,'memcheck',2,memcheck=True,chemistry_diagnostics='report-steps')
    assert len(list(case.glob('memcheck.*.log')))==ranks
    state_sanity(case/'outdat/new/checkpoints/step000000000002/state.h5')


def test_uncompensated_filter_transport(tmp_path):
    args=engineering(tmp_path,(1,2,1),workspace='full')
    args.compensation='off'
    args.mean_statistics=False
    cases=[]
    for transport in ('pinned','device-aware'):
        args.halo_transport=transport
        case,_=launch(args,'gpu',2,transport,4)
        cases.append(case/'outdat/new/checkpoints/step000000000004/state.h5')
    compare_fields(*cases)
    state_sanity(cases[0])


def test_mach4_hundred_step_integration(tmp_path,monkeypatch):
    original=development.prepare_window
    selection={}

    def prepare(case,variant,steps,preflight):
        runtime,record=original(case,variant,steps,preflight)
        runtime.update(selection)
        record['local_extension']=dict(selection)
        return runtime,record

    monkeypatch.setattr(development,'prepare_window',prepare)
    monkeypatch.setattr(development,'BUDGET',64*1024**3)
    campaign=ROOT/'tests/gpu_validation/out/air5_g2_64_20261013'
    records={}
    records['frozen']=development.run_case(tmp_path/'frozen',REFERENCE,'combined',100,True,campaign)
    for transport in ('pinned','device-aware'):
        selection.update(ASTR_AIR5_PRIMITIVE_REUSE='chemistry',
                         ASTR_AIR5_CHEMISTRY_REDUCTIONS='packed',ASTR_GPU_HALO_TRANSPORT=transport)
        if transport=='device-aware':
            selection.update(OMPI_MCA_pml='ucx',OMPI_MCA_coll='^hcoll,ucc,cuda',
                OMPI_MCA_osc='pt2pt',OMPI_MCA_opal_cuda_support='true',
                UCX_MEMTYPE_CACHE='n',UCX_CUDA_COPY_ENABLE_FABRIC='no',
                UCX_CUDA_COPY_DMABUF='no',UCX_CUDA_IPC_ENABLE_MNNVL='no',
                UCX_TLS='self,sm,cuda_copy,cuda_ipc')
        records[transport]=development.run_case(tmp_path/transport,
            arguments(tmp_path).executable,'combined',100,True,campaign)
        compare_fields(Path(records['frozen']['final_state']),Path(records[transport]['final_state']))
        for name in ('air5_profiles.dat','air5_wall.dat'):
            assert (tmp_path/'frozen/monitor'/name).read_bytes()==(tmp_path/transport/'monitor'/name).read_bytes()
    (tmp_path/'mach4_extension.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


@pytest.mark.parametrize('transport', ['pinned','device-aware'])
def test_combination_failure_gates(tmp_path,transport):
    args=engineering(tmp_path,(2,1,1),history='previous')
    args.halo_transport=transport
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    args.mean_statistics=False
    first,_=launch(args,'gpu',2,'first',5,chemistry_diagnostics='report-steps')
    library=ROOT/'tests/gpu_validation/out/air5_g2_64_20261013/le2_frozen_20261018/packed_failure.so'
    failed,_=launch(args,'gpu',2,'chemistry_failure',12,
        restore=first/'outdat/new/checkpoints/step000000000005',override=True,
        chemistry_diagnostics='report-steps',air5_phase_timing=True,
        chemistry_failure_hook=chemistry_failure_hook(args.executable,library),
        reject='air5 GPU chemistry half 1 failed, status=99')
    text=(failed/'run.log').read_text()
    assert 'AIR5_TEST_INJECT chemistry rank=1 count=1 status=99' in text
    assert 'chem_failure_reduce' in text
    assert all(name not in text for name in ('chem_boundary','chem_report_reduce','transport_filter'))
    failed,_=launch(args,'gpu',2,'filter_failure',1,filter_failure_rank=1,
        chemistry_diagnostics='report-steps',reject='injected failure before commit')
    text=(failed/'run.log').read_text()
    assert all(f'AIR5_FILTER_TRANSACTION rank={rank} unchanged=T' in text for rank in (0,1))


@pytest.mark.parametrize('topology', [(2,1,1),(1,2,1),(1,1,2)])
def test_owned_face_payloads(tmp_path,topology):
    args=engineering(tmp_path,topology)
    args.mean_statistics=False
    old=SimpleNamespace(**vars(args))
    old.executable=REFERENCE.resolve(strict=True)
    reference,_=launch(old,'gpu',2,'frozen',1,symmetric_face_probe=True)
    records={}
    for transport in ('pinned','device-aware'):
        args.halo_transport=transport
        case,_=launch(args,'gpu',2,transport,1,symmetric_face_probe=True)
        records[transport]=check_shared_face_payloads(case,topology,periodic=(False,False,True))
        for path in (reference/'validation').glob('air5*shared_faces*.bin'):
            assert path.read_bytes()==(case/'validation'/path.name).read_bytes()
    (tmp_path/'owned_faces.json').write_text(json.dumps(dict(topology=topology,checks=records),indent=2)+'\n')


def test_five_round_timing(tmp_path,monkeypatch):
    campaign=ROOT/'tests/gpu_validation/out/air5_g2_64_20261013'
    source=campaign/'controls_conservative_20261016_r1/flow_control.json'
    controls=json.loads(source.read_text())
    assert controls['status']=='local-flow-control-error-passed-not-performance'
    references=load_references(campaign/'references_conservative_20261016_r1/reference_summary.json',
                               'conservative-flux')
    assert references==controls['reference_summary']
    old_rows={f"gpu_np{row['ranks']}":row for row in controls['cases']
              if row['backend']=='gpu' and row['tolerance']=='baseline' and row['history']=='off'}
    observed={}
    original=performance.budgeted_launch
    current=[]

    def checked(options,backend,ranks,name,steps,**kwargs):
        case,size=original(options,backend,ranks,name,steps,**kwargs)
        if name=='verification':
            label=f'{backend}_np{ranks}'
            previous=old_rows[label]
            assert previous['error']['within_reference_margin'] and previous['error']['within_candidate_error']
            relative='outdat/new/checkpoints/step000000000024/state.h5'
            datasets=compare_fields(case/relative,Path(previous['state']))
            initial=compare_fields(kwargs['restore']/'state.h5',
                Path(previous['case'])/'outdat/new/checkpoints/step000000000000/state.h5')
            observed[f'{current[0]}_{label}']=dict(
                qualified_control_report_sha256=digest(source),
                qualified_state=str(previous['state']),qualified_state_sha256=digest(Path(previous['state'])),
                exact_final_datasets=datasets,exact_initial_datasets=initial,
                actual_error_reused_by_bitwise_q_and_carry=previous['error'])
            (tmp_path/'g2_reuse.json').write_text(json.dumps(observed,indent=2,allow_nan=False)+'\n')
        return case,size

    # Compare with qualified physical states before the first performance round,
    # without another identical 64^3 trajectory or another reference inversion.
    monkeypatch.setattr(performance,'budgeted_launch',checked)
    executable=arguments(tmp_path).executable
    settings=[('frozen',REFERENCE,'off','baseline','pinned'),
        ('baseline',executable,'off','baseline','pinned'),
        ('reuse',executable,'chemistry','baseline','pinned'),
        ('packed',executable,'off','packed','pinned'),
        ('combined',executable,'chemistry','packed','pinned'),
        ('device-aware',executable,'chemistry','packed','device-aware')]
    result={}
    for label,binary,reuse,reductions,transport in settings:
        current[:]=[label]
        prior=os.environ.get('ASTR_AIR5_FROZEN_TIMING') if label=='frozen' else None
        if prior:
            prior=Path(prior).resolve(strict=True)
            saved=json.loads(prior.read_text())
            expected=dict(status='bounded-timing-collected',executable_sha256=digest(REFERENCE),
                warmup=4,measured_steps=20,repeats=5,grid_intervals=[64,64,64],dt=1e-10,
                case='sbli',timing_start_step=0,profiled=False,phase_timing=False,
                chemistry_diagnostics='report-steps',source_evaluation='source-only',
                primitive_reuse='off',chemistry_reductions='baseline',halo_transport='pinned',
                chemistry_tolerance='baseline',step_history='off',chemistry_block_size=32,
                axis='x',topology=None,top_mode='characteristic',filter_workspace='scalar',
                air5_filter_mode='conservative-flux')
            assert all(saved[key]==value for key,value in expected.items())
            old_checks=json.loads((prior.parents[1]/'g2_reuse.json').read_text())
            for key in ('gpu_np1','gpu_np2'):
                row=saved['cases'][key]
                assert len(row['rounds'])==5
                assert performance.batch_identity(Path(row['seed']))==row['seed_files_sha256']
                proof=old_checks[f'frozen_{key}']
                assert proof['qualified_control_report_sha256']==digest(source)
                assert proof['qualified_state_sha256']==digest(Path(proof['qualified_state']))
                observed[f'frozen_{key}']=proof
            result[label]=dict(saved,reused_from=str(prior),reused_summary_sha256=digest(prior))
            (tmp_path/'all_timings.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            (tmp_path/'g2_reuse.json').write_text(json.dumps(observed,indent=2,allow_nan=False)+'\n')
            continue
        args=SimpleNamespace(output=tmp_path/label,executable=binary,mpiexec=arguments(tmp_path).mpiexec,
            nsys=False,phase_timing=False,repeats=5,warmup=4,steps=20,report_interval=4,
            case='sbli',chemistry_diagnostics='report-steps',source_evaluation='source-only',
            chemistry_tolerance='baseline',step_history='off',primitive_reuse=reuse,
            chemistry_reductions=reductions,halo_transport=transport,chemistry_block_size=32,
            axis='x',topology=None,top_mode='characteristic',filter_workspace='scalar',
            compare_to=tmp_path/'baseline' if label not in ('frozen','baseline') else None,
            cases=['gpu_np1','gpu_np2'],grid=(64,64,64),host_budget_bytes=256*1024**2,
            device_budget_bytes=256*1024**2,directory_budget_bytes=512*1024**2,
            start_from_initial=True,air5_filter_mode='conservative-flux',flow_control=None,
            campaign_root=campaign,campaign_budget_bytes=64*1024**3,runtime_timeout_seconds=1800)
        performance.run(args)
        result[label]=json.loads((args.output/'timing_summary.json').read_text())
        (tmp_path/'all_timings.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert len(observed)==12


def test_private_transfer_profile(tmp_path):
    args=engineering(tmp_path,(2,1,1))
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    args.mean_statistics=False
    current=args.executable
    index={}
    for label,binary,transport in [('frozen',REFERENCE,'pinned'),
                                   ('pinned',current,'pinned'),('device-aware',current,'device-aware')]:
        args.executable=binary
        args.halo_transport=transport
        case,_=launch(args,'gpu',2,label,2,checkpoint_enabled=False,nsys_trace=True,
                      chemistry_diagnostics='report-steps')
        databases=sorted(case.glob('trace.rank*.sqlite'))
        assert len(databases)==2
        index[label]=dict(executable_sha256=digest(binary),transport=transport,
                          sqlite_files=[str(path) for path in databases],
                          attribution_only_not_performance=True)
        (tmp_path/'profile_index.json').write_text(json.dumps(index,indent=2)+'\n')


def private_transfer_summary(index_path):
    """Attribute GPU copy records to completed private kernel windows, not wire traffic."""
    index_path=Path(index_path)
    index=json.loads(index_path.read_text())
    report=dict(scope='two profiled 16^3 steps, NP2 x; not performance or network wire volume',cases={})
    for label,row in index.items():
        ranks=[]
        for path in row['sqlite_files']:
            with sqlite3.connect(path) as database:
                names=dict(database.execute('SELECT id,value FROM StringIds'))
                kinds={key:value for key,_,value in database.execute('SELECT * FROM ENUM_CUDA_MEMCPY_OPER')}
                kernels=[(start,end,names[name]) for start,end,name in database.execute(
                    'SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start')]
                copies=list(database.execute('SELECT start,end,bytes,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'))
                allocations=[(start,end,names[name]) for start,end,name in database.execute(
                    'SELECT start,end,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME')
                    if any(part in names[name] for part in ('MemAlloc','MemHostAlloc','HostRegister'))]
            ranges=dict(filter=[],symmetric_faces=[])
            filter_start=face_start=None
            for start,end,name in kernels:
                if name.endswith('_filter_pack_gpu_') and filter_start is None:
                    filter_start=start
                elif name.endswith('_filter_check_state_gpu_'):
                    assert filter_start is not None
                    ranges['filter'].append((filter_start,end))
                    filter_start=None
                elif name.endswith('_air5_symmetric_plane_kernel_'):
                    if face_start is None:
                        face_start=end
                    else:
                        ranges['symmetric_faces'].append((face_start,start))
                        face_start=None
            assert filter_start is None and face_start is None
            assert len(ranges['filter'])==18 and len(ranges['symmetric_faces'])==30
            phases={}
            for phase,windows in ranges.items():
                selected=[copy for copy in copies if any(lo<=copy[0] and copy[1]<=hi for lo,hi in windows)]
                amounts={}
                for start,end,size,kind in selected:
                    item=amounts.setdefault(kinds[kind],dict(records=0,bytes=0,device_seconds=0.))
                    item['records']+=1; item['bytes']+=size; item['device_seconds']+=(end-start)*1e-9
                calls=[name for start,end,name in allocations
                       if any(lo<=start and end<=hi for lo,hi in windows)]
                phases[phase]=dict(windows=len(windows),copies=amounts,
                    allocation_or_registration_calls=calls,
                    host_device_bytes=sum(size for _,_,size,kind in selected if kind in (1,2)))
            ranks.append(dict(database=str(path),phases=phases,
                host_device_bytes=sum(phase['host_device_bytes'] for phase in phases.values())))
        report['cases'][label]=dict(ranks=ranks,
            host_device_bytes=sum(rank['host_device_bytes'] for rank in ranks))
    destination=index_path.with_name('transfer_summary.json')
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


@pytest.mark.parametrize('topology,prior', [((2,2,1),'top0'),((2,1,2),'top1'),((1,2,2),'top2')])
def test_padding_change_reuses_qualified_multiaxis_cpu(tmp_path,topology,prior):
    args=engineering(tmp_path,topology,workspace='full',history='previous')
    args.grid=(32,)*3
    args.share_gpus=True
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    args.halo_transport='device-aware'
    case,_=launch(args,'gpu',4,'current',12,chemistry_diagnostics='report-steps')
    original=ROOT/('tests/gpu_validation/out/air5_g2_64_20261013/le3_topologies_20261018_r2/'
                   f'test_integrated_topologies_{prior}')
    exact={name:compare_fields(case/FINAL/name,original/'gpu_np4_full'/FINAL/name)
           for name in ('state.h5','statistics.h5')}
    errors=field_errors(original/'cpu_np4_whole'/FINAL/'state.h5',case/FINAL/'state.h5')
    evidence=dict(topology=topology,executable_sha256=digest(args.executable),
        immutable_prior=str(original),exact_prior_gpu_datasets=exact,cpu_gpu_errors=errors,
        density_padding_nodes=density_padding(case/FINAL/'state.h5'))
    (tmp_path/'multiaxis_reuse.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')


@pytest.mark.parametrize('topology,prior', [((2,1,1),'top1'),((1,2,1),'top2')])
def test_final_pinned_restart(tmp_path,topology,prior):
    args=engineering(tmp_path,topology,history='previous')
    args.primitive_reuse='chemistry'
    args.chemistry_reductions='packed'
    whole,_=launch(args,'gpu',2,'whole',12,chemistry_diagnostics='report-steps')
    first,_=launch(args,'gpu',2,'first',5,chemistry_diagnostics='report-steps')
    args.halo_transport='device-aware'
    resumed,_=launch(args,'gpu',2,'resumed',12,
        restore=first/'outdat/new/checkpoints/step000000000005',chemistry_diagnostics='report-steps')
    exact={name:compare_fields(whole/FINAL/name,resumed/FINAL/name)
           for name in ('state.h5','statistics.h5')}
    original=ROOT/('tests/gpu_validation/out/air5_g2_64_20261013/le3_topologies_20261018/'
                   f'test_integrated_topologies_{prior}/cpu_np2_whole')
    errors=field_errors(original/FINAL/'state.h5',whole/FINAL/'state.h5')
    (tmp_path/'final_restart.json').write_text(json.dumps(dict(topology=topology,
        executable_sha256=digest(args.executable),exact_datasets=exact,cpu_reference=str(original),
        cpu_gpu_errors=errors,history=history_summary(first/'outdat/new/checkpoints/step000000000005/state.h5'),
        density_padding_nodes=density_padding(whole/FINAL/'state.h5')),indent=2,allow_nan=False)+'\n')


@pytest.mark.parametrize('topology,prior', [((2,1,1),'topol0'),((1,2,1),'topol1'),((1,1,2),'topol2')])
def test_final_pinned_faces(tmp_path,topology,prior):
    args=engineering(tmp_path,topology)
    args.mean_statistics=False
    case,_=launch(args,'gpu',2,'pinned',1,symmetric_face_probe=True,checkpoint_enabled=False)
    checks=check_shared_face_payloads(case,topology,periodic=(False,False,True))
    original=ROOT/('tests/gpu_validation/out/air5_g2_64_20261013/le3_final_native_20261018/'
                   f'test_owned_face_payloads_{prior}/gpu_np2_frozen/validation')
    for path in original.glob('air5*shared_faces*.bin'):
        assert path.read_bytes()==(case/'validation'/path.name).read_bytes()
    (tmp_path/'final_faces.json').write_text(json.dumps(dict(topology=topology,checks=checks,
        executable_sha256=digest(args.executable),reference=str(original)),indent=2)+'\n')


def test_final_pinned_medium_window(tmp_path,monkeypatch):
    original=development.prepare_window
    def prepare(case,variant,steps,preflight):
        runtime,record=original(case,variant,steps,preflight)
        runtime.update(ASTR_AIR5_PRIMITIVE_REUSE='chemistry',ASTR_AIR5_CHEMISTRY_REDUCTIONS='packed')
        record['local_extension']=dict(primitive_reuse='chemistry',chemistry_reductions='packed')
        return runtime,record
    monkeypatch.setattr(development,'prepare_window',prepare)
    monkeypatch.setattr(development,'BUDGET',64*1024**3)
    campaign=ROOT/'tests/gpu_validation/out/air5_g2_64_20261013'
    case=tmp_path/'pinned'
    executable=arguments(tmp_path).executable
    result=development.run_case(case,executable,'combined',100,True,campaign)
    previous=campaign/'le3_mach4_20261018/test_mach4_hundred_step_integr0/pinned'
    result['exact_datasets']=compare_fields(Path(result['final_state']),
        previous/'outdat/new/checkpoints/step000000000100/state.h5')
    for name in ('air5_wall.dat','air5_profiles.dat'):
        assert (case/'monitor'/name).read_bytes()==(previous/'monitor'/name).read_bytes()
    result['executable_sha256']=digest(executable)
    (tmp_path/'final_medium.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
