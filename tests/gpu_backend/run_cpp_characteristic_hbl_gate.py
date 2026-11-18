"""Actual resident MP3-HBL1 advancement; final fields and initial sensor only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import h5py
import numpy as np

from run_cpp_tgv_gate import compare,logged_steps,cfl_log
from run_cpp_multi_tgv_gate import layouts
from prepare_characteristic_hbl_reference import read_volume,conservative,positive,read_sensor_dump,FIELD_NAMES,FIELD_ATOL,STAT_ATOL,SENSOR_ATOL


def absolute(actual,expected):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError('HBL comparison shape/finiteness')
    return np.max(abs(actual-expected),axis=tuple(range(actual.ndim-1)))


def sensors(case,expected,ranks):
    maximum=0.;mismatches=0;pieces=[];partitions=layouts(case)
    for rank in range(ranks):
        path=case/'outdat'/('initial_sensor.dat' if ranks==1 else f'initial_sensor.dat.rank{rank}')
        current=read_sensor_dump(path);pieces.append(current)
        if current.offset!=tuple(partitions[rank]['origin']) or current.shape!=tuple(n+1 for n in partitions[rank]['upper']):
            raise AssertionError('HBL sensor dump does not match its MPI partition')
        selected=tuple(slice(o,o+n) for o,n in zip(current.offset,current.shape))
        maximum=max(maximum,float(np.max(abs(current.sensor-expected.sensor[selected]))))
        mismatches+=int(np.count_nonzero(current.mask!=expected.mask[selected]))
    if maximum>SENSOR_ATOL or mismatches:raise AssertionError('frozen HBL initial-sensor contract')
    return dict(port_max_abs=maximum,mask_mismatches=mismatches),pieces


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','reference','mpiexec','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--ranks',type=int,choices=(1,2),default=1)
    parser.add_argument('--axis',choices=('x','y'),default='x')
    parser.add_argument('--transport',choices=('pinned','device'),default='pinned')
    parser.add_argument('--sync',choices=('explicit','dependency'),default='explicit')
    parser.add_argument('--sanitizer',choices=('memcheck','initcheck'))
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binary=args.binary.resolve();reference=args.reference.resolve()
    shutil.copyfile(__file__,out/'runner.py');shutil.copyfile(binary,out/'tested_astr')
    topology=[1,1,1];topology['xy'.index(args.axis)]=args.ranks
    env={key:value for key,value in os.environ.items() if not key.startswith('ASTR_VALIDATION_')}
    env.update(OMPI_MCA_pml='ob1',OMPI_MCA_btl='self,vader,tcp',OMPI_MCA_osc='pt2pt',
        OMPI_MCA_coll='^cuda,hcoll,ucc',OMPI_MCA_opal_cuda_support='0',
        ASTR_GPU_HALO_TRANSPORT=args.transport,ASTR_GPU_SYNC_MODE=args.sync,
        ASTR_FORCE_MPI_TOPOLOGY=','.join(map(str,topology)),ASTR_GPU_MP_POSITIVITY='off')
    if args.transport=='device':env.update(OMPI_MCA_pml='ucx',OMPI_MCA_opal_cuda_support='1')
    report=dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),topology=topology,
        transport=args.transport,sync=args.sync,field_atol=FIELD_ATOL,statistics_atol=STAT_ATOL,sensor_atol=SENSOR_ATOL,
        scope='192x192x8 controlled MP3-HBL1, two fixed dt1e-5 steps; no checkpoint or steady physics claim',cases={})
    def run(name,mixed,dump=True):
        case=out/name;case.mkdir()
        previous=reference/('old_gpu_mixed' if mixed else 'old_gpu_fp64')
        shutil.copytree(previous/'datin',case/'datin',copy_function=shutil.copyfile);(case/'datin').chmod(0o755)
        (case/'outdat/new').mkdir(parents=True)
        command=[str(args.mpiexec.resolve()),'--bind-to','none','-n',str(args.ranks)]
        if args.sanitizer and name=='mixed':
            command+=['/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/compilers/bin/compute-sanitizer','--tool',args.sanitizer,
                '--error-exitcode','99','--log-file',str(case/'sanitizer.%p.log')]
            if args.sanitizer=='memcheck':command+=['--leak-check','full']
        with (case/'run.log').open('w') as log:
            subprocess.run(command+[str(binary),'run'],cwd=case,
                env=env|dict(ASTR_SHOCK_SENSOR_DUMP='outdat/initial_sensor.dat' if dump else ''),
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        text=(case/'run.log').read_text()
        if text.count('backend=cuda')!=args.ranks or ('ASTR_GPU_CPP mixed_workspace=characteristic_flux' in text)!=mixed:
            raise AssertionError('actual C++ HBL precision/backend selection')
        if list((case/'outdat/new/checkpoints').glob('step*')):raise AssertionError('HBL gate must not checkpoint')
        if list((case/'outdat').glob('rhs.*')):raise AssertionError('HBL gate must not write per-stage field traces')
        return case
    fields={};statistics={};cases={};initial={}
    for mode in ('fp64','mixed'):
        mixed=mode=='mixed';case=run(mode,mixed);cases[mode]=case
        previous=reference/('old_gpu_mixed' if mixed else 'old_gpu_fp64')
        p=read_volume(case);positive(p);old=read_volume(previous)
        q=conservative(p);oldq=conservative(old)
        scalar=np.loadtxt(case/'flowstate.dat',skiprows=1,ndmin=2)
        old_scalar=np.loadtxt(previous/'flowstate.dat',skiprows=1,ndmin=2)
        delta_p=absolute(p,old);delta_q=absolute(q,oldq);delta_stat=absolute(scalar,old_scalar)
        if not np.array_equal(scalar[:,:2],old_scalar[:,:2]):raise AssertionError('HBL sample identity')
        if mixed:
            if max(delta_p)>FIELD_ATOL or max(delta_q)>FIELD_ATOL or max(delta_stat)>STAT_ATOL:
                raise AssertionError('matching-original HBL candidate field/statistics gates')
        else:
            compare(p,old);compare(q,oldq);compare(scalar,old_scalar)
            report['cpu_fp64_field_scaled']=compare(p,read_volume(reference/'cpu'))
        sensor_report,initial[mode]=sensors(case,read_sensor_dump(previous/'outdat/initial_sensor.dat'),args.ranks)
        if not np.array_equal(logged_steps(case,2),logged_steps(previous,2)):raise AssertionError('HBL complete-step clocks')
        cfl=cfl_log(case,2)
        if not np.isfinite(cfl).all():raise AssertionError('HBL CFL finite')
        report['cases'][mode]=dict(matching_old_primitive_abs=delta_p.tolist(),matching_old_q_abs=delta_q.tolist(),
            matching_old_scalar_abs=delta_stat.tolist(),sensor=sensor_report,cfl=cfl.tolist())
        fields[mode],statistics[mode]=p,scalar
    delta_p=absolute(fields['mixed'],fields['fp64'])
    delta_q=absolute(conservative(fields['mixed']),conservative(fields['fp64']))
    delta_stat=absolute(statistics['mixed'],statistics['fp64']);raw=0.
    for first,second in zip(initial['fp64'],initial['mixed']):
        if first.offset!=second.offset or not np.array_equal(first.mask,second.mask):raise AssertionError('HBL candidate masks')
        raw=max(raw,float(np.max(abs(first.sensor-second.sensor))))
    if max(delta_p)>FIELD_ATOL or max(delta_q)>FIELD_ATOL or max(delta_stat)>STAT_ATOL or raw>SENSOR_ATOL:
        raise AssertionError('new HBL candidate exceeds the original frozen precision gates')
    quiet=run('mixed_no_initial_dump',True,dump=False)
    if list((quiet/'outdat').glob('initial_sensor.dat*')):raise AssertionError('disabled initial observer still wrote files')
    if fields['mixed'].tobytes()!=read_volume(quiet).tobytes():raise AssertionError('initial dump changed published HBL fields')
    if not np.array_equal(statistics['mixed'],np.loadtxt(quiet/'flowstate.dat',skiprows=1,ndmin=2)) or \
        not np.array_equal(logged_steps(cases['mixed'],2),logged_steps(quiet,2)):
        raise AssertionError('initial dump changed HBL scalar values or clocks')
    if args.sanitizer:
        paths=list(cases['mixed'].glob('sanitizer.*.log'))
        if len(paths)!=args.ranks or any('ERROR SUMMARY: 0 errors' not in path.read_text() for path in paths):
            raise AssertionError('HBL main sanitizer')
        if args.sanitizer=='memcheck' and any('LEAK SUMMARY: 0 bytes leaked' not in path.read_text() for path in paths):
            raise AssertionError('HBL main device leak')
    report.update(candidate_primitive_abs=delta_p.tolist(),candidate_q_abs=delta_q.tolist(),
        candidate_scalar_abs=delta_stat.tolist(),candidate_initial_sensor_abs=raw,
        initial_dump_field_scalar_clock_bits_unchanged=True,checkpoint_tested=False,dcu_tested=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(topology=topology,transport=args.transport,primitive=max(delta_p),
        conservative=max(delta_q),statistics=max(delta_stat),initial_sensor=raw,mask_mismatches=0),indent=2))


if __name__=='__main__':main()
