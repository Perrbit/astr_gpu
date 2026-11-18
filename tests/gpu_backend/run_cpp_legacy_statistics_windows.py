"""Reuse the established read-only input.run replay for C++ mean window changes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace

import h5py
import numpy as np

from run_cpp_insitu_gate import compare_hdf
from run_cpp_legacy_statistics_gate import compact_statistics
from run_cpp_tgv_gate import compare,checkpoint

VALIDATION = Path(__file__).resolve().parents[1] / "gpu_validation"
sys.path.insert(0, str(VALIDATION))
from run_adaptive_mean_statistics_gate import run, compare_plane,read_clocks
from run_adaptive_tgv_gate import compare_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("binary", "mpiexec", "replay-library", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--precision',choices=('fp64','fp32'),default='fp64')
    parser.add_argument('--freeze-only',action='store_true',help='freeze actual CPU/old GPU from a dual-runtime original executable')
    parser.add_argument('--handover',action='store_true',help='disable at the existing step4 sample/write event')
    parser.add_argument("--ranks", type=int, choices=(1, 2), default=1)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    args = parser.parse_args()
    if not args.freeze_only and args.reference is None:parser.error('--reference is required for C++ comparison')
    root = args.output.resolve()
    reference = args.reference.resolve() if args.reference else None
    dtype=np.float32 if args.precision=='fp32' else np.float64
    width=np.dtype(dtype).itemsize
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root / "runner.py")
    shutil.copyfile(args.binary.resolve(), root / "tested_astr")
    options = SimpleNamespace(output=root, analyze_only=False, case="flatplate", lifecycle=True, handover=args.handover,
        cpu=args.binary.resolve(),gpu=args.binary.resolve(), mpiexec=args.mpiexec.resolve(), replay_library=args.replay_library.resolve())
    topology = [1, 1, 1]
    topology["xyz".index(args.axis)] = args.ranks
    token = ",".join(map(str, topology))
    final = Path("outdat/new/checkpoints/step000000000010")
    def clock_word(words,index):
        value=np.asarray(words[index:index+1],np.int64)
        return float(value.astype(np.int32).view(np.float32)[0] if width==4 else value.view(np.float64)[0])
    def exact_continuation(current,restored,clocks,tail):
        for name in ('state.h5','statistics.h5'):
            if args.handover and name=='statistics.h5':
                if (current/final/name).exists() or (restored/final/name).exists():
                    raise AssertionError('disabled handover serialized statistics at step10')
                continue
            compare_hdf(current/final/name,restored/final/name,exact=True)
        for name in ('control.bin','archives.bin','insitu_control.bin'):
            if (current/final/name).read_bytes()!=(restored/final/name).read_bytes():
                raise AssertionError((name,'window exact restart'))
        if tail!=clocks[5:]:raise AssertionError('window continuation changed time/dt')
        for case,expected in ((current,[0,*range(1,11)]),(restored,[0,*range(6,11)])):
            observed=list(map(int,re.findall(r'ASTR_CONTROLLER_TEST_READ completed_step=(\d+)',(case/'run.log').read_text())))
            if observed!=expected:raise AssertionError(('normal runtime input read sequence',observed,expected))
            with h5py.File(case/final/'state.h5') as source:
                if source['q0001'].dtype.itemsize!=width:raise AssertionError('actual window numerical kind differs')
        if (current/'outdat/new/checkpoints/step000000000005/statistics.h5').exists():
            raise AssertionError('disabled statistics unexpectedly serialized at step 5')
    def published_on_disable(case,clocks,backend):
        path=case/('outdat/meanflow.h5' if backend=='cpu' else 'outdat/statistics.step000000000004.h5')
        with h5py.File(path) as source:
            if backend=='cpu':
                count=int(source['nsamples'][0]);step=int(source['nstep'][0])
                duration=float(source['mean_time_weight'][0])
            else:
                metadata=source['metadata'][:];identity=source['identity'][:]
                count=int(metadata[7]);step=int(identity[8]);duration=clock_word(metadata,13)
                if clock_word(identity,11)!=clocks[3][4]:raise AssertionError('deferred write lost dt_next')
            def finite(name,value):
                if isinstance(value,h5py.Dataset) and value.dtype.kind=='f' and not np.isfinite(value[:]).all():
                    raise AssertionError(('nonfinite handover statistic',name))
            source.visititems(finite)
        if count!=3 or step!=4 or duration!=clocks[3][2]:
            raise AssertionError('old window must publish endpoints0,2,4 before disabling')
        return dict(samples=count,publication_step=step,time_weight=duration)
    if args.freeze_only:
        report=dict(precision=args.precision,topology=topology,handover=args.handover,
            binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),cases={})
        for backend in ('cpu','gpu'):
            label=backend+f'_np{args.ranks}'
            current,clocks=run(options,backend,args.ranks,label,topology=token)
            restored,tail=run(options,backend,args.ranks,label+'_resumed',
                current/'outdat/new/checkpoints/step000000000005',topology=token)
            if 'backend=cuda' in (current/'run.log').read_text():
                raise AssertionError('new C++ cannot serve as the original reference')
            exact_continuation(current,restored,clocks,tail)
            if args.handover:
                report['cases'][backend]=dict(exact_state_controls=True,clocks=clocks,
                    publication=published_on_disable(current,clocks,backend))
                continue
            with h5py.File(current/final/'statistics.h5') as source:
                metadata=source['metadata'][:]
                compact=backend=='gpu'
                count=int(metadata[7 if compact else 1])
                duration=clock_word(metadata,13 if compact else 8)
                expected=dtype(0)
                for start,end in ((6,8),(8,10)):
                    expected=dtype(expected+dtype(dtype(clocks[end-1][2])-dtype(clocks[start-1][2])))
                if count!=3 or duration!=float(expected):raise AssertionError('original window count/coverage differs')
            report['cases'][backend]=dict(exact_state_statistics_controls=True,clocks=clocks,samples=count,time_weight=duration)
        errors={}
        compare_state(root/f'cpu_np{args.ranks}'/final/'state.h5',root/f'gpu_np{args.ranks}'/final/'state.h5',
            False,errors,max_scaled_error=None if width==4 else 2e-10)
        report['cpu_gpu_fields']=errors
        if not args.handover:
            report['cpu_normalized_means_and_stresses']=compare_plane(root/f'cpu_np{args.ranks}',root/f'gpu_np{args.ranks}',final,
                limit=None if width==4 else 2e-10)
        (root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        for p in root.rglob('*'):p.chmod(0o555 if p.is_dir() else 0o444)
        root.chmod(0o555)
        print(json.dumps(report,indent=2));return
    current, clocks = run(options, "gpu", args.ranks, "cpp", topology=token)
    restored, tail = run(options, "gpu", args.ranks, "cpp_resumed",
        current / "outdat/new/checkpoints/step000000000005", topology=token)
    exact_continuation(current,restored,clocks,tail)
    for case in (current,restored):
        text = (case / "run.log").read_text()
        if text.count("backend=cuda") != args.ranks:
            raise AssertionError("actual C++ main not selected")
    windows = {}
    for step, expected_count, last in (() if args.handover else ((9, 2, 8), (10, 3, 10))):
        path = current / f"outdat/statistics.step{step:012d}.h5"
        metadata, values = compact_statistics(path)
        weight = clock_word(metadata,13)
        if metadata[7] != expected_count or metadata[8] != 6 or metadata[9] != last:
            raise AssertionError(("window samples/reset boundary", metadata))
        expected_weight=dtype(0)
        for start,end in ((6,8),(8,10)):
            if end<=last:expected_weight=dtype(expected_weight+dtype(dtype(clocks[end-1][2])-dtype(clocks[start-1][2])))
        if weight != float(expected_weight):
            raise AssertionError("disabled interval or cadence change polluted time weight")
        if not np.isfinite(values).all() or not np.any(values[:, :, 1:13]):
            raise AssertionError("empty/nonfinite re-enabled mean window")
        with h5py.File(path) as source:
            identity = source["identity"][...]
            if source['q0001'].dtype.itemsize!=width:raise AssertionError('actual window storage kind differs')
            if identity[8] != step or clock_word(identity,9) != clocks[step-1][2]:
                raise AssertionError("sample versus publication clock confused")
        windows[str(step)] = dict(count=int(metadata[7]), first_step=int(metadata[8]), last_sample=int(metadata[9]),
                                 time_weight=weight)
    statistics_path=Path('outdat/statistics.step000000000004.h5') if args.handover else final/'statistics.h5'
    cm, c = compact_statistics(current / statistics_path)
    pm, p = compact_statistics(reference / "gpu_np1" / statistics_path)
    if not np.array_equal(cm[7:], pm[7:]) or clocks!=read_clocks(reference/'gpu_np1',False):
        raise AssertionError("frozen old-GPU and C++ window clocks differ")
    fields = {}
    for backend in ("cpu", "gpu"):
        details = {}
        compare_state(reference / (backend+"_np1") / final / "state.h5", current / final / "state.h5", False, details,
            max_scaled_error=None if width==4 else 2e-10)
        details['conservative_by_variable']=compare(checkpoint(current/final/'state.h5').astype(np.float64),
            checkpoint(reference/(backend+'_np1')/final/'state.h5').astype(np.float64),limit=None if width==4 else 2e-10)
        fields[backend] = details
    report = dict(binary=str(args.binary.resolve()), sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        topology=topology,precision=args.precision,handover=args.handover,windows=windows, field_references=fields,
        compact_scaled=compare(c, p,limit=None if width==4 else 2e-10),
        restart_while_disabled_exact=True, normal_runtime_parser_replayed=True)
    if args.handover:
        report['publication_on_disable']=published_on_disable(current,clocks,'gpu')
    else:
        report['cpu_normalized_means_and_stresses']=compare_plane(reference / "cpu_np1", current, final,
            limit=None if width==4 else 2e-10)
    (root / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
