"""IS8 bounded same-product timing/resource observation; not a production benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import h5py
import numpy as np

from run_insitu_acceptance_timing import read_timing
from run_output_restart_validation import run_case,compare_fields
from test_output_insitu_restart import ROOT,EXE,arguments,configuration,check_render
from test_insitu_device_products import device_configuration,check_device_render


def face_usage(case,ranks,mode):
    rows=re.findall(r'ASTR_INSITU_FACE_USAGE rank=(\d+) step=(\d+) generation=(\d+) '
        r'host_bytes=(\d+) device_bytes=(\d+) cumulative_seconds_d2h_mpi_h2d_local_sync=([^\n]+)',
        (case/'run.log').read_text())
    assert len(rows)==ranks*2,rows
    result=[]
    for rank,step,generation,host,device,values in rows:
        seconds=list(map(float,values.split()))
        assert len(seconds)==5 and all(np.isfinite(v) and v>=0 for v in seconds)
        assert int(generation)==1 and int(step) in (2,4) and int(rank)<ranks
        assert int(device)==219024
        assert int(host)==(219024 if mode=='pinned' and ranks==2 else 0)
        if mode!='pinned' or ranks==1:
            assert seconds[0]==seconds[2]==0
        result.append(dict(rank=int(rank),step=int(step),generation=int(generation),
            host_bytes=int(host),device_bytes=int(device),
            cumulative_seconds=dict(zip(('d2h','mpi_inclusive','h2d','local_d2d','receive_sync'),seconds))))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace',action='store_true',help='Capture separate NP2 all-product device traces instead of timing')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope='32^3 TGV four full steps, dt=1e-3, NP1/2 x, images at steps2/4',
        pairs=[],traces=[],directory_budget_bytes=256*1024**2,
        provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (EXE,ROOT/'scripts/insitu/tgv_pipeline.py')},
        nesting='Each duration is max of per-rank accumulated local durations, not sum across ranks or nested phases. '
            'Face MPI includes error-consensus collectives; consumer includes extraction, compact read, Catalyst and rendering. '
            'Initialization is separate from four-step window; lazy first-frame setup is inside consumer.',
        comparison='Products, schedules, mean coverage, physics, cameras and files match. Host uses original VTK accepted-length '
            'counter; device uses approved corrected counter, so streamline vertex counts may differ. Do not attribute the '
            'entire wall difference solely to volume transfers.')
    try:
        for ranks in ((2,) if args.trace else (1,2)):
            local=arguments(output);local.directory_budget_bytes=256*1024**2
            local.runtime_timeout_seconds=300
            configs={mode:(device_configuration(mode,interval=2) if mode in ('pinned','device-aware') else
                configuration(render=mode!='off',interval=2).replace('render=.true.,',
                    "render=.true.,derivative_backend='gpu',"))
                .replace('statistics_window=0.0005,0.0115','statistics_window=0.003,0.004')
                for mode in ('off','host','pinned','device-aware')}
            cases={};times={};sizes={};resources={};slots={}
            for mode in (('pinned','device-aware') if args.trace else configs):
                case,size=run_case(local,ROOT,'gpu',ranks,'trace_'+mode if args.trace else 'timing_'+mode,4,
                    grid='32,32,32',checkpoint_interval=2,insitu_config=configs[mode],
                    postprocess_transport=mode if mode in ('pinned','device-aware') else None,
                    insitu_timing=True,nsys_trace=args.trace,monitor_resources=not args.trace,
                    resource_baseline=resources.get('off'))
                cases[mode]=case;sizes[mode]=size
                if mode in ('pinned','device-aware'):
                    check_device_render(case,ranks,[2,4],mean_steps=[4]);slots[mode]=face_usage(case,ranks,mode)
                elif mode=='host':check_render(case,ranks,[2,4])
                if args.trace:
                    paths=sorted(case.glob('trace.rank*.sqlite'))
                    assert len(paths)==ranks,paths
                    report['traces'].append(dict(np=ranks,mode=mode,case=str(case),sqlite=list(map(str,paths))))
                else:
                    resources[mode]=json.loads((case/'resources.sampled.json').read_text())
                    times[mode]=read_timing(case,ranks)
                    if mode in ('pinned','device-aware'):
                        required={'device_sample_inclusive','device_mean_supply_inclusive','device_snapshot_inclusive',
                            'device_halo_inclusive','device_gradient_q_inclusive','device_consumer_inclusive',
                            'device_geometry_extract_inclusive','device_geometry_read','device_instant_lines_inclusive',
                            'device_crossing_lines_inclusive','device_reynolds_lines_inclusive','device_favre_lines_inclusive',
                            'extraction','render_inclusive','geometry_write_inclusive','image_encoding','image_publication'}
                        assert required<=times[mode].keys(),required-times[mode].keys()
                for step,count in ((2,4),(4,6)) if mode!='off' else ():
                    for rank in range(ranks):
                        products=json.loads((case/f'outdat/render/mesh_step{step:08d}_rank{rank}.json').read_text())['products']
                        assert len(products)==count and all(v['image']['status']=='published' for v in products.values())
            if not args.trace:
                for mode in ('host','pinned','device-aware'):
                    for name in ('input.tgv','input.output','controller'):
                        assert (cases['off']/'datin'/name).read_bytes()==(cases[mode]/'datin'/name).read_bytes()
                    for name in ('state.h5','statistics.h5'):
                        compare_fields(cases['off']/f'outdat/new/checkpoints/step000000000004/{name}',
                            cases[mode]/f'outdat/new/checkpoints/step000000000004/{name}')
                report['pairs'].append(dict(np=ranks,topology=[ranks,1,1],timing=times,
                    directory_bytes=sizes,resources=resources,face_slots=slots))
            (output/'report.json').write_text(json.dumps(report,indent=2))
        report['status']='passed-bounded-observation-not-production-certification'
    except BaseException as error:
        report.update(status='failed',error=str(error));raise
    finally:
        (output/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
