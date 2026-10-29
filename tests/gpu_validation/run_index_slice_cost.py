"""Local moderate-M12 numerical-slice cost report; no rendering or full-field archive."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from run_m12_local_compare import run

ROOT = Path(__file__).resolve().parents[2]


def product(name, fields, i=(0,96), j=(0,48)):
    return dict(id=name,kind='slice',enable_request=1,source='instantaneous',fields=fields,
        sampling=dict(mode='grid_index',axis='k',index=16,index_ranges=dict(i=list(i),j=list(j))),
        output=dict(channel='data',format='hdf5-xdmf'),schedule=dict(mode='steps',interval=2))


def summarize(case):
    text = (case/'run.log').read_text()
    timings = {}
    for stage,step,rank,value in re.findall(r'ASTR_INSITU_STAGE_TIMING\s+(\S+)\s+(\d+)\s+(\d+)\s+(\S+)',text):
        key = (stage,int(step),int(rank))
        timings[key] = timings.get(key,0.) + float(value)
    stages = {}
    for stage in sorted({key[0] for key in timings}):
        steps = sorted({key[1] for key in timings if key[0]==stage and key[1]>0})
        values = [max(value for (label,s,_),value in timings.items() if label==stage and s==step) for step in steps]
        stages[stage] = dict(per_step_rank_max_seconds=values,total_seconds=sum(values),
                            median_seconds=float(np.median(values)) if values else None)
    packets = [dict(rank=int(rank),id=name,step=int(step),neighborhood_bytes=int(work),
        field_d2h_bytes=int(download),private_exchange_bytes=int(exchange)) for rank,name,step,work,download,exchange in
        re.findall(r'ASTR_SLICE rank=(\d+) id=(\S+) step=(\d+) neighborhood_bytes=(\d+) field_d2h_bytes=(\d+) '
                   r'private_exchange_bytes=(\d+) diagnostic_seconds=\s*\S+',text)]
    archive = case/'outdat/new/slice_products'
    resources = json.loads((case/'resources.sampled.json').read_text())
    return dict(case=str(case),stages=stages,packets=packets,
        field_d2h_total_bytes=sum(row['field_d2h_bytes'] for row in packets),
        private_strip_send_receive_bytes=sum(row['private_exchange_bytes'] for row in packets),
        max_rank_neighborhood_bytes=max((row['neighborhood_bytes'] for row in packets),default=0),
        archive_bytes=sum(path.stat().st_size for path in archive.rglob('*') if path.is_file()),
        archive_files=sum(path.is_file() for path in archive.rglob('*')),
        sampled_host_peak_bytes=resources['host_rss_peak_bytes'],
        sampled_device_peaks={key:value['peak_bytes'] for key,value in resources['devices'].items()},
        sampling_period_seconds=resources['sampling_period_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',type=Path,default=ROOT/'build_m12_gpu/bin/astr')
    parser.add_argument('--source',type=Path,default=Path('/home/dell/workspace/M12_C13'))
    parser.add_argument('--mpiexec',type=Path,default=Path('/opt/nvidia/hpc_sdk/Linux_x86_64/26.1/comm_libs/hpcx/bin/mpiexec'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.executable=args.executable.resolve(strict=True)
    args.intervals=(96,48,32);args.memcheck=False;args.diagnostics=False;args.capture_rhs=False
    args.halo_transport='pinned';args.insitu_write_eps='0';args.slice_timing=True
    basic=['velocity_x','pressure']
    derived=['density','velocity_x','velocity_y','velocity_z','pressure','temperature',
             'Q_rs','velocity_divergence','density_gradient_magnitude']
    configurations=[('off',None),('basic',[product('basic',basic)]),
        ('derived',[product('derived',derived)]),('region',[product('region',derived,(24,48),(0,16))]),
        ('two-products',[product('basic',basic),product('region',derived,(24,48),(0,16))])]
    report=dict(grid=list(args.intervals),topology=[2,1,1],steps=10,deltat=.02,
                rendering=False,checkpoints=False,full_field_output=False,cases=[])
    for name,definitions in configurations:
        case=run(args,'gpu','2,1,1',case_label=name,product_catalog_records=definitions)
        result=summarize(case);result['configuration']=name
        report['cases'].append(result)
        (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(summary=str(args.output/'summary.json'),cases=[dict(configuration=row['configuration'],
        complete_window_seconds=row['stages']['completed_window']['total_seconds'],
        field_d2h_total_bytes=row['field_d2h_total_bytes']) for row in report['cases']]),indent=2))


if __name__=='__main__':
    main()
