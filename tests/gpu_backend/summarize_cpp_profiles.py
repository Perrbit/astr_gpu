"""Attribute native new/old timings from exported Nsight SQLite and CSV data."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3


def summarize(root, name):
    path = root/(name+'_trace.sqlite')
    with sqlite3.connect(path) as database:
        operations = dict(database.execute('select id,label from ENUM_CUDA_DEV_MEM_EVENT_OPER'))
        device_kind, = database.execute("select id from ENUM_CUDA_MEM_KIND where label='Device'").fetchone()
        static_kind, = database.execute("select id from ENUM_CUDA_MEM_KIND where label='Device Static'").fetchone()
        total, peak = 0, 0
        for size, kind in database.execute('select bytes,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS '
                                           'where memKind=? order by start', (device_kind,)):
            if operations[kind] == 'Allocation':
                total += size
            elif operations[kind] == 'Deallocation':
                total -= size
            else:
                raise AssertionError('unhandled allocation operation')
            peak = max(peak, total)
        if total != 0:
            raise AssertionError('captured CUDA allocation/free events do not close')
        static_bytes, = database.execute('select coalesce(sum(bytes),0) from CUDA_GPU_MEMORY_USAGE_EVENTS '
                                          'where memKind=? and memoryOperationType=0', (static_kind,)).fetchone()
    with (root/(name+'_trace_cuda_gpu_kern_sum.csv')).open() as stream:
        kernels = list(csv.DictReader(stream))
    result = dict(cuda_api_visible_peak_dynamic_device_bytes=peak, registered_static_device_bytes=static_bytes,
                  kernel_ns=sum(int(row['Total Time (ns)']) for row in kernels))
    names = ({'convection': 'divergence_kernel<(bool)0>', 'viscous_divergence': 'divergence_kernel<(bool)1>'}
             if name == 'cpp' else
             {'convection': 'solver_gpu_convective_rhs_', 'viscous_divergence': 'solver_gpu_diffusion_rhs_'})
    for label, pattern in names.items():
        matching = [row for row in kernels if pattern in row['Name']]
        count = sum(int(row['Instances']) for row in matching)
        if count != 72:
            raise AssertionError(f'{name}/{label}: expected 8 steps x 3 RK stages x 3 axes')
        result[label] = dict(instances=count, device_ns=sum(int(row['Total Time (ns)']) for row in matching))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = {name: summarize(args.directory, name) for name in ('cpp', 'cuda_fortran')}
    report['scope'] = '128^3, 8 steps, NP=1, scalar, pinned, explicit; profile times are not benchmark windows'
    report['memory_scope'] = 'CUPTI dynamic device allocation peak and registered static bytes separately; not driver/process/system-wide peak'
    (args.directory/'attribution.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
