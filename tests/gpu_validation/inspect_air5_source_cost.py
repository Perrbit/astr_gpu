#!/usr/bin/env python3
"""Record bounded compiled evidence for U5; static counts are not runtime cost."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from run_air5_performance_diagnosis import digest


def function_body(ptx, name):
    body = ptx[ptx.index('.visible .func ' + name):]
    end = re.search(r'(?m)^\.visible \.(?:func|entry) ', body[1:])
    return body[:end.start() + 1] if end else body


def inspect(tool, kinetics_object, solver, pure_path=False):
    source_name = 'chemistry_source_gpu_air5_instantaneous_source_gpu_'
    kernel_name = 'chemistry_coupling_gpu_air5_chemistry_half_step_kernel_'
    command = [str(tool), '--dump-ptx', '--function', source_name, str(kinetics_object)]
    ptx = subprocess.check_output(command, text=True)
    body = function_body(ptx, source_name)
    calls = re.findall(r'(?m)^(chemistry_\w+_),\s*$', body)
    derivative_calls = sorted({name for name in calls if 'derivative' in name})
    if not derivative_calls:
        raise ValueError('no retained derivative calls; inspect compiler elimination before splitting')
    resources_command = [str(tool), '--dump-resource-usage', '--function', kernel_name, str(solver)]
    resources = subprocess.check_output(resources_command, text=True)
    match = re.search(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)', resources)
    if not match:
        raise ValueError('missing linked chemistry kernel resource evidence')
    result = dict(status='compiled-derivative-work-retained', commands=[command, resources_command],
        kinetics_object_sha256=digest(kinetics_object), solver_sha256=digest(solver),
        source_function=source_name, retained_derivative_callees=derivative_calls,
        source_local_arrays=re.findall(r'(?m)^\.local .*$' , body),
        linked_kernel=dict(name=kernel_name, **dict(zip(('registers', 'stack_bytes', 'shared_bytes', 'local_bytes'),
            map(int, match.groups())))),
        interpretation='PTX retains tangent routines, including their status-dependent checks. '
            'STACK is per-thread frame storage, not a measured spill traffic counter; '
            'LOCAL=0 does not imply that the stack is free or that occupancy is the only bottleneck.')
    if pure_path:
        name = 'chemistry_source_gpu_air5_pure_source_gpu_'
        command = [str(tool), '--dump-ptx', '--function', name, str(kinetics_object)]
        body = function_body(subprocess.check_output(command, text=True), name)
        calls = sorted(set(re.findall(r'(?m)^(chemistry_\w+_),\s*$', body)))
        derivatives = [callee for callee in calls if 'derivative' in callee]
        if derivatives:
            raise ValueError(f'pure source still calls tangent routines: {derivatives}')
        result.update(status='optional-pure-path-no-tangent-callees-joint-path-retained',
            pure_source_command=command, pure_source_function=name, pure_retained_callees=calls,
            pure_retained_derivative_callees=derivatives,
            pure_local_arrays=re.findall(r'(?m)^\.local .*$', body))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuobjdump', type=Path, required=True)
    parser.add_argument('--object', type=Path, required=True)
    parser.add_argument('--solver', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pure-path', action='store_true')
    args = parser.parse_args()
    result = inspect(args.cuobjdump, args.object, args.solver, args.pure_path)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
