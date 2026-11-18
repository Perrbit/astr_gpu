"""Device physical-end stages with real MPI, compared to frozen global trajectories."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np


def field_packet(path):
    with path.open('rb') as source:
        header = np.fromfile(source, np.int32, 9)
        values = np.fromfile(source, np.uint64)
    if header.size != 9 or values.size != int(np.prod(header[:3]+1))*5:
        raise AssertionError(f'{path}: final physical q packet layout')
    return header, values


def check_packets(directory, ranks, reference=None, walls=3):
    packets = sorted(directory.glob('*.bin'))
    if len(packets) != 4*walls*ranks:
        raise AssertionError(f'{directory}: missing final physical q observations')
    for packet in packets:
        header, values = field_packet(packet)
        peers = []
        if '_debug1.' in packet.name:
            peers.append(directory/packet.name.replace('_debug1.', '_debug0.'))
        if reference is not None:
            peers.append(reference/packet.name)
        for peer in peers:
            rh, rv = field_packet(peer)
            if not np.array_equal(header, rh) or not np.array_equal(values, rv):
                raise AssertionError(f'{packet}: sync/transport/storage changed physical q bits')
    return len(packets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--mpiexec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transport', choices=['pinned', 'device'], default='pinned')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--force-fixed', type=float, default=0.)
    parser.add_argument('--filter-workspace', choices=['off','scalar','full'], default='off')
    parser.add_argument('--compare-output', type=Path)
    parser.add_argument('--method', choices=['explicit','compact'], default='explicit')
    parser.add_argument('--compact-transport', choices=['pinned','device-aware'], default='pinned')
    parser.add_argument('--viscous-form', choices=['standard','me6_opti'], default='standard')
    parser.add_argument('--filter-method', choices=['explicit','compact'], help='defaults to the derivative method')
    parser.add_argument('--boundary-type',type=int,choices=[41,42,50,60],default=41)
    parser.add_argument('--convection-form',choices=['standard','keep'],default='keep')
    parser.add_argument('--wall-axes',nargs='+',choices=['x','y','z'],default=['x','y','z'])
    args = parser.parse_args()
    filter_method=args.filter_method or args.method
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ | dict(OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader,tcp', OMPI_MCA_osc='pt2pt',
                           OMPI_MCA_coll='^cuda,hcoll,ucc', OMPI_MCA_opal_cuda_support='0')
    if args.transport == 'device':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_opal_cuda_support='1')
    if args.compact_transport == 'device-aware':
        env.update(OMPI_MCA_pml='ucx', OMPI_MCA_opal_cuda_support='1')
    report = {}
    for ranks, axis in [(1, 1), (2, 1), (2, 2), (2, 3)]:
        name = f'np{ranks}_axis{axis}'
        dump = args.output/name
        dump.mkdir()
        command = [str(args.mpiexec), '--bind-to', 'none', '-n', str(ranks),
                   str(args.binary.resolve()), str(args.reference.resolve()), str(args.steps), str(axis), args.transport,
                   str(args.force_fixed), args.filter_workspace, str(dump.resolve()), args.compact_transport, args.method,
                   args.viscous_form, filter_method, ''.join(args.wall_axes), str(args.boundary_type), args.convection_form]
        with (args.output/f'{name}.log').open('w') as log:
            subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        text = (args.output/f'{name}.log').read_text()
        rows = re.findall(r'^CPP_WALL41_STEPS_PASS\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)', text, re.M)
        stage_count=12*len(args.wall_axes)*args.steps
        if len(rows) != ranks or any(int(row[0]) != stage_count for row in rows):
            raise AssertionError(f'{name}: missing full wall-stage observations')
        gradients = re.findall(r'^CPP_WALL41_GRADIENT\s+(\d+)\s+(\d+)\s+(\S+)', text, re.M)
        if len(gradients)!=ranks or any(int(r[1])!=stage_count for r in gradients):
            raise AssertionError(f'{name}: missing same-phase velocity/temperature gradients')
        packet_count = check_packets(dump, ranks, args.compare_output/name if args.compare_output is not None else None,
                                     walls=len(args.wall_axes))
        report[name] = dict(command=command, stages_per_rank=stage_count,
                            max_q_primitive_rhs=[max(float(row[c]) for row in rows) for c in range(1, 4)],
                            final_q_packets=packet_count,
                            max_gradient=max(float(r[2]) for r in gradients),
                            bitwise_reference=str(args.compare_output) if args.compare_output is not None else None)
        if args.viscous_form == 'me6_opti':
            traffic = re.findall(r'^CPP_WALL41_ME6_TRAFFIC\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', text, re.M)
            if len(traffic)!=ranks or any(int(r[1])!=int(args.transport=='device') for r in traffic):
                raise AssertionError(f'{name}: missing ME6 private-edge communication evidence')
            if args.transport=='device' and any(int(r[3])!=0 for r in traffic):
                raise AssertionError(f'{name}: ME6 device transport staged on host')
            report[name]['me6'] = dict(sent_bytes=sum(int(r[2]) for r in traffic),
                                       host_stage_bytes=sum(int(r[3]) for r in traffic))
        if args.method == 'compact' or (filter_method=='compact' and args.filter_workspace!='off'):
            residuals = re.findall(r'^CPP_WALL41_COMPACT_RESIDUAL\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)', text, re.M)
            traffic = re.findall(r'^CPP_WALL41_COMPACT_TRAFFIC\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', text, re.M)
            samples=27 if args.method=='compact' else 0
            if filter_method=='compact' and args.filter_workspace!='off':
                samples+=45 if args.filter_workspace=='scalar' else 9
            mode = int(args.compact_transport=='device-aware')
            samples=samples//3*len(args.wall_axes)
            if len(residuals)!=ranks or len(traffic)!=ranks or any(int(r[1])!=samples for r in residuals):
                raise AssertionError(f'{name}: missing actual compact-system/communication evidence')
            if any(int(r[1])!=mode or (mode==1 and int(r[3])!=0) for r in traffic):
                raise AssertionError(f'{name}: wrong compact interface transport or unexpected host staging')
            report[name]['compact'] = dict(linear_samples_per_rank=samples,
                residual=max(float(r[2]) for r in residuals), lu_error=max(float(r[3]) for r in residuals),
                sent_bytes=sum(int(r[2]) for r in traffic), host_stage_bytes=sum(int(r[3]) for r in traffic))
        print(name, report[name]['max_q_primitive_rhs'], flush=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
