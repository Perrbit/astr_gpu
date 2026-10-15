"""Decode the versioned AIR5 configuration, including optional numerical tails."""
import struct

FILTER_IDENTITY = (1, 123, 5, 1, 1)


def read_air5_contract(payload):
    counts = {b'ASTRA501': 3, b'ASTRA502': 4}
    if payload[:8] not in counts:
        raise ValueError('unknown AIR5 configuration version')
    base = 1608 + 8*counts[payload[:8]]
    lengths = (base, base+128, base+144, base+192)
    if len(payload) not in lengths:
        raise ValueError('AIR5 configuration size/tail')
    source = 'joint' if len(payload) == base else payload[base:base+128].decode('ascii').rstrip()
    if source not in ('joint', 'source-only'):
        raise ValueError('AIR5 source evaluation contract mismatch')
    control = (0, 0) if len(payload) < base+144 else struct.unpack_from('<2q', payload, base+128)
    filter_control = (0,)*5
    if len(payload) == base+192:
        if payload[base+144:base+152] != b'ASTRF001':
            raise ValueError('AIR5 filter control version/size')
        filter_control = struct.unpack_from('<5q', payload, base+152)
        if filter_control != FILTER_IDENTITY:
            raise ValueError('AIR5 filter control contract mismatch')
    return dict(source_evaluation=source, integration_control=control,
                filter_identity=filter_control, integration_offset=base+128,
                filter_offset=base+144)
