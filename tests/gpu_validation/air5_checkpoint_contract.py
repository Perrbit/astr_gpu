"""Decode the versioned AIR5 configuration, including optional numerical tails."""
import struct
import math

FILTER_IDENTITY = (1, 123, 5, 1, 1)


def read_air5_contract(payload):
    counts = {b'ASTRA501': 3, b'ASTRA502': 4, b'ASTRA503': 0, b'ASTRA504': 0}
    if payload[:8] not in counts:
        raise ValueError('unknown AIR5 configuration version')
    base = 1608 + 8*counts[payload[:8]]
    if len(payload) < base:
        raise ValueError('AIR5 configuration size/tail')
    source = 'joint' if len(payload) == base else payload[base:base+128].decode('ascii').rstrip()
    if source not in ('joint', 'source-only'):
        raise ValueError('AIR5 source evaluation contract mismatch')
    offset=base if len(payload)==base else base+128
    control=(0,0)
    if offset<len(payload):
        if offset+16>len(payload):
            raise ValueError('AIR5 configuration size/tail')
        control=struct.unpack_from('<2q',payload,offset)
        offset+=16
    filter_control = (0,)*5
    activity_version,activity_threshold=1,0.0
    seen=set()
    while offset<len(payload):
        magic=payload[offset:offset+8]
        if magic in seen:
            raise ValueError('duplicate AIR5 numerical control')
        seen.add(magic)
        offset+=8
        if magic==b'ASTRF001' and offset+40<=len(payload):
            filter_control=struct.unpack_from('<5q',payload,offset)
            offset+=40
            if filter_control!=FILTER_IDENTITY:
                raise ValueError('AIR5 filter control contract mismatch')
        elif magic==b'ASTRACT1' and offset+16<=len(payload):
            activity_version,activity_threshold=struct.unpack_from('<qd',payload,offset)
            offset+=16
            if activity_version!=1 or not math.isfinite(activity_threshold) or activity_threshold<0:
                raise ValueError('AIR5 activity control contract mismatch')
        else:
            raise ValueError('AIR5 numerical control version/size')
    primitive_reuse = payload[8+6*128:8+7*128].decode('ascii').strip() or 'off'
    reductions = payload[8+7*128:8+8*128].decode('ascii').strip() or 'baseline'
    return dict(source_evaluation=source, integration_control=control,
                primitive_reuse=primitive_reuse, chemistry_reductions=reductions,
                activity_version=activity_version,activity_threshold=activity_threshold,
                filter_identity=filter_control, integration_offset=base+128,
                filter_offset=base+144)
