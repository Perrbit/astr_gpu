"""Incremental, full-precision CFL checks for bounded scale tests."""
from math import isfinite
import re


class CflGate:
    def __init__(self, path, limit, dt, first_step, stop_step):
        if not isfinite(limit) or limit <= 0 or not isfinite(dt) or dt <= 0:
            raise ValueError('CFL gate requires a positive finite limit and timestep')
        self.path, self.limit, self.dt = path, limit, dt
        self.next_step, self.stop_step = first_step, stop_step
        self.offset, self.pending, self.header = 0, b'', None
        self.maximum = 0.

    def check(self, final=False):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            data = self.pending + stream.read()
            self.offset = stream.tell()
        rows = data.split(b'\n')
        self.pending = rows.pop()
        for raw in rows:
            line = raw.decode('utf-8', errors='replace').strip()
            if line.startswith('ASTR_CFL complete_step='):
                match = re.fullmatch(r'ASTR_CFL complete_step=(\d+)\s+state_time=\s*(\S+)\s+dt=\s*(\S+)', line)
                if not match or self.header is not None:
                    raise ValueError('Malformed or unpaired CFL identity')
                step, time, dt = int(match[1]), float(match[2]), float(match[3])
                if step != self.next_step or dt != self.dt or not isfinite(time):
                    raise ValueError('CFL sampling sequence or timestep differs')
                self.header = step
            elif line.startswith('ASTR_CFL directional/local_sum/upper_bound='):
                values = [float(value) for value in line.split('=', 1)[1].split()]
                if self.header is None or len(values) != 5 or not all(isfinite(v) and v >= 0 for v in values):
                    raise ValueError('Missing identity or invalid CFL values')
                if values[-1] > self.limit:
                    raise ValueError(f'CFL gate exceeded at step {self.header}: {values[-1]} > {self.limit}')
                self.maximum = max(self.maximum, values[-1])
                self.next_step += 1
                self.header = None
        if final and (self.header is not None or self.next_step != self.stop_step):
            raise ValueError('Missing per-step CFL records')
