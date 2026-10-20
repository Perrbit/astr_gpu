set pagination off
set confirm off
set print thread-events off
set disable-randomization off
python
import gdb
import json
import os
import time

rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
traffic = {name: dict(calls=0, sent_bytes=0, host_stage_bytes=0)
           for name in ('exchange', 'broadcast')}

class InitializationReturn(gdb.FinishBreakpoint):
    def __init__(self, label):
        super().__init__(gdb.newest_frame(), internal=True)
        self.label = label
        self.started = time.monotonic()
    def stop(self):
        print('COMPACT_INIT_DIAGNOSTIC ' + json.dumps(dict(
            rank=rank, phase=self.label, seconds=time.monotonic()-self.started)))
        return False

class InitializationEntry(gdb.Breakpoint):
    def __init__(self, symbol, label):
        super().__init__(symbol, internal=True)
        self.label = label
    def stop(self):
        self.enabled = False
        InitializationReturn(self.label)
        return False

class TransportReturn(gdb.FinishBreakpoint):
    def __init__(self, label, pointer, before):
        super().__init__(gdb.newest_frame(), internal=True)
        self.label, self.pointer, self.before = label, pointer, before
    def stop(self):
        state = self.pointer.dereference()
        row = traffic[self.label]
        row['calls'] += 1
        row['sent_bytes'] += int(state['sent_bytes']) - self.before[0]
        row['host_stage_bytes'] += int(state['host_stage_bytes']) - self.before[1]
        return False

class TransportEntry(gdb.Breakpoint):
    def __init__(self, symbol, label):
        super().__init__(symbol, internal=True)
        self.label = label
    def stop(self):
        state = gdb.newest_frame().read_var('transport')
        if state.type.code == gdb.TYPE_CODE_PTR:
            state = state.dereference()
        TransportReturn(self.label, state.address,
                        (int(state['sent_bytes']), int(state['host_stage_bytes'])))
        return False

InitializationEntry('compact_runtime_configure_global_compact_', 'direction_groups')
InitializationEntry('compact_operator_gpu_prepare_operator_', 'derivative_setup')
InitializationEntry('compact_operator_gpu_prepare_filter_operator_', 'filter_setup')
TransportEntry('compact_transport_gpu_exchange_compact_interfaces_', 'exchange')
TransportEntry('compact_transport_gpu_broadcast_compact_interfaces_', 'broadcast')
end
run
python
print('COMPACT_TRANSPORT_DIAGNOSTIC ' + json.dumps(dict(rank=rank, counters=traffic)))
end
