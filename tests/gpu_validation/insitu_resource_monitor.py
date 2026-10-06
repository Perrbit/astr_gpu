"""Local test process-tree memory sampling; not a production allocation guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import psutil
import pynvml as nvml


def run_monitored(command, directory, runtime, stream, report_path, baseline=None,
                  device_extra_budget_bytes=2*1024**3, host_extra_budget_bytes=4*1024**3,
                  device_reserve_bytes=1024**3, timeout_seconds=180, check_progress=None):
    nvml.nvmlInit()
    devices = [(nvml.nvmlDeviceGetUUID(h), h) for h in (
        nvml.nvmlDeviceGetHandleByIndex(i) for i in range(nvml.nvmlDeviceGetCount()))]
    report = dict(status='running', sampling_period_seconds=0.02, samples=0,
                  host_rss_peak_bytes=0, devices={},
                  scope='sum of process-tree RSS; deduplicated NVML compute/graphics PID memory')
    process = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(command, cwd=directory, env=runtime, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        tree = psutil.Process(process.pid)
        while True:
            try:
                children = [tree, *tree.children(recursive=True)]
            except psutil.NoSuchProcess:
                children = []
            pids = set()
            rss = 0
            for child in children:
                try:
                    rss += child.memory_info().rss
                    pids.add(child.pid)
                except psutil.NoSuchProcess:
                    pass
            report['host_rss_peak_bytes'] = max(report['host_rss_peak_bytes'], rss)
            for uuid, handle in devices:
                by_pid = {}
                for query in (nvml.nvmlDeviceGetComputeRunningProcesses,
                              nvml.nvmlDeviceGetGraphicsRunningProcesses):
                    for item in query(handle):
                        if item.pid not in pids:
                            continue
                        value = item.usedGpuMemory
                        if value is None or value == (1 << 64)-1:
                            raise ValueError('NVML cannot report per-process device memory')
                        by_pid[item.pid] = max(by_pid.get(item.pid, 0), value)
                if not by_pid:
                    continue
                memory = nvml.nvmlDeviceGetMemoryInfo(handle)
                entry = report['devices'].setdefault(uuid, dict(peak_bytes=0, min_free_bytes=memory.free))
                entry['peak_bytes'] = max(entry['peak_bytes'], sum(by_pid.values()))
                entry['min_free_bytes'] = min(entry['min_free_bytes'], memory.free)
                if memory.free < device_reserve_bytes:
                    raise ValueError(f'{uuid}: device free-memory reserve violated')
                if baseline is not None:
                    if uuid not in baseline['devices']:
                        raise ValueError('GPU identity differs from off baseline')
                    if entry['peak_bytes']-baseline['devices'][uuid]['peak_bytes'] > device_extra_budget_bytes:
                        raise ValueError(f'{uuid}: sampled additional device-memory budget exceeded')
            if baseline is not None and rss-baseline['host_rss_peak_bytes'] > host_extra_budget_bytes:
                raise ValueError('Sampled additional host-RSS budget exceeded')
            if check_progress is not None:
                check_progress()
            report['samples'] += 1
            code = process.poll()
            if code is not None:
                if code:
                    raise subprocess.CalledProcessError(code, command)
                break
            if time.monotonic()-started > timeout_seconds:
                raise TimeoutError('Local resource probe exceeded its time budget')
            time.sleep(0.02)
        if not report['devices']:
            raise ValueError('No GPU process memory observed')
        if baseline is not None:
            if set(report['devices']) != set(baseline['devices']):
                raise ValueError('Observed GPU set differs from baseline')
            report['additional_host_peak_difference_bytes'] = (
                report['host_rss_peak_bytes']-baseline['host_rss_peak_bytes'])
            report['additional_device_peak_difference_bytes'] = {
                uuid: value['peak_bytes']-baseline['devices'][uuid]['peak_bytes']
                for uuid, value in report['devices'].items()}
        report['status'] = 'passed-sampled-observation-not-allocation-proof'
    except BaseException as error:
        report.update(status='failed', error=str(error))
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-started
        Path(report_path).write_text(json.dumps(report, indent=2))
        nvml.nvmlShutdown()
    return report
