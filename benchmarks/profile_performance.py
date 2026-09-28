"""Separate wall-clock, cProfile, traced memory and torch allocation measurements.

Run one workload per process to isolate allocator history. Profiler timings are
diagnostic only; never compare them to uninstrumented benchmark times.
"""
import argparse
from contextlib import redirect_stdout
import cProfile
import gc
import hashlib
import io
import json
from pathlib import Path
import pstats
import statistics
import threading
import time
import tracemalloc
from unittest.mock import patch

import psutil
from performance_workloads import NAMES, run, reset_runtime, snapshot, environment, torch


def measured(name):
    reset_runtime()
    with redirect_stdout(io.StringIO()):
        return run(name)


def digest(result):
    return hashlib.sha256(json.dumps(snapshot(result), sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def python_profile(name):
    profile = cProfile.Profile()
    reset_runtime()
    with redirect_stdout(io.StringIO()):
        profile.enable()
        result = run(name)
        profile.disable()
    rows = []
    for (filename, line, function), (primitive, total, own, cumulative, callers) in pstats.Stats(profile).stats.items():
        rows.append(dict(file=filename.replace('\\', '/'), line=line, function=function,
                         calls=total, primitive_calls=primitive, self_seconds=own,
                         cumulative_seconds=cumulative))
    return {'self_seconds': sum(r['self_seconds'] for r in rows),
            'functions': sorted(rows, key=lambda r: r['cumulative_seconds'], reverse=True),
            'state_sha256': digest(result)}


def memory_profile(name):
    gc.collect()
    process = psutil.Process()
    before = process.memory_info().rss
    peaks = [before]
    stop = threading.Event()

    def sample_rss():
        while not stop.wait(.005):
            peaks.append(process.memory_info().rss)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    try:
        result = measured(name)
        peaks.append(process.memory_info().rss)
    finally:
        stop.set()
        sampler.join()
    rss_digest = digest(result)
    del result
    gc.collect()
    # Keep Python tracing overhead out of the RSS measurement above.
    tracemalloc.start()
    try:
        result = measured(name)
        current, peak = tracemalloc.get_traced_memory()
        allocation_snapshot = tracemalloc.take_snapshot()
    finally:
        tracemalloc.stop()
    lines = [{'location': str(s.traceback[0]), 'retained_bytes': s.size, 'blocks': s.count}
             for s in allocation_snapshot.statistics('lineno')[:30]]
    return {'rss_before_bytes': before, 'rss_sampled_peak_bytes': max(peaks),
            'rss_sampled_growth_bytes': max(peaks)-before, 'sample_interval_ms': 5,
            'python_traced_peak_bytes': peak, 'python_traced_retained_bytes': current,
            'retained_allocation_lines': lines, 'state_sha256': digest(result),
            'rss_state_sha256': rss_digest}


def tensor_profile(name):
    reset_runtime()
    rows = []
    steps = 0
    original_step = torch.optim.Adam.step

    def collect(profiler):
        rows.extend([{'operator': event.key, 'calls': event.count,
                      'self_cpu_us': event.self_cpu_time_total, 'cpu_us': event.cpu_time_total,
                      'self_cpu_memory_bytes': event.self_cpu_memory_usage,
                      'cpu_memory_bytes': event.cpu_memory_usage}
                     for event in profiler.key_averages()])

    with redirect_stdout(io.StringIO()), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True,
            record_shapes=False, with_stack=False,
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1),
            on_trace_ready=collect) as prof:
        def step(optimizer, *args, **kwargs):
            nonlocal steps
            result = original_step(optimizer, *args, **kwargs)
            steps += 1
            prof.step()
            return result

        with patch.object(torch.optim.Adam, 'step', step):
            result = run(name)
    return {'operators': sorted(rows, key=lambda r: r['self_cpu_us'], reverse=True),
            'state_sha256': digest(result), 'total_adam_steps': steps,
            'capture_window': 'construction through first 3 Adam steps only; remaining execution untraced'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workload', choices=NAMES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timing-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report.get('complete'):
            parser.error('Choose a new report path')
        assert report['workload'] == args.workload
        expected = report['state_sha256']
    else:
        reset_runtime()
        warmup = measured(args.workload)
        expected = digest(warmup)
        del warmup
        times = []
        hashes = []
        for _ in range(args.repeats):
            gc.collect()
            reset_runtime()
            with redirect_stdout(io.StringIO()):
                start = time.perf_counter()
                result = run(args.workload)
                times.append(time.perf_counter()-start)
            hashes.append(digest(result))
            del result
        report = {'workload': args.workload, 'environment': environment(),
                  'seconds': times, 'median_seconds': statistics.median(times),
                  'state_sha256': expected, 'timing_states_identical': all(h == expected for h in hashes)}
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(args.workload, 'timing complete', report['median_seconds'], flush=True)
    if not args.timing_only:
        for mode, operation in (('python', python_profile), ('memory', memory_profile), ('torch', tensor_profile)):
            if mode not in report:
                report[mode] = operation(args.workload)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(args.workload, mode, 'complete', flush=True)
        assert all(report[mode]['state_sha256'] == expected for mode in ('python', 'memory', 'torch'))
    assert report['timing_states_identical']
    report['complete'] = True
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(args.workload, report['median_seconds'], flush=True)


if __name__ == '__main__':
    main()
