"""Small paired CPU warp regression gate; no Git checkout or data required.

The control uses the preserved custom-expansion fallback (pre-P01 calculation).
A generous 25% ratio gate detects large regressions, not small timing noise.
This is not a replacement for the full performance study in docs/.
"""
import argparse
import json
import os
import statistics
import time

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import torch
from threadpoolctl import threadpool_limits
from hdpgpc.alignment import GaussianProcessWarp


class OriginalExpansion(GaussianProcessWarp):
    def _expand_ctrl_to_T(self, controls):
        return super()._expand_ctrl_to_T(controls)


def capture(cls):
    torch.manual_seed(42)
    x = torch.linspace(0., 4., 64, dtype=torch.float64)[:, None]
    template = torch.exp(-((x-1.8)/.65)**2)
    target = torch.exp(-((x-2.1)/.65)**2)
    start = time.perf_counter()
    output = cls(x, recursive=False).compute_warp(x, target, template, train_iter=50)
    seconds = time.perf_counter()-start
    def encode(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [encode(item) for item in value]
        return value
    return seconds, encode(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    samples = {'current': [], 'control': []}
    with threadpool_limits(limits=1):
        for iteration in range(8):
            results = {}
            order = ('current', 'control') if iteration % 2 else ('control', 'current')
            for name in order:
                elapsed, result = capture(GaussianProcessWarp if name == 'current' else OriginalExpansion)
                results[name] = result
                if iteration:
                    samples[name].append(elapsed)
            assert results['current'] == results['control'], 'Warp values or objective changed'
    medians = {name: statistics.median(values) for name, values in samples.items()}
    ratio = medians['current']/medians['control']
    report = dict(seconds=samples, median_seconds=medians, ratio=ratio,
                  exact_outputs=True, max_absolute_difference=0., maximum_ratio=1.25)
    text = json.dumps(report, indent=2)+'\n'
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text, encoding='utf-8')
    print(text)
    if ratio > 1.25:
        raise SystemExit('Performance regression: investigate on an idle machine; do not loosen the gate')


if __name__ == '__main__':
    main()
