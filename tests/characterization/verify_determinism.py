"""Repeat complete captures in fresh interpreters and compare exact JSON bytes.

One capture reverses case order. Repeated captures must match one another exactly.
Each also meets the reviewed scientific baseline tolerance across CPU/BLAS builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from ._support import assert_snapshot, snapshot_tolerances, RTOL, ATOL


def assert_baseline_snapshot(actual, expected, filename):
    """Use the same case-relative paths as the characterization unit tests."""
    assert_snapshot(unittest.TestCase(), actual, expected, path=Path(filename).stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--report", type=Path, help="Optional new JSON report path; never overwrites.")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("At least two independent runs are required")
    if args.report and args.report.exists():
        parser.error("Report already exists; use a new report path")
    workspace = Path(__file__).resolve().parents[2]
    baseline = Path(__file__).with_name("baselines")
    reference = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in baseline.glob("*.json") if p.name != "manifest.json"}
    if not reference:
        raise RuntimeError("No reviewed baselines exist")
    runs = []
    first_hashes = None
    with tempfile.TemporaryDirectory(prefix="hdpgpc-characterization-") as temporary:
        for number in range(args.runs):
            output = Path(temporary) / f"run-{number + 1}"
            reverse = number % 2 == 1
            cmd = [sys.executable, "-B", "-m", "tests.characterization.record_baseline", "--output", str(output),
                   "--reference-commit", "determinism verification"]
            if reverse:
                cmd.append("--reverse")
            start = time.perf_counter()
            subprocess.run(cmd, cwd=workspace, check=True)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.glob("*.json") if p.name != "manifest.json"}
            outside_tolerance = []
            baseline_differences = {}
            for name in sorted(reference.keys() | hashes.keys()):
                if name not in reference or name not in hashes:
                    outside_tolerance.append(name)
                    baseline_differences[name] = ('Missing captured snapshot' if name not in hashes
                                                 else 'No reviewed baseline')
                    continue
                try:
                    assert_baseline_snapshot(json.loads((output/name).read_text()),
                                             json.loads((baseline/name).read_text()), name)
                except AssertionError as error:
                    outside_tolerance.append(name)
                    baseline_differences[name] = str(error)
            if first_hashes is None:
                first_hashes = hashes
            changed = sorted(name for name in first_hashes.keys() | hashes.keys()
                             if first_hashes.get(name) != hashes.get(name))
            runs.append({"run": number + 1, "reverse_order": reverse,
                         "seconds": round(time.perf_counter() - start, 3),
                         "different_from_first_run": changed,
                         "outside_baseline_tolerance": sorted(outside_tolerance),
                         "baseline_differences": baseline_differences, "sha256": hashes})
            print(f"Run {number + 1}: {len(changed)} exact repeat differences; "
                  f"{len(outside_tolerance)} baseline regressions", flush=True)
            for name in changed:
                print(f"  Exact repeat difference: {name}", flush=True)
            for name, difference in baseline_differences.items():
                print(f"  Baseline difference in {name}: {difference}", flush=True)
        exact_repeats = all(not run["different_from_first_run"] for run in runs)
        baseline_matches = all(not run["outside_baseline_tolerance"] for run in runs)
        report = {"comparison": f"byte-for-byte repeated snapshots; default baseline rtol={RTOL:g}, atol={ATOL:g}",
                  "baseline_precision_exceptions": {
                      "warper_scalar/batch.calls[*].warm_start.values[*]": {
                          "rtol": RTOL,
                          "atol": snapshot_tolerances('warper_scalar.calls[0].warm_start.values[0]')[1]}},
                  "python_executable": sys.executable, "runs": runs,
                  "exact_repeats_identical": exact_repeats, "baseline_matches": baseline_matches,
                  "all_identical": exact_repeats and baseline_matches}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["all_identical"]:
        failures = []
        if not exact_repeats:
            failures.append('Repeated snapshots differ')
        if not baseline_matches:
            failures.append('Snapshots differ from reviewed baselines')
        raise SystemExit('; '.join(failures) + ': inspect the per-case differences above')


if __name__ == "__main__":
    main()
