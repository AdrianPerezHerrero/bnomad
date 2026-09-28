"""Repeat complete captures in fresh interpreters and compare exact JSON bytes.

One capture reverses case order. Numerical tolerances are deliberately NOT used
here: variability is reported rather than hidden by the regression tolerance.
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
    with tempfile.TemporaryDirectory(prefix="hdpgpc-characterization-") as temporary:
        for number in range(args.runs):
            output = Path(temporary) / f"run-{number + 1}"
            reverse = number % 2 == 1
            cmd = [sys.executable, "-B", "-m", "tests.characterization.record_baseline", "--output", str(output)]
            if reverse:
                cmd.append("--reverse")
            start = time.perf_counter()
            subprocess.run(cmd, cwd=workspace, check=True)
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.glob("*.json") if p.name != "manifest.json"}
            changed = sorted(name for name in reference.keys() | hashes.keys() if reference.get(name) != hashes.get(name))
            runs.append({"run": number + 1, "reverse_order": reverse,
                         "seconds": round(time.perf_counter() - start, 3),
                         "different_from_baseline": changed, "sha256": hashes})
            print(f"Run {number + 1}: {len(changed)} exact differences", flush=True)
        report = {"comparison": "byte-for-byte snapshots, no numerical tolerance",
                  "python_executable": sys.executable, "runs": runs,
                  "all_identical": all(not run["different_from_baseline"] for run in runs)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["all_identical"]:
        raise SystemExit("Snapshot variability detected: inspect the listed cases before changing tolerances")


if __name__ == "__main__":
    main()
