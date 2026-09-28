"""Explicit, separate capture tool. Normal tests never generate expectations.

Write into a NEW directory, inspect the diffs, then deliberately promote reviewed
files. This command refuses to overwrite an existing destination.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path

from ._cases import CASES
from ._support import encode, environment, reset_runtime, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true", help="Reverse case order to check global-state leakage.")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new directory and review before promotion.")
    args.output.mkdir(parents=True)
    before = source_hashes()
    names = list(CASES)
    if args.reverse:
        names.reverse()
    for name in names:
        reset_runtime()
        with redirect_stdout(io.StringIO()):
            payload = encode(CASES[name]())
        path = args.output / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Captured {name}", flush=True)
    if source_hashes() != before:
        raise AssertionError("Production source changed during capture")
    manifest = {"schema_version": 1, "reference_commit": "615109a9e474d37626abaff6c51f1c159b26124a",
                "cases": sorted(CASES), "environment": environment(),
                "comparison": {"discrete": "exact", "float_rtol": 1e-7, "float_atol": 1e-9}}
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
