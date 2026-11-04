# Release checklist: 0.3.0

The [current audit](documentation_audit.md) records completed local checks and
remaining hosted validation. The [September migration report](migration_0_3/report.md)
is historical; its deferred checks do not describe the present audit.

## Before tagging

- [x] Declare metadata, Python 3.10–3.12 and bounded dependencies in pyproject.
- [x] Separate dataset, notebook/YAML and development extras.
- [x] Include GPL license and bnpy BSD attribution in artifacts.
- [x] Include software citation metadata without inventing a DOI or arXiv identifier.
- [x] Document public API, component clocks, observation models and unsupported paths.
- [x] Package synthetic configurations and the pinned offline numerical reference.
- [x] Retain reviewed characterization fixtures without regenerating them for CI.
- [x] Complete local wheel/sdist, installed-package, tutorial and deterministic
  checks; record their exact scope in the current audit.
- [ ] Obtain green hosted CI for every declared Python/OS matrix entry.
- [ ] Review version, author metadata and release notes with the maintainer.
- [ ] Tag the reviewed commit, archive artifacts/checksums and publish.
- [ ] Verify installation of the published artifact in another fresh environment.

## Reusable local verification

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
python tools/check_artifacts.py dist
python -m venv /path/outside-checkout/clean
# Activate clean and install the wheel noneditably; do not enable system site packages.
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install /path/to/dist/hdpgpc-0.3.0-py3-none-any.whl
python -m pip check
# Change to a directory outside the development checkout:
python -I -m hdpgpc.examples.reference --check --output reference.json
```

Extract the source archive outside the checkout, set `HDPGPC_TEST_INSTALLED=1`,
clear `PYTHONPATH`, then run from its root:

```sh
python -B -m unittest discover -s tests -v
python -B -m tests.characterization.verify_determinism --runs 3 --report determinism.json
python tools/performance_smoke.py --output performance.json
python -m pip freeze > environment.txt
```

Alternatively, `python tools/verify_installed.py --output /new/evidence/directory`
runs the installed console command, provenance, dependency, documentation, reference, all five
standalone examples, full-suite, three determinism captures and performance checks
sequentially, with logs and a machine-readable status. Standalone examples use
Python's isolated mode (`-I`), so success cannot depend on checkout path injection.
This is the same verifier invoked by CI.

The install test asserts `site-packages` provenance. Archive the logs, numerical
comparison reports and environment. Do not regenerate fixtures or tune priors to
make a platform pass. Any precision-aware exception needs measured differences,
a documented numerical bound and checks that still reject substantive drift; see
[reproducibility](reproducibility.md). Investigate unsupported paths explicitly.
