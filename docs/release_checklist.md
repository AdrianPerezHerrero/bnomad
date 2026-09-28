# Release checklist: 0.3.0 preparation

This is a release candidate preparation, not a claim that a release was published.

## Before tagging

- [x] Preserve inference mathematics; remove retired implementation and compatibility APIs.
- [x] Declare metadata, Python 3.10–3.12 and bounded core dependencies in pyproject.
- [x] Separate dataset, notebook/YAML and development extras.
- [x] Include GPL license and bnpy BSD attribution in artifacts; retain source notices.
- [x] Include citation metadata without inventing a DOI.
- [x] Document public API, migration, deterministic recipes and known unsupported paths.
- [x] Package synthetic data generation/configurations and an offline numerical reference.
- [x] Bring original characterization fixtures into the repository unchanged.
- [ ] Verify final wheel and source distribution with build, twine and artifact checks.
- [ ] Verify clean noneditable installation and examples outside the development checkout.
- [ ] Run all unit, characterization and integration tests against the installed artifact.
- [ ] Repeat deterministic captures and the lightweight paired performance gate.
- [ ] Record exact environment, commands and results in release validation documentation.
- [ ] Obtain green CI for every declared Python/OS matrix entry; local testing is not CI.
- [ ] Review version, repository URLs, author metadata and release notes with the maintainer.
- [ ] Tag the reviewed commit, archive artifacts/checksums and publish using trusted credentials.
- [ ] Verify installation of the published artifact in another fresh environment.

## Reusable local verification

```sh
python -m pip install build twine
python -m build
python -m twine check dist/*
python tools/check_artifacts.py dist
python -m venv /path/outside-checkout/clean
# Activate clean and install the wheel noneditably; do not enable system site packages.
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
runs the installed console command, provenance, dependency, reference, all five
standalone examples, full-suite, three determinism captures and performance checks
sequentially, with logs and a machine-readable status. Standalone examples use
Python's isolated mode (`-I`), so success cannot depend on checkout path injection.
This is the same verifier invoked by CI.

The install test asserts `site-packages` provenance. Archive the logs, numerical
comparison reports and environment. Do not loosen baselines or tolerances to
make another platform pass. Investigate and report unsupported paths explicitly.
