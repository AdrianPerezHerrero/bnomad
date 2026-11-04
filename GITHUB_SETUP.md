# Repository and GitHub validation

The repository is [AdrianPerezHerrero/bnomad](https://github.com/AdrianPerezHerrero/bnomad).
The distribution/import remains `hdpgpc`, version 0.3.0. Install with CPython
3.10–3.12 as described in the [installation guide](docs/installation.md).

The [Installed package workflow](https://github.com/AdrianPerezHerrero/bnomad/actions/workflows/tests.yml) builds a wheel
and source archive, checks their contents and metadata, then installs the wheel
with dependency resolution in a fresh virtual environment. CPU-only PyTorch is
installed first to avoid unnecessary CUDA downloads. The source archive is
unpacked outside the checkout; isolated examples and provenance checks verify
that imports come from the installed wheel.

`tools/verify_installed.py` runs the console reference, provenance, dependencies,
documentation links, five standalone examples, the full regression suite,
repeated deterministic captures and a paired performance check. On failure,
the uploaded evidence artifact retains the complete stage log and `status.json`.
Read every failure block; an error in the test stage does not mean installation
failed. See the [current CI diagnosis](docs/documentation_audit.md).

Use the [release checklist](docs/release_checklist.md) for local reproduction and
remaining publication steps. The [initial migration report](docs/migration_0_3/report.md)
is historical evidence, not the current repository or validation status.

Notebook execution is a separate gate requiring `.[tutorials]`. See
[tutorials/README.md](tutorials/README.md) for automatic validation and building
the Colab wheel/helper/data bundle from the same source checkout. The normal
wheel stays small and excludes datasets and experimental outputs.
