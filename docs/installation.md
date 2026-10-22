# Installation and supported environments

Use CPython 3.10, 3.11 or 3.12 and a virtual environment. The CPU reference
environment is Windows, Python 3.11. Python 3.13+, PyPy and CUDA reproducibility
are not supported by this release. CI declares Linux and Windows coverage for
all three Python versions; see the release checklist for actual validation status.

From the repository root:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
hdpgpc-reference --check --output reference.json
```

The last command generates synthetic data, runs offline inference, checks the
packaged numerical reference and writes the configuration, data hash, results and
environment. It works from any directory. No datasets or path manipulation are
needed. `python -m hdpgpc.examples.reference` is an equivalent entry point.
Choose a new output filename on each run.

The CPU-only PyTorch installation above avoids downloading CUDA libraries on
machines that do not need GPU execution. It follows the
[official PyTorch 2.9.0 installation instructions](https://pytorch.org/get-started/previous-versions/#v290).
Installing the package directly on Linux without first choosing a PyTorch build
can download several gigabytes of CUDA dependencies and exhaust a small CI disk.
The installed-package CI workflow preinstalls CPU-only PyTorch in its fresh
environment, then installs the built wheel with dependency resolution enabled.
It uses Python archive extraction to handle Linux and Windows paths consistently.

Optional extras:

```sh
python -m pip install '.[datasets]'  # WFDB and audio readers; no data download
python -m pip install '.[examples]'  # notebook tools and full YAML parser
python -m pip install '.[tutorials]' # automatic execution of the real-record notebooks
python -m pip install '.[yaml]'      # full YAML parser only
python -m pip install '.[test]'      # test tooling
python -m pip install '.[dev]'       # build, test and profiling tools
```

The runnable synthetic examples need only the core installation. Bundled recipes
use JSON syntax, which is also valid YAML, and load without PyYAML. Dataset files
and historical experiment output are excluded from release archives. Obtain any
external datasets separately and respect their licenses.

Dependency ranges deliberately stay close to the characterized stack: NumPy 1.26,
SciPy 1.14, PyTorch 2.9.0, GPyTorch 1.13, linear-operator 0.5 and scikit-learn 1.5.
PyTorch is bounded below 2.9.1: the Windows 2.9.1 wheel bundles a newer MKL and
failed exact inference-trace comparisons at approximately 7e-17 in a covariance
entry. The public offline result remained within 5.6e-17 of the reference. This
release retains exact-reference compatibility rather than relaxing those tests.
Pyro, plotting and pandas remain core dependencies because existing model imports
use them eagerly. Moving them behind optional imports would be a separate change.
Python's lower bound follows [PyTorch 2.9](https://pypi.org/project/torch/2.9.0/)
and [SciPy 1.14](https://pypi.org/project/scipy/1.14.1/); the upper bound retains
the NumPy 1.26 reference family. Do not interpret bounded ranges as bitwise
equivalence across platforms. Use `requirements/reference.txt` as constraints
for the direct scientific versions used by the frozen reference fixtures:

```sh
python -m pip install -c requirements/reference.txt '.[test]'
python -m pip check
```

Those constraints are not a complete transitive lock. Archive `pip freeze` with
each experiment. The original 0.2 folder retains its historical `hdpgpc/docs/requirements.txt`
development snapshot; it is not included or required in this release.

For source development, editable installation remains available with
`python -m pip install -e '.[dev]'`. Release verification always uses a noneditable
installation in a fresh virtual environment.
