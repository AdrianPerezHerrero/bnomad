# HDP-GPC

Hierarchical Dirichlet Process Gaussian Process Clustering for waveform sequences,
with static or LDS dynamics and optional GP temporal alignment.

**Version 0.3.0 - native modular implementation.** Author: Adrian Perez Herrero.

## Install and reproduce a small experiment

Use Python 3.10–3.12 in a fresh virtual environment. From this repository:

```sh
python -m pip install .
hdpgpc-reference --check --output reference.json
```

The command generates six synthetic waveforms, fits the offline model, checks
its numerical reference and saves configuration, results and environment metadata.
It works outside the checkout; no datasets or manual path changes are required.

## Python API

```python
from hdpgpc import HDPGPC
from hdpgpc.examples import configuration, waveforms

model = HDPGPC(configuration('offline')).fit(waveforms())
print(model.labels_)  # [0 0 1 1 0 1]

stream = HDPGPC(configuration('online'))
for x in waveforms()[:4]:
    result = stream.update(x)
    print(result.assigned_cluster)
```

These recipes explicitly choose small-data hyperparameters. They do not replace
model defaults. Predictions and stored latent states are available through
`predictive(cluster)` and `posterior(cluster)`. Retained assignments are hard;
soft assignment probabilities are explicitly unavailable. Online warping and
online multi-output paths have known limitations; use the offline tutorials.

## Documentation

- [Installation, dependency extras and Python support](docs/installation.md)
- [Online, offline, static, warping and multi-output tutorials](docs/tutorials.md)
- [Choose LDS, HDP, reduction, and warp options](docs/model_choices.md)
- [Comparable dataset experiment scripts and run folders](docs/experiment_workflows.md)
- [Complete-record experiments](docs/record_workflow.md)
- [Real-record Google Colab notebooks](tutorials/README.md)
- [Public API](docs/public_api.md) and [typed configuration](docs/configuration.md)
- [Reproducibility and tests](docs/reproducibility.md)
- [Migration from the old API](docs/migration.md)
- [Release checklist](docs/release_checklist.md) and [all documentation](docs/README.md)
- [Changelog](CHANGELOG.md)

The 0.3 release removes the old model classes, flat constructor and historical
pickle paths. All execution uses the native modular implementation. See the
[migration guide](docs/migration.md) before upgrading an older experiment.
The five maintained examples and eight tutorials replace the path-dependent
historical study scripts. Their original copies remain in the 0.2 archive.
Optional notebook/dataset tools: `pip install '.[datasets,tutorials]'`.

## License and citation

GPL-3.0-only; see [LICENSE](LICENSE). Adapted bnpy routines retain their BSD
attribution, reproduced in [NOTICE.md](NOTICE.md) and [licenses/bnpy.txt](licenses/bnpy.txt).
Use [CITATION.cff](CITATION.cff) to cite this software and record the exact version,
configuration and environment. No paper DOI is currently specified.
