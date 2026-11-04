# HDP-GPC: real-record tutorials

Start with **00 → 01 → 02**, then choose **03** for streaming or **04** for batch
inference. Continue to **05** for timing variation, **06** for multiple channels,
and **07** only after standard online inference is familiar. All examples use CPU
float64, seed 42 and one numerical thread. A GPU is neither required nor validated.

| Notebook | Model mode | Main assumption | Local runtime | Difficulty |
| --- | --- | --- | --- | --- |
| [00 Quickstart](00_quickstart.ipynb) | Static, offline, single-output | Stationary cluster representatives | 6.8–6.9 s | Beginner |
| [01 Static clustering](01_static_clustering.ipynb) | Static with HMM assignments | Zero latent process covariance, not independent assignments | 6.9–7.1 s | Beginner |
| [02 Dynamic clustering](02_dynamic_clustering.ipynb) | LDS, offline, single-output | Representatives evolve across assigned observations | 6.8–6.9 s | Intermediate |
| [03 Online inference](03_online_inference.ipynb) | LDS, sequential | Commit one arriving observation at a time | 3.8–3.9 s | Intermediate |
| [04 Offline inference](04_offline_inference.ipynb) | LDS, repeated batch updates | Earlier samples may be revisited | 6.9–7.0 s | Intermediate |
| [05 Temporal warping](05_temporal_warping.ipynb) | LDS with offline GP alignment | Timing differences can be aligned | 7.1–7.2 s | Advanced |
| [06 Multi-output](06_multioutput.ipynb) | Two outputs, SNR/uniform comparison | Shared labels with output-specific models | 18.6–18.7 s | Intermediate |
| [07 Fast online](07_fast_online.ipynb) | Distinct online approximation | Different history/update behavior is acceptable | 4.6–4.8 s | Advanced |

Runtime ranges are the two complete local CPU runs recorded in the
[November 2026 audit](../docs/documentation_audit.md) on Linux/Python 3.12,
including setup with an already installed package. First-time dependency
installation and hosted Colab timings are excluded.

The [pre-creation inventory](inventory.md) is historical 0.2 evidence, not a current
feature matrix. Version 0.3 supports [PCA/affine observations](../docs/fixed_observation_models.md),
[learned MFA and shared loadings](../docs/mfa_emission.md), and
[fixed/shared LDS policies](../docs/model_choices.md). The eight notebooks focus
on waveform GP examples rather than covering every supported representation.
`share_gp` does not implement shared C; `bayesian_params=False` does not freeze
dynamics. Arbitrary nonlinear decoders require their own compatible inference.
Online warping and online multi-output have known unsupported paths and are not
presented as working tutorials.

## Google Colab: no notebook edits required

1. Open one of these `.ipynb` files in Colab using **File → Upload notebook**.
2. Use a CPU runtime and **Run all**. When prompted, upload `hdpgpc-colab.zip`.
   The bundle is created under `tutorials/bundles/` by the commands below. It contains the current
   wheel, two shared helpers, record 100, its annotations and license information.
3. Continue running cells in order. The one upload is reused by other tutorials
   opened in the same runtime. Scientific results and figures appear in the notebook;
   reports are written under `tutorial-results/` in the runtime's working directory.

The current source repository is [bnomad](https://github.com/AdrianPerezHerrero/bnomad);
the Python distribution/import name remains `hdpgpc`, version 0.3.0. A source
checkout is separate from a published PyPI release. Build the uploadable wheel
bundle from the same checkout as the notebooks. For reproducible source installs,
record a commit SHA rather than depending on a moving branch. The checked-in
Colab setup deliberately uses bundle upload; it does not install GitHub main or
replace the host runtime's scientific stack.

The setup creates a private Python environment rather than replacing Colab's
preloaded NumPy/PyTorch. It provisions Python 3.11 with uv (downloading an interpreter
if needed, independently of the host's `venv`/`ensurepip` support), installs a CPU Torch wheel and the bundled
package, and starts a persistent Jupyter worker. `%%hdpgpc` sends normal Python to
that worker and forwards text/plots. It is notebook infrastructure, not a second
model API. Variables persist between those cells; plain host-cell variables live
separately. Setup is idempotent and restarts the worker when rerun.

First installation requires internet, disk space for scientific dependencies and
potentially several minutes. It downloads **no large external dataset**. The data
included are only the roughly 3.3 MB original record/annotation pair. If those files
are unavailable locally, the loader can fetch exactly that pair from a pinned
repository commit and verifies their SHA-256 hashes.

[Google's runtime documentation](https://research.google.com/colaboratory/runtime-version-faq.html)
tracks changing runtime versions. The isolated environment
also handles hosts with newer Python; it does not depend on an undocumented runtime
metadata field or require a Colab kernel restart.

## Local use and building the uploadable bundle

From the repository root in a supported environment:

```sh
python -m pip install '.[tutorials]'
python -m pip install 'build>=1.2,<2'
python -m build
python tools/build_colab_bundle.py
```

Open a notebook with Jupyter/VS Code (install `.[examples]` for JupyterLab if needed).
The setup discovers the checkout, loads the shared helpers and reuses the installed
package in a fresh worker. No `sys.path` manipulation is used to import hdpgpc.
The sdist includes notebooks and helpers; large data files stay outside normal
package artifacts. The Colab bundle is a separate, explicitly requested data bundle.

## What the results mean

- The common input is six consecutive record-100 beats beginning at 1700. Both
  recorded channels are preserved for 06. Fifteen retained points and first-beat
  centering/scaling make the experiment small; these are explicit preprocessing
  choices, not package presets or a recommended clinical preprocessing pipeline.
- `100_labels.npy` has 2273 entries but the waveform array has 2272. No label pairing
  is assumed, and no ground-truth accuracy, channel name or physical unit is invented.
- Assignments are hard. Calibrated posterior assignment probabilities are not exposed.
- Representatives come from `predictive()`; their shaded covariance is the returned
  observation covariance, not a new total posterior or forecast uncertainty.
- `posterior()` supplies latest stored latent means/covariances. Private A/C histories,
  transitions and warp caches are intentionally not read. Notebook 05's visible warp
  is a separate public-component fit of two real beats, not a claim about cached model warps.
- The two-iteration offline budget keeps the teaching run bounded and does not
  establish convergence. The inventory's three-iteration empty-cluster failure is
  historical 0.2 evidence. Evaluate larger budgets against the current version
  separately and retain their logs.

## Automatic validation

```sh
python tools/validate_tutorials.py --output tutorials/.validation/new-run
```

Every code cell executes, including setup, in order. Each notebook runs twice in
fresh host/worker kernels. Public-import/private-attribute audits run before execution.
The validator compares exact JSON hashes of configuration, input, assignments,
representatives, latent means/covariances, online events and pairwise warp outputs;
timing/environment strings are excluded from scientific equality. Executed notebooks
retain plots and logs under the requested output directory. A summary includes
end-to-end and inference times and flags runs exceeding five CPU minutes.

See the [current documentation/CI audit](../docs/documentation_audit.md) for this
revision's executed checks; [validation.md](validation.md) preserves the original
0.2 preparation record. Local notebook execution does not establish hosted Colab
service availability or cross-platform bitwise equality.
