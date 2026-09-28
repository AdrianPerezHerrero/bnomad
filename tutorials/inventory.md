> Historical tutorial preparation record for 0.2. Current verification is in [the 0.3 report](../docs/migration_0_3/report.md).

# Tutorial inventory (before notebook creation)

Inspected the public `HDPGPC` facade, exported configuration dataclasses, packaged
recipes, public result types, alignment exports, tests, installation metadata and
both `hdpgpc/tests/test_step.ipynb` and `test_step_ocean.ipynb`.

The ECG notebook loads `hdpgpc/data/mitbih/100.npy`, selects 1700:1950, estimates
noise scales, initializes an RBF/white-noise GP, streams observations and later
performs batch inference. It plots waveforms, cluster representatives, assignments,
transition quantities, LDS covariance/mapping matrices and MDS. Those latter
private-state plots are not copied into the new facade tutorials. The old notebook
also unpacks two outputs from `compute_estimators_LDS`, whose current implementation
returns four. The tutorials therefore show explicit hyperparameters in scaled units
instead of copying that stale call or its raw-amplitude values.

## Data

- `hdpgpc/data/mitbih/100.npy`: float64 (2272, 90, 2), 3,271,808 bytes.
- `100_labels.npy`: (2273,), one-character annotations, 9,220 bytes. The unequal
  lengths prevent assuming verified alignment; annotations are not used for fitting
  or accuracy scores.
- Other MIT-BIH records and `hdpgpc/data/ocean/{data,freq}.npy` exist. They are not
  needed or downloaded by this suite.
- There is no loader exported by `hdpgpc.datasets`; the legacy loader embeds
  machine-specific paths. A tutorial helper uses `numpy.load(allow_pickle=False)`.

The common teaching subset is six consecutive beats beginning at 1700. Every sixth
point is retained (15 points); both original channels remain available. Channel
centering/scaling is fixed from the first beat only, avoiding future observations
in online preprocessing. This is explicit coarse subsampling, not PCA, an internal
model transform, calibrated physical units or a clinical analysis.

## Supported-mode matrix

| Variation | Public configuration / result | Tutorial |
| --- | --- | --- |
| Static, single output | `DynamicsConfig(model_type='static')` | 00, 01 |
| Cluster-specific LDS, single output | `model_type='dynamic'`; `posterior()` | 02 |
| Sequential online | `schedule='online'`; `update()` | 03 |
| Repeated batch inference | `schedule='offline'`; `fit()` | 04 |
| Offline GP alignment | `policy='gp'`, `warp=True`; standalone `GaussianProcessWarp` | 05 |
| Multiple channels | `n_outputs=2`; per-output predictions/states | 06 |
| Uniform versus SNR weighting | `use_snr=False/True` | 06 comparison |
| Fast online approximation | `schedule='online_fast'` | 07, explicitly distinct algorithm |
| HDP concentration presets | `hdp_hyp='less'/'balanced'/'more'` | Explicit balanced setting, exercises |
| Temporal assignment dependence | `hmm_switch=True/False` | Explained separately from static dynamics; exercise |
| Custom kernels, inducing basis | emission configuration fields | Kernel exercise; advanced basis choices documented, no unvalidated demo |
| ML versus Bayesian parameter estimates | `bayesian_params` | Bayesian main flow; explain ML is not fixed dynamics |
| Shared versus per-cluster C | No facade configuration for this | No notebook; `share_gp` is not this switch |
| Arbitrary fixed A/C dynamics | No reliable freeze switch in the facade | No notebook; `estimation_limit` is not a general freeze contract |
| PCA/custom representation transforms | Not implemented in the public package | No notebook; do not invent transform interfaces |
| Variance-based output reduction | `reduce_outputs` selects channels, not PCA | Document distinction; no extra representation notebook |
| Online warping/multi-output | Known characterized failures | Excluded from supported main flows |

Static dynamics initialize identity A/C and zero process covariance. This does not
automatically disable the HDP/HMM assignment transitions; the tutorials retain
`hmm_switch=True`. Filtered/smoothed public snapshots are available; A/C histories,
transition matrices and model-owned warp caches are not facade outputs. Hard
responsibilities are not calibrated posterior probabilities.

## Execution constraints found during inspection

GitHub `main` still points to original commit `615109a9` (checked 2026-09-27).
Installing it cannot supply the new facade. Supply an uploadable wheel/helper
bundle until the current package is published; do not claim remote publication.
Colab dependencies are isolated in a supported Python worker, leaving its host
kernel intact. The `%%hdpgpc` cell magic forwards ordinary Python and figures to
that persistent worker; model state remains between cells.

The six-beat nonaligned static/dynamic pilot hits an existing empty-cluster
reallocation `IndexError` in `inference/_offline.py:estimate_q_first` with three
outer iterations. Two iterations succeed. Main notebooks explicitly use a two-pass
budget and describe this limitation; no inference code or optimizer is patched,
and reaching that budget is not presented as convergence. Raising it is not a safe
unqualified exercise for this subset.
