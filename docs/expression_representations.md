# Expression representation experiments

The expression workflow supports learned MFA emissions and fixed PCA or
autoencoder decoders. Dataset preparation, HMM
assignments, state comparisons, reference segmentations and the standard result
schema are shared.

| Dataset | PCA entry point | Autoencoder entry point |
| --- | --- | --- |
| Zebrafish | `examples/zebrafish_pca_clustering.py` | `examples/zebrafish_autoencoder_clustering.py` |
| Xenopus | `examples/xenopus_pca_clustering.py` | `examples/xenopus_autoencoder_clustering.py` |

```sh
python examples/zebrafish_pca_clustering.py --download
python examples/zebrafish_autoencoder_clustering.py --download --epochs 250
python examples/xenopus_pca_clustering.py --download --clutches A B
python examples/xenopus_autoencoder_clustering.py --download --clutches A B --epochs 250
```

The Xenopus autoencoder entry script defaults to both clutches, dynamic states,
the `more` HDP preset, left-to-right assignments, process variance 0.1, LDS
degrees 3, 10 latent components and up to 200 preliminary PCA components.
Observation covariance multipliers are **0.01 for clutch A** and **0.005 for
clutch B**. The equivalent commands with explicit settings are:

```sh
python examples/xenopus_autoencoder_clustering.py --clutch A --sigma-prior 0.01
python examples/xenopus_autoencoder_clustering.py --clutch B --sigma-prior 0.005
```

Running that script without arguments selects these settings; individual CLI
arguments override them. An explicit `--sigma-prior` applies to every selected
clutch, including when its value is 0.01. Each run saves the effective value in
its settings. The shared defaults described below still apply to
the other expression entry scripts.

## Observation models

PCA calibrates the existing affine Gaussian observation model from principal
directions, a mean offset, latent variances and diagonal reconstruction noise.
Its decoder C and offset remain fixed during clustering.

The autoencoder is the package's **tied linear orthonormal autoencoder**. It is
trained before clustering, then its affine decoder and offset are frozen. This
uses the existing exact Gaussian LDS inference; it does not add a nonlinear
decoder. Calibration is performed once per selected time course and shared by
both configurations, without reference labels.

Both fixed-decoder experiments default to fixed calibrated observation noise.
`--sigma-prior` is a covariance multiplier (shared default **1**; Xenopus
autoencoder **0.01 for A, 0.005 for B**); `--sigma-mode`
selects supported fixed, cluster, diagonal or shared covariance policies.
These fixed-decoder scripts do not use MFA's learned-C/shared-emission controls.

The independent configuration fixes **A=0** and learns **Gamma per cluster**,
so independent latent states can still have cluster-specific covariance. The
dynamic configuration learns both A and Gamma. Both initialize latent state
covariance from decoder calibration and use the same HMM priors. This differs
from independent MFA, which fixes Gamma=I while learning its emission maps and
cluster means.

## Dimensions and arguments

All endogenous genes remain in dataset preparation. Optional preliminary all-gene PCA
(`--projection-components`, shared default **30**, Xenopus autoencoder **200**) controls observation dimension and
caps itself at numerical rank.

For the fixed-decoder scripts, `--components`
(default **10**) controls the fixed decoder's latent state dimension.
`--projection-components 0` bypasses the preliminary projection; fixed PCA or
autoencoder calibration still performs the representation requested by its
entry script. Direct gene-space calibration retains the existing dense
observation-covariance memory and computation costs.

Shared controls include `--configurations independent dynamic`,
`--projection-components`, `--sigma-prior`, `--gamma-prior`, `--max-clusters`,
`--iterations`, `--hdp-preset`, `--transition-topology`, plotting switches and
`--resume`. Autoencoder controls additionally include `--epochs`/`--ae-epochs`,
`--learning-rate` (default 0.02), and `--noise-floor-fraction` (default 0.05).
All scripts support a `DEFAULTS` dictionary using CLI destination names.

Zebrafish retains replicate aggregation and its supplied developmental labels.
Xenopus retains separate clutches, published QC exclusions and explicitly
time-derived broad references. See [zebrafish](zebrafish_workflow.md) and
[Xenopus](xenopus_workflow.md) for source interpretation and temporal limits.

## Saved results

Default roots are `agents/zebrafish_pca_runs/`,
`agents/zebrafish_autoencoder_runs/`, `agents/xenopus_pca_runs/` and
`agents/xenopus_autoencoder_runs/`. Each uses the existing content-derived run
folders, root results table and per-dataset comparison plots, including the
reference segmentation above inferred states.

In addition to prepared data, configuration, assignments, metrics, logs and LDS
histories, each run saves:

- `observation_calibration.npz`: preliminary projection, observations and the
  applied fixed decoder under `decoder_*` keys.
- PCA: `pca_calibration.npz` with calibration indices, latent variances and
  calibration explained variance.
- Autoencoder: `training.npz` with calibration indices and loss history, plus
  `autoencoder_calibration.png` when plots are enabled.

Both arms retain exactly the same observations and representation settings.
MFA posterior files are specific to MFA. Successful complete resumed runs skip
decoder training and model inference. These artifacts are
calibration/diagnostic records, not restart checkpoints. No held-out predictive
performance follows from the in-sample clustering or calibration scores.
