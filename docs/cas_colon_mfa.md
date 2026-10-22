# CAS-Colon with factor-analysis emissions

This experiment fits the existing `HDPMFA` model directly to grayscale image
vectors. Every cluster learns its own image mean and loading matrix:

\[
y_i=\mu_{s_i}+W_{s_i}z_i+\epsilon_i,\qquad
z_i\sim N(0,I),\qquad\epsilon_i\sim N(0,\operatorname{diag}\psi).
\]

The diagonal pixel-noise vector is shared across clusters. No PCA or autoencoder
is calibrated first. `--components` controls factor rank, while
`--initial-clusters` controls initial variational support. Both image means and
loadings have Gaussian posteriors and contribute their parameter KLs to the
objective. See [MFA mathematics](factor_analysis_emission.md) and the
[HDP-HMM-MFA model](hmm_mfa.md).

## Run

Install the package and its image extra (`pip install -e ".[images]"`), then
extract the same official `Frame.rar` and `Label.rar` used by the
[CAS-Colon PCA experiment](cas_colon_pca.md). The default root is
`agents/data/cas_colon/CAS-ColonDB`.

```bash
python examples/cas_colon_mfa_clustering.py --records 1 --max-series 150
python examples/cas_colon_mfa_clustering.py --records 1 --components 8 --image-size 24 24 --hdp-preset more_sticky
python examples/cas_colon_mfa_clustering.py --records all --resume
```

HMM regime assignments are enabled by default. Use `--no-hmm-switch` for a DP
mixture of independent images. In HMM mode only the regime sequence is Markov;
continuous factors do **not** propagate through an LDS. Their coordinates are
component-local and cannot be compared or averaged across clusters without
aligning the bases. Use the existing PCA/autoencoder HDP-GPC runners when
continuous cluster-specific latent dynamics are the experimental target.

The existing loader orders frames by filename seconds, deduplicates identical
images, and retains conflicting anatomical annotations as ambiguous. Each
contiguous segment is fitted independently. `--gap-policy error` rejects gaps;
the default `split` starts a new model at each gap. `--frame-step N` subsamples
by filename second. `--crop LEFT TOP RIGHT BOTTOM` crops before grayscale
conversion and resizing. `--max-series` limits selected frames per video before
segmenting. Short segments reduce initial support to their frame count; factor
rank stays as requested.

## Main controls

| Argument | Default | Meaning |
| --- | --- | --- |
| `--components` / `--factors` | 4 | Factor dimension per component |
| `--initial-clusters` | 4 | Initial support slots, possibly MAP-unoccupied |
| `--max-clusters` | 12 | Upper bound on adaptive component support |
| `--component-strategy` | `birth_prune` | Objective-gated birth/pruning; `fixed` keeps initial support |
| `--iterations` | 100 | Maximum coordinate-ascent sweeps |
| `--hdp-preset` | `balanced` | Existing HDP concentration/stickiness bundle |
| `--loading-prior-precision` | 0.1 | Gaussian row-prior precision for both mean and loadings |
| `--loading-prior-mean` | 0 | Gaussian prior mean for augmented row entries |
| `--noise-initial` | 0.01 | Initial pixel **variance** on grayscale [0,1] data |
| `--noise-floor` | 0.000001 | Lower bound on learned pixel variance |
| `--no-learn-noise` | off | Fix shared diagonal noise at its initial variance |
| `--chunk-size` | 256 | Chunk size for local factor calculations |
| `--frames-per-cluster` | 4 | Random observed/reconstructed pairs in the gallery |
| `--min-cluster-samples` | 1 | Minimum MAP occupancy for gallery rows only |
| `--verbose` | `progress` | `silent`, `progress` or `debug` |

Use `--help` for tolerance, proposal refinement/exploration, seed, thread and
output controls. The noise value is an initialization (or fixed value), not an
inverse-Wishart prior. This MFA model does not have the GP/LDS Sigma and Gamma
policies. Gaussian posteriors for augmented loading rows require storage
proportional to `clusters * pixels * (factors + 1)^2`; begin with modest image
size, factor rank and support when exploring runtime.

## Saved results

Runs use the existing content-derived folder convention:
`agents/cas_colon_mfa_runs/video_<id>_segment_<segment>/<run_id>/`.
`--resume` skips a completed matching run. Nothing is shown interactively; use
`--no-plots` to omit PNG generation.

- `clustering.log`, typed `config.yaml`, provenance `settings.json`.
- `model.json`, reloadable with `HDPMFA.load(path)`.
- `assignments.csv`, preserving frame seconds, filenames, ambiguous annotation
  options, source variants, clusters and evaluation inclusion.
- `mfa_parameters.npz`: cluster means/loadings, shared noise, responsibilities,
  occupancies, stored-MAP local factor means, per-component factor covariance,
  ELBO history, times and, in HMM mode, transition/start probabilities.
- `history.json`: coordinate-ascent and birth/pruning history.
- `results.json`, `confusion_matrix.json`, and a root `results.csv` summary.
- `clusters.png`: regimes over time, reference anatomy, majority-label regimes,
  objective, a **display-only common pixel PCA projection**, and transitions
  (or occupancies for the independent mixture).
- `image_gallery.png`: cluster mean, marginal pixel SD, random frames and their
  `mu_k + W_k E[z_i]` reconstructions. SD includes standard-normal factor
  variation, loading-posterior uncertainty and pixel noise; it is not a
  confidence interval for the mean image. Grayscale panels use [0,1] display
  limits; saved model values are not clipped.

Labels never enter fitting, initialization or proposal selection. Purity, ARI,
NMI and the true-versus-majority-label confusion matrix are descriptive
**in-sample** scores over unambiguous frames. Ambiguous frames still participate
in inference. A segment with no unambiguous labels retains its model/results
without reporting label metrics. Gallery filtering does not change inference
or evaluation.
