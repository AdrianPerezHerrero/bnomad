# UCR direct-waveform DTW-GP experiment

[The direct-waveform runner](../examples/ucr_waveform_clustering.py) fits the 2018
UCR archive without an autoencoder. It reads the standard TRAIN/TEST pairs
directly from the official ZIP, interpolates missing points, resamples to a
common grid, and normalizes using TRAIN waveforms only. Labels are used only
for descriptive clustering scores after fitting.

```powershell
# Download the archive if absent, then fit all series in all 128 datasets.
python examples/ucr_waveform_clustering.py --download --resume

# A smaller diagnostic run.
python examples/ucr_waveform_clustering.py --datasets Coffee CBF --max-series 30 --max-points 32 --iterations 1

# Reproduce the original tight DTW-GP settings after the no-alignment comparison.
python examples/ucr_waveform_clustering.py --datasets ACSF1 Adiac --alignment dtw_gp --dtw-amplitude 0.005 --dtw-anchor-noise 0.0005
```

The runner defaults to `--alignment dtw_gp`. Use `--no-warp` or
`--alignment none` for an unaligned comparison. The existing fast deterministic
DTW path smoothed by a GP displacement prior; its warp penalty enters the
observation score. The alignment policy has its
own CLI controls (`--dtw-points`, `--dtw-inducing`, `--dtw-lengthscale`,
`--dtw-amplitude`, and `--dtw-anchor-noise`).

Offline alignment uses each cluster's selected observation (`f_ind_old`) as
its template. The initial template is the first shuffled series. Each birth
aligns against its own proposed seed; fitting the GP does not replace that
template with a new GP mean. The GP still supplies the observation score,
and the warp penalty is included from the first pass onward.
Proposal LDS KL is divided by the latent LDS dimension by default. Use
`--no-birth-lds-kl-per-dimension` to disable this normalization. The setting
is included in the run ID so resumed results keep the two scoring choices distinct.

For offline fits, `--birth-plots N` saves the first N attempted birth
diagnostics. The default is 0; use `--birth-plots 4` for four proposals. Each PNG compares the raw series and the aligned
series under three warps: the one used by the existing baseline score, the
one recomputed for the proposed parent, and the newborn warp. It also shows
the GP means, input maps, and per-series GP data score and warp penalty.
Rejected births are captured before their temporary cluster state is removed.
The heading reports the proposal's total score change and emission change.
Birth plots are saved beside the dataset summary PNG. They use fit-order
series numbers after the seeded permutation, and display the seed's true label
only as post-fit context. Online schedules do not currently capture birth
proposal plots.

The matrix policies are explicit:

| Component | UCR policy |
| --- | --- |
| A | Fixed zero matrix: no prediction from the previous archive row |
| Gamma | Fixed `--gamma` times identity (default 0.1) |
| C | Fixed identity |
| Sigma | Learned per cluster, full inverse-Wishart by default; `--sigma-mode cluster_diagonal` selects diagonal noise |
| GP | Per-cluster representer learns from assigned series; kernel hyperparameters are fitted by default |

`hmm_switch=False` selects independent HDP assignments. The runner applies a
seeded permutation to the fitted series so the archive's arbitrary row order
does not choose the GP calibration series. The first series still initializes
the GP and the LDS has a separate initial-state prior, so the procedure is
not exactly invariant to permutation. `--fixed-gp-hyperparameters` skips
kernel fitting but retains GP posterior/representer updates.

The default fits all TRAIN and TEST series unsupervised, so the reported TEST
ARI is **transductive**, not held-out classification accuracy. Each successful
dataset gets a summary PNG showing a two-dimensional PCA view of its input
waveforms colored by fitted cluster and true label, plus a within-cluster
class-composition panel. The CSV, PNGs, and error logs go under the ignored
`agents/ucr_waveform_results` directory. `--resume` uses the archive hash and
model settings, and skips only completed runs whose requested summary and
cluster PNGs still exist. `--no-plots` avoids PNG generation.

Each dataset/settings combination gets `<dataset>/<run_id>/`, containing
`clusters_output_0.png`, `projection.png`, settings, resolved configuration,
log, assignments, matrices and JSON metrics. With alignment enabled,
`warps_output_0.png` is also saved. Enable `--lds-plots` and
`--lds-dynamics-plots` for the original detailed LDS diagnostics. The gallery
uses the same `hdpgpc.diagnostics.plot_clustering_summary` as ECG; unordered
UCR data show class composition rather than a transition matrix. See the
[common workflow guide](experiment_workflows.md) for the complete output schema.

GP kernel fitting uses the package's first-series optimizer, which can cost
much more than the DTW-GP alignment itself. The cheap setting is
`--fixed-gp-hyperparameters`; the GP representer still updates, though the
kernel stays at its supplied values.
