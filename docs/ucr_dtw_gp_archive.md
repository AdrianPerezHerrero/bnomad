# UCR direct-waveform DTW-GP experiment

[The direct-waveform runner](../examples/ucr_dtw_gp_archive.py) fits the 2018
UCR archive without an autoencoder. It reads the standard TRAIN/TEST pairs
directly from the official ZIP, interpolates missing points, resamples to a
common grid, and normalizes using TRAIN waveforms only. Labels are used only
for descriptive clustering scores after fitting.

```powershell
# Download the archive if absent, then fit all series in all 128 datasets.
python examples/ucr_dtw_gp_archive.py --download --resume

# A smaller diagnostic run.
python examples/ucr_dtw_gp_archive.py --datasets Coffee CBF --max-series 30 --max-points 32 --iterations 1

# Reproduce the original tight DTW-GP settings after the no-alignment comparison.
python examples/ucr_dtw_gp_archive.py --datasets ACSF1 Adiac --alignment dtw_gp --dtw-amplitude 0.005 --dtw-anchor-noise 0.0005
```

The runner temporarily defaults to `--alignment none` to compare births without
warping. Use `--alignment dtw_gp` to restore the existing fast deterministic
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

For offline DTW-GP fits, `--birth-plots` saves the first four attempted birth
diagnostics by default; set it to 0 to disable them or to a larger number to
inspect more proposals. Each PNG compares the raw series and the aligned
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
`agents/ucr_dtw_gp_results` directory. `--resume` uses the archive hash and
model settings, and skips only completed runs whose requested summary and
cluster PNGs still exist. `--no-plots` avoids PNG generation.

The final clusters also get `plots/<dataset>_<run-id>_clusters_output_0.png`,
using the same `plot_record_results` function as MIT-BIH. Its first row shows
assignments in fit order, all fitted observation means, and true-class
composition. The gallery below has three columns with one panel per final
cluster: member waveforms coloured by their true class, the fitted mean
coloured by the cluster's majority class, and the predictive +/-2 SD band.
Panel sizes stay fixed as additional cluster rows increase the figure height.
The CSV's `cluster_plot` field points to this figure. No temporal transition
matrix or latent/Gamma overlay is shown for UCR. With alignment active, member
curves show original inputs and model bands use aligned coordinates, as the
figure heading states. Plots are saved and closed, without opening a window.

GP kernel fitting uses the package's first-series optimizer, which can cost
much more than the DTW-GP alignment itself. The cheap setting is
`--fixed-gp-hyperparameters`; the GP representer still updates, though the
kernel stays at its supplied values.
