# CAS-Colon: ordered image clustering with PCA

This first image experiment uses the existing `HDPGPC(config).fit(values, grid=grid)`
model. A fixed affine PCA decoder maps latent states to resized image pixels;
cluster-specific A/Gamma describe their evolution. Anatomical labels are used
only after fitting for evaluation and plots.

For the same workflow with a trained tied linear encoder/decoder, see the
[CAS-Colon autoencoder example](cas_colon_autoencoder.md).
For calibration that uses temporal order explicitly, see the
[Linear SFA example](linear_sfa_observation.md).

## Prepare the dataset

Download **Frame.rar** and **Label.rar** from the
[official CAS-Colon repository](https://doi.org/10.6084/m9.figshare.28287929)
and extract them. The frame archive is about 4.15 GB; raw video archives are not
needed. This runner reads extracted frames, without downloading archives or
decoding videos automatically. It expects:

```text
agents/data/cas_colon/CAS-ColonDB/
  Frame/
    1_terminalIleum_0s.jpg
    ...
  Label/frame/Frame_Label.csv
```

Image files may be in subdirectories of Frame. An annotation table elsewhere can
be supplied with `--annotations`. Install the optional image dependency with
`pip install -e ".[images]"` if Pillow is absent.

The [dataset paper](https://www.nature.com/articles/s41597-025-05588-3) describes
78 withdrawal-phase videos with ten anatomical regions. The frame CSV is
grouped by label, **not chronological order**. The loader sorts by the acquisition
second in filenames, validating video IDs and labels. Seconds are one-second
filename bins, not exact subsecond camera timestamps.

## Run one video

```powershell
python examples/cas_colon_pca_clustering.py --records 1
python examples/cas_colon_pca_clustering.py --records 1 --image-size 24 24 --components 8 --max-series 150
python examples/cas_colon_pca_clustering.py --records 1 2 --sigma-mode cluster_diagonal --sigma-prior 1
python examples/cas_colon_pca_clustering.py --records all --resume
```

Defaults:

- 32x32 grayscale images in [0,1]. `--crop LEFT TOP RIGHT BOTTOM` removes a
  chosen source-pixel rectangle before resizing, useful for borders/overlays.
- PCA dimension 10, calibrated on up to 400 uniform frames from the selected
  sequence; dimensions are capped if calibration has fewer frames.
- Fixed PCA mean/C and calibrated diagonal Sigma; A/Gamma learn per cluster.
- HMM enabled; offline `conjugate_batch`, final scoring, hard assignments.
- Split/merge search, one refinement pass and three exploration slots per
  iteration. Common iteration/stopping controls remain available.

`--sigma-prior` multiplies calibrated Sigma; 1 preserves it. `--gamma-prior`
sets the process covariance center to that scalar times identity (default 0.01
for [0,1] pixels). Other supported Sigma policies can be selected. Full learned
pixel Sigma costs much more and can absorb morphology as correlated noise;
start with fixed or diagonal observation noise when interpreting state dynamics.

`--frame-step N` selects one frame every N filename seconds. `--max-series N`
takes a chronological prefix. Time gaps create **independent segment runs**
with fresh models; `--gap-policy error` instead requires a complete sequence.
Isolated single-frame segments are skipped, with a message outside silent mode.
Missing image files raise errors; missing time is not silently compressed into
adjacent model steps. Official annotations contain four timestamps with two
anatomical labels (video 9: 42s; video 15: 10s; video 19: 326s and 327s).
Their image copies are identical. The loader verifies byte or decoded-pixel
identity and includes the image only once. Conflicting labels become `ambiguous`:
the frame remains in model fitting and trajectory plots but is excluded from
label-based scores and majority-label mapping. Different images at the same
timestamp still raise an error. No anatomical label is inferred from CSV order.

## Trajectory visualizations

```text
agents/cas_colon_pca_runs/video_1_segment_00/<run_id>/
  settings.json, config.yaml, clustering.log
  assignments.csv, results.json
  observation_calibration.npz, lds_parameters.npz
  latent_trajectories.npz
  clusters_output_0.png
  latent_trajectories.png
  projection.png
```

**Image summary (`clusters_output_0.png`)** includes purity, ARI and NMI in its
header. Its taller timeline section shows anatomical labels, inferred regimes
colored by cluster, and inferred regimes colored by each cluster's majority
reference label. Label colors match the anatomical reference plot; ambiguous
frames do not vote for a majority and clusters without reference labels are gray.
The other top panels show state paths in the first two common PCA coordinates
and empirical consecutive-frame transitions. Below, a gallery per
cluster has four aligned image rows: observed frames, ordinary PCA projection,
smoothed LDS reconstruction, and final-A one-step reconstruction. This separates
compression error from posterior shrinkage and predicted evolution. Up to ten
distinct frames per cluster are sampled randomly using `--seed` and displayed
in chronological order. Use `--frames-per-cluster N` to change the gallery size;
small clusters show their available observations without duplication. The figure
grows with the number of clusters while retaining panel dimensions. Only display
is clipped to [0,1]; model states and saved arrays are not clipped.

**Latent trajectories (`latent_trajectories.png`)** has three panels per cluster:
state path with time-colored points and labeled start/end; the first three
components with +/-2 posterior SD; and orange `(A-I)m` arrows in the first two
coordinates. Paths break across MAP membership gaps. Arrows show a native
one-cluster-step prediction and omit components beyond the displayed two;
they are not measured camera motion or observed future frames. PCA coordinates
are shared and fixed, not independently rotated for each cluster.

`latent_trajectories.npz` saves included observation indices, smoothed means/
covariances, prior moments, one-step means and video seconds. Soft-assignment
histories may include observations belonging to another MAP cluster; plots show
MAP-owned states. Image shape and calibration indices are in report metadata.
Final A/C/Gamma/Sigma remain in the standard LDS artifact.
Assignments also record all `annotation_options`, `source_variants` and an
`evaluation_included` flag. Reports/logs identify how many frames were scored;
components containing only ambiguous frames have no majority reference label.

`--no-plots` disables all figures. `--no-trajectory-plots` disables the detailed
trajectory figure only. `--projection-plots` controls the standard embedding
summary. `--lds-plots` optionally adds matrix diagnostics. Figures are saved
and closed without opening GUI windows.

Reusable functions:

```python
from hdpgpc.diagnostics import (
    model_latent_trajectories, plot_latent_trajectories,
    plot_image_clustering_summary,
)
paths = model_latent_trajectories(model)
figure = plot_latent_trajectories(model, times=seconds, trajectories=paths)
figures = plot_image_clustering_summary(
    model, values, region_labels, image_shape=(32, 32), times=seconds,
    trajectories=paths,
)
```

## Interpretation and limits

This is **in-sample temporal clustering**. PCA sees the selected segment without
labels. Uniform calibration uses future frames; prefix calibration is available
with `--calibration-sampling prefix`, but clustering/smoothing remain offline.
No held-out or causal accuracy is claimed. Videos and discontinuous intervals
are not joined into one trajectory.

For independence comparisons use `--a-mode fixed` (A=0), and/or
`--no-hmm-switch` (independent assignments). These are separate controls. Fixed
Gamma (`--gamma-mode fixed`) requires A fixed. They use the existing typed
matrix/noise policies; default A/Gamma remain learned per cluster.

This first runner supports offline schedules retaining complete state histories.
Fixed affine restrictions remain: one flattened output, no 1D warp, no GP kernel
fitting, final scoring and non-legacy LDS updates. The flattened grid is a pixel
index for API compatibility, not a spatial GP distance. Low-resolution grayscale
PCA can discard information useful for anatomical recognition.

Native A advances between included observations of a cluster. Returning after
time assigned elsewhere does not propagate A for elapsed seconds automatically.
Anatomical correspondence and physical interpretations of A must be evaluated,
not assumed. Inspect time-resolved assignments and ARI/NMI; purity alone can be
misleading for imbalanced regions.

The loader was designed against the official annotation archive and authors'
[preprocessing code](https://github.com/krystal21/CAS-Colon/blob/main/pre.ipynb).
See `agents/cas_colon_duplicate_frames.md` for the subsequent duplicate-frame
diagnosis and validation against the locally downloaded images. No full archive
clustering benchmark is claimed.
