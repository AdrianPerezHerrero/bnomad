# UCR archive autoencoder experiment

The [archive script](../examples/ucr_autoencoder_archive.py) applies the same
tied, orthonormal **linear** autoencoder and fixed affine Gaussian decoder as
the ECG autoencoder example to the 2018 univariate UCR classification
archive. The [official archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
has 128 standard TRAIN/TEST dataset pairs. The script selects those pairs
inside the ZIP and ignores its separate adjusted-data folder.

```powershell
# Download the official ZIP to the ignored agents/data/ucr directory and run all 128.
python examples/ucr_autoencoder_archive.py --download

# A shorter diagnostic run across every dataset; only 50 series are fitted per set.
python examples/ucr_autoencoder_archive.py --max-series 50 --max-points 32 --ae-epochs 10 --iterations 1 --max-clusters 2

# Inspect one dataset without re-running completed successful datasets.
python examples/ucr_autoencoder_archive.py --datasets Coffee --resume
```

The default processes **every** TRAIN and TEST series in all 128 datasets.
It trains the autoencoder, noise scale, and global normalization using TRAIN
only, then fits HDP-GPC unsupervised to the concatenated TRAIN and TEST
series. Missing points are linearly interpolated per series, and each series
is represented on at most 64 equally spaced points by default. This
resolution cap can be changed with `--max-points`; the original ZIP is not
altered. A failed dataset gets an error row and traceback while later
datasets continue. `--resume` skips datasets already marked successful in
`results.csv` under the same archive hash and experiment settings. Each row
records that run ID, so changing model controls cannot silently reuse an old
result.

Each successful dataset produces `plots/<dataset>_<run-id>_summary.png` under
the output directory. The figure shows every series in a two-dimensional PCA
view of its autoencoder code, coloured first by fitted cluster and then by
UCR class. A third panel shows the true-class composition within each fitted
cluster. The figure header includes the number of series and clusters, purity,
ARI, and NMI. No LDS or transition matrix is plotted. Use `--no-plots` to skip
PNG generation; `--resume` also checks that a requested PNG exists.

The ZIP is about 316 MB; a complete model fit is substantially more expensive
than downloading it. `results.csv` and error logs are saved under the ignored
`agents/ucr_autoencoder_results` directory by default. The reported purity,
ARI, and NMI are **descriptive clustering** scores. TEST waveforms participate
in fitting, so `ari_test_transductive` is not a held-out classification
estimate. The UCR archive documentation says row order generally has no
meaning; the script disables the HDP Markov transition (`hmm_switch=False`),
fixes the LDS transition matrix to zero, and initializes a learned Gamma
posterior separately for each cluster from the autoencoder calibration
covariance. The latent prior is therefore
independent of the preceding archive row. The observation decoder and noise
are also fixed after calibration. These numbers remain a pipeline experiment,
not a competitive UCR classification benchmark.
The first series still uses the LDS initial-state prior, so this configuration
removes predictive dependence on the previous row without claiming exact
permutation invariance of every inference update.

An earlier experiment that also fixed Gamma produced one cluster on all 128
datasets in a 20-series, one-iteration diagnostic pass, and on full-series
Coffee and CBF. That result motivated leaving Gamma trainable. It does not
describe the current configuration's clustering performance.

With cluster-learned Gamma and the current full-series defaults, CBF fitted
five clusters across its 930 series (ARI 0.228, NMI 0.343). Coffee still fitted
one cluster. These are descriptive, transductive scores.

`--latent-covariance-scale` (also accepted as `--process-noise` for the earlier
script interface) multiplies the calibrated latent covariance before it is
used as the initial-state covariance and each cluster's Gamma prior. Gamma is
then updated during fitting. The scale's default is 1.
