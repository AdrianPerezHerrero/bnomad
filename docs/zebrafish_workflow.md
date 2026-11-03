# Zebrafish temporal expression states

Cluster chronological expression snapshots using every measured gene, with
the ordinary native HDPGPC engines. The paired configurations use the same
HMM assignments and MFA emission priors; only factor evolution changes.

```sh
python examples/zebrafish_mfa_clustering.py --download
python examples/zebrafish_mfa_clustering.py --resume
```

Install the core package with YAML support (`pip install '.[yaml]'`). No
single-cell libraries, image packages or raw sequencing tools are needed.

## Data, chronology and targets

Source: White et al. (2017), *A high-resolution mRNA expression time course of
embryonic development in zebrafish*, [doi:10.7554/eLife.30860](https://doi.org/10.7554/eLife.30860).
Accessions: **E-ERAD-475 / ERP014517**.

`--download` caches versioned supplementary files 1 (metadata) and 2 (full
gene counts) in `agents/data/zebrafish/`, verifying pinned SHA256 checksums.
It also verifies existing cached files and does not overwrite them. Without
`--download`, both files must exist under `--data-dir`. Every run records
actual source hashes; local loading permits separately supplied source copies.

All **32,105 genes** in the count table are retained, including **3,769 genes
with zero counts in all samples**. No selection uses expression, variance,
biotype or labels. Five HTSeq bookkeeping rows (`__no_feature`, etc.) are saved
separately: they are not genes and do not enter normalization. The TPM
supplement is deliberately unused because it already contains an
expression-filtered subset of 23,642 genes.

There are 90 RNA-seq samples: five biological replicates at each of 18 ordered
stages. Metadata/count columns are joined by `sample` ID, not table position.
DeTCT metadata belongs to a different assay and is excluded. Missing samples,
duplicate IDs, malformed counts and nonfinite/negative values raise errors.

Targets are fine stage names and eight broader developmental categories:
Zygote, Cleavage, Blastula, Gastrula, Segmentation, Pharyngula, Hatching and
Larval. Both are scored after fitting; `--target broad` chooses the primary
reporting target. The 18 fine labels each describe one aggregated observation,
so fine-target ARI and purity have limited value. Labels never enter projection
or inference. Developmental annotations need not coincide with expression
regime boundaries.

Approximate hours post-fertilisation follow Figure 1A (including 0 h for
1-cell, 4.3 h for Dome, 10.3 h for 1–4 somites and 19 h for 20–25 somites).
Intervals are unequal. These are stage-associated times, not repeated
measurements of individual embryos.

## All-gene representation

1. Compute CPM using each sample's total assigned **gene** count, then natural
   `log1p(CPM)`. Preserve zero genes; exclude HTSeq bookkeeping from the sum.
2. Average transformed replicates within each stage, retaining all gene
   columns. Save between-replicate SDs and counts as diagnostics.
3. Optionally fit centered thin-SVD PCA to these stage means, with
   positive `--projection-components` dimensions, capped at numerical rank.
   Use one common score scale so average retained component variance is one;
   neither whiten components nor standardize genes separately.
4. Fit learned affine MFA emissions to the chosen observations with
   `--components` factors (default 10).

`--projection-components 0` skips the preliminary PCA entirely:
the stage-mean log1p(CPM) gene vectors enter MFA unchanged, without centering,
rescaling or dimensionality reduction before MFA. The spectral MFA prior
initializer still runs; it initializes loadings and does not replace these
gene-space observations. Positive values must exceed `--components`. The
current shared default is 30 projected dimensions and 10 MFA factors.

Enabling the projection avoids dense gene-by-gene covariance allocations in the current
affine emission implementation. Its loadings keep one row per gene, including
constant/zero genes, and support approximate reconstruction. Native MFA
provides a second reduction and learns observation noise. High variance can
reflect technical effects: reduction does not guarantee removal of irrelevant
variation. Explained variance describes retrospective calibration on the fitted
sequence, not held-out performance. With projection disabled, the existing
affine emission implementation allocates dense covariances over all 32,105
genes, so direct gene-space MFA has substantial memory and computation costs.

Simultaneous replicates produce **18 temporal observations**, not 90 artificial
steps. Their SD is not automatically supplied as known measurement noise. The
LDS advances between successive observations included in a cluster; it does
not automatically propagate according to elapsed experimental hours.

## Configurations and shared CLI conventions

| Configuration | Assignment process | Factor process |
| --- | --- | --- |
| `independent` | HMM by default | Fixed A=0, Gamma=I; independent standard-normal factors |
| `dynamic` | Same HMM priors | Cluster-learned A and Gamma |

Both arms receive identical observations, spectral MFA initialization,
emission priors, seed, inference budget and cluster cap. Shared learned C and
diagonal Sigma are the default, with cluster-specific means.
`--no-shared-emission` selects cluster-local emissions in both arms. The runner
explicitly preserves HMM assignments for independent factors, overriding the
general MFA helper's independent-assignment convenience default. No native
inference formulas or helper defaults are changed.

```sh
python examples/zebrafish_mfa_clustering.py --components 3 --projection-components 12
python examples/zebrafish_mfa_clustering.py --projection-components 0
python examples/zebrafish_mfa_clustering.py --transition-topology left_to_right
python examples/zebrafish_mfa_clustering.py --configurations dynamic
python examples/zebrafish_mfa_clustering.py --no-hmm-switch
```

The runner reuses `add_common_arguments`, `ExperimentRun`, `save_results`,
root results-table updates and the common MFA posterior exporter. Defaults:
offline split/merge, soft assignments, balanced HDP, 30 iterations, five
exploration steps, one refinement pass and a ten-cluster cap. The cap is not
chosen from the target labels. Flexible factors or strict priors may yield one
cluster; this is not evidence of recovered developmental states.

Shared controls include `--offline-strategy`, `--iterations`, `--max-clusters`,
`--hdp-preset`, `--split-merge-refine-steps`, `--exploration-steps`,
`--hard-assignments`/`--no-hard-assignments`, `--lds-degrees`, `--lds-update`,
`--seed`, `--plots`/`--no-plots`, `--projection-plots`, `--lds-plots`,
`--lds-dynamics-plots`, `--min-cluster-samples` and `--resume`.
Representation controls include `--sigma-prior`, `--gamma-prior`,
`--noise-floor`, `--mean-precision`, `--loading-precision`,
`--calibration-samples` (MFA initialization only), `--trajectory-plots`, and
`--max-series` (chronological stage prefix).

This runner requires offline inference, no alignment/GP fitting and final LDS
scoring. Shared MFA requires conjugate batch updates. Resume skips successful
complete artifacts; it does not restart interrupted inference.

## Saved results and package ownership

```text
agents/zebrafish_mfa_runs/
  results.csv
  comparison.json
  comparison.png
  zebrafish_development/<configuration-and-content-derived-run-id>/
    settings.json, config.yaml, clustering.log, results.json
    assignments.csv, label_metrics.json
    prepared_data.npz, observation_calibration.npz
    lds_parameters.npz, mfa_posteriors.npz, mfa_shared_posterior.npz
    latent_trajectories.npz
    clusters_output_0.png, projection.png, latent_trajectories.png
```

Figures follow plotting switches; the shared posterior exists only in shared
mode. `prepared_data.npz` saves every gene ID/name, transformed stage means,
replicate SD/counts, times/targets, source sample IDs, assigned library sizes
and HTSeq counters. `observation_calibration.npz` saves the all-gene center,
loadings, scale, variance ratios, rank, projected observations and applied MFA
prior when PCA is enabled. With PCA disabled it records `projection_applied=False`,
the unchanged gene-space observations and the MFA prior, omitting PCA-specific
arrays. Reports record `projection_components=0`, the actual
`observation_dimension`, and null PCA explained variance. MFA NPZs are
diagnostics, not training checkpoints.

Assignments follow the common schema, adding stage, category, hours and
replicate count. Both target metrics appear in `label_metrics.json` and report
metadata. The root CSV uses the primary target. `comparison.json` and its
figure summarize the current invocation; per-run folders and `results.csv`
retain earlier runs. Use separate output roots for separate comparisons.
The comparison figure includes the original broad reference segmentation above
the inferred states, with aligned axes and dashed reference boundaries.

ARI/NMI compare partitions. Purity and majority-label confusion matrices use
an in-sample mapping, not held-out classification. Compare ordered results
with simple time-based partitions before claiming biological state discovery.

Reusable extraction/aggregation lives in `datasets/zebrafish.py`; compact
all-column projection in `preprocessing.py`; shared execution in
`experiments/_expression.py`, with dataset dispatch in `experiments/zebrafish.py`;
figures in diagnostics. [Xenopus](xenopus_workflow.md) uses the same execution
path with its own extraction and annotation policies.

Fixed [PCA and autoencoder variants](expression_representations.md) use the same
dataset preparation and shared experiment runner through dedicated scripts.
