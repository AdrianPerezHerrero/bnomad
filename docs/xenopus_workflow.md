# Xenopus temporal expression states

This denser developmental time course uses the same expression-state experiment
as zebrafish: HMM assignments with independent versus evolving MFA factors.

```sh
python examples/xenopus_mfa_clustering.py --download
python examples/xenopus_mfa_clustering.py --download --clutches A B --resume
```

## Source and extraction

Owens et al. (2016), *Measuring Absolute RNA Copy Numbers at High Temporal
Resolution Reveals Transcriptome Kinetics in Development*,
[doi:10.1016/j.celrep.2015.12.050](https://doi.org/10.1016/j.celrep.2015.12.050).
Public data: [GEO GSE65785](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65785).

The loader caches/checksums the unsmoothed polyA+ relative-TPM files for the
selected clutches under `agents/data/xenopus/`. It uses the measured table
headers for sample IDs and actual hours post-fertilisation. It does not load
the authors' GP median/CI files, concatenate clutches, interpolate observations
or invent a regular sampling grid.

Each source has 28,376 `Gene` rows and 43,661 `Isoform` rows. The latter are
alternate representations and would double count abundance. Of the gene rows,
100 are explicitly identified calibration controls: 92 ERCC and eight
ArrayControl (`AC_EC`) rows. Their TPMs/IDs are saved separately. All **28,276
endogenous gene rows** remain, including zero/constant and unnamed genes.
No expression- or variance-based gene selection is performed.

The input is `log1p(TPM)` directly; do not apply count-library normalization to
these already normalized values. Xenopus TPM and zebrafish CPM have different
source definitions. Optional all-gene PCA and MFA calibration otherwise follow
the common pipeline.

| Clutch | Measured samples | Default usable samples | Coverage |
| --- | --- | --- | --- |
| A | 90 | 87 | 0–66 hours |
| B | 49 | 48 | 0–24 hours |

Default QC exclusions reproduce the paper's supplementary methods: A at 21.5,
33 and 52 hours; B at 22.5 hours. `--include-qc-failures` (or
`--no-exclude-qc-failures`) retains them for sensitivity analysis. These are
sample QC exclusions, not gene selection. Each remaining observation represents
one measured sample, with no claim of replicated measurements at every time.
The two clutches are independent biological time courses and are always fitted
separately. Stage-specific latent histories/transition models are not pooled.

## Evaluation references

The paper's [supplementary PDF, Table S1](https://pmc.ncbi.nlm.nih.gov/articles/instance/4731879/bin/NIHMS746658-supplement-6.pdf)
maps observed Nieuwkoop–Faber (NF) stages to hours. These anchors are transcribed
in `datasets/xenopus.py` with the source PDF's SHA256. No PDF/Excel package or
supplement download is required at experiment runtime.

Two reference sets are reported:

- `broad` (default): **time-derived intervals** anchored at Table S1 times:
  NF1–6 before 3 h; NF7–9 from 3 h; NF10–12.5 from 6.5 h; NF13–21 from
  12 h; NF22–40 from 17.5 h; NF41+ from 51 h. These are approximate temporal
  references, not independently measured per-sample biological targets.
- `stage`: exact Table S1 anchors only. Combined labels preserve ambiguities:
  NF16/17 at 15 h, NF18/19 at 15.5 h, NF29/30 at 29 h, NF33/34 at 33 h.
  Other times are marked `unannotated` and excluded from fine-stage scores.
  No NF stage is interpolated or extrapolated past the final 56 h anchor.

After default QC, fine-stage scoring covers 35/87 observations in A and 22/48
in B. Broad references cover all observations. Both masks/label sets are saved;
`--target stage` changes primary reporting, not fitting. Labels do not enter
the projection, priors, cluster-count selection or inference.

Agreement with time-derived intervals is descriptive. Contiguous time bins can
also agree with these references, so agreement alone does not establish
biological state recovery. Fine targets mostly identify individual time anchors
and also provide limited clustering evidence. Inspect inferred boundaries,
gene-space reconstructions and sensitivity across clutches/settings.

## Arguments and results

The expression runner is shared in `experiments/_expression.py`, with small
dataset entry points in `experiments/zebrafish.py` and `experiments/xenopus.py`.
Extraction remains in dataset modules; plotting remains in diagnostics.
Current shared defaults preserve the zebrafish recipe: 30 PCA dimensions,
10 latent factors, Sigma multiplier 0.001, Gamma prior 0.01, soft assignments,
shared MFA emissions and offline split/merge. Both factor configurations get
identical observations/emission priors. `--projection-components 0` bypasses
PCA and retains the existing dense gene-space covariance costs.

```sh
python examples/xenopus_mfa_clustering.py --clutches A --configurations dynamic
python examples/xenopus_mfa_clustering.py --clutches A B --projection-components 24 --components 10
python examples/xenopus_mfa_clustering.py --transition-topology left_to_right
```

`--records` aliases `--clutches`. All expression/common controls are reused,
including `--components`, `--projection-components`, `--sigma-prior`,
`--gamma-prior`, `--max-series`, `--max-clusters`, `--iterations`,
`--hard-assignments`, plotting switches and `--resume`. `--max-series` selects
a chronological prefix separately in each clutch. Actual unequal sampling
times and gaps are retained; the native LDS still advances by cluster-member
observation steps rather than elapsed hours.

```text
agents/xenopus_mfa_runs/
  results.csv
  comparison_clutch_A.json, comparison_clutch_A.png
  comparison_clutch_B.json, comparison_clutch_B.png
  xenopus_clutch_A/<run_id>/
  xenopus_clutch_B/<run_id>/
```

Each run follows the standard schema: settings, full configuration, fitting log,
assignments, results, both target metrics, observation calibration, prepared
data, LDS/MFA posteriors and diagnostic figures. Prepared data retains all gene
IDs/names, measured TPM/log-TPM arrays, times/sample IDs, NF targets/masks,
control abundances, original times and excluded QC times. Per-clutch comparison
files prevent one clutch's report from replacing another's. Source and code
hashes define run folders; completed runs resume through the common mechanism.
Comparison figures align the time-derived reference segmentation above both
inferred state sequences, with dashed reference boundaries across all strips.

Dedicated [PCA and autoencoder scripts](expression_representations.md) share
this extraction, QC, labeling and result layout, using fixed calibrated decoders.
