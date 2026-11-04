# Bibliography and data sources

The repository is bnomad; its Python distribution/import is `hdpgpc` (0.3.0).
Use [CITATION.cff](../CITATION.cff) to cite the software. It declares no software
paper DOI or arXiv identifier. The references below describe methods and data;
they are not substitute publication identifiers for this package. Dataset
papers do not supply sample-level targets unless the workflow explicitly says so.

## Methods

- Teh, Jordan, Beal and Blei (2006), *Hierarchical Dirichlet Processes*.
  [Journal DOI](https://doi.org/10.1198/016214506000000302).
- Fox, Sudderth, Jordan and Willsky (2011), *A sticky HDP-HMM with application
  to speaker diarization*. [Journal DOI](https://doi.org/10.1214/10-AOAS395).
- Ustyuzhaninov et al. (2020), *Monotonic Gaussian Process Flows*.
  [AISTATS/PMLR publication](https://proceedings.mlr.press/v108/ustyuzhaninov20a.html).
- Mikheeva et al. (2022), *Aligned Multi-Task Gaussian Process*.
  [AISTATS/PMLR publication](https://proceedings.mlr.press/v151/mikheeva22a.html).
- Wiskott and Sejnowski (2002), *Slow Feature Analysis: Unsupervised Learning
  of Invariances*. [Journal DOI](https://doi.org/10.1162/089976602317318938).

These references motivate parts of the implementation; consult the corresponding
[alignment](alignment_choices.md), [HDP](hdp_presets.md) and
[SFA](linear_sfa_observation.md) guides for the implemented assumptions.
The attributed bnpy optimizer is described in [NOTICE.md](../NOTICE.md).

## Dataset publications

- White et al. (2017), *A high-resolution mRNA expression time course of embryonic
  development in zebrafish*. [eLife](https://doi.org/10.7554/eLife.30860).
  The [workflow](zebrafish_workflow.md) explains temporal reference groups and
  retrospective calibration.
- Owens et al. (2016), *Measuring Absolute RNA Copy Numbers at High Temporal
  Resolution Reveals Transcriptome Kinetics in Development*.
  [Cell Reports](https://doi.org/10.1016/j.celrep.2015.12.050).
  The DOI contains 2015; the publication year is 2016. See the
  [Xenopus workflow](xenopus_workflow.md) for clutch and stage definitions.
- Santos et al. (2024), *Wind tunnel bench test of a pitch-and-plunge aeroelastic
  model undergoing nonlinear post-flutter oscillations*.
  [Journal article and supplement](https://doi.org/10.1007/s44245-024-00043-w).
  [Flutter record contexts](flutter_workflow.md) are record-level descriptions,
  not measured instability-onset or breakage labels.
- Wang, Lei, Li and Li (2020), *A Hybrid Prognostics Approach for Estimating
  Remaining Useful Life of Rolling Element Bearings*.
  [IEEE Transactions on Reliability](https://doi.org/10.1109/TR.2018.2882682),
  [creator's XJTU-SY data page](https://biaowang.tech/xjtu-sy-bearing-datasets/).
  The DOI contains 2018; the volume publication year is 2020. See
  [XJTU-SY](xjtu_sy_workflow.md) for acquisition timing and terminal inspections.
- Song et al. (2025), *CAS-Colon: A Comprehensive Colonoscopy Anatomical
  Segmentation Dataset for Artificial Intelligence Development*.
  [Scientific Data](https://doi.org/10.1038/s41597-025-05588-3).
  See the [CAS-Colon workflow](cas_colon_pca.md) for the data-layout contract.

Other dataset access and licensing references are maintained in the
[MIT-BIH](mitbih_annotations.md), [Daphnet](daphnet_workflow.md) and
[UCR](ucr_autoencoder_archive.md) guides. Access restrictions or a publisher's
automated-request rejection do not establish that a DOI is invalid.
