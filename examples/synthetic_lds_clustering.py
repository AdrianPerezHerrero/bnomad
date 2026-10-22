"""Cluster synthetic LDS images using the existing HDP-GPC model.

python examples/synthetic_lds_clustering.py
python examples/synthetic_lds_clustering.py --input agents/synthetic_lds_images/per_regime_seed_42_6b1c23194133/dataset.npz
python examples/synthetic_lds_clustering.py --representation pca
python examples/synthetic_lds_clustering.py --scenario dynamics_only --representation pca --components 4
python examples/synthetic_lds_clustering.py --a-mode fixed --no-hmm-switch
python examples/synthetic_lds_clustering.py --shuffle

Default: generate 720 overlapping physical wave/eddy images, six latent
coordinates, per-regime histories; use its known fixed C/Sigma,
and learn cluster A/Gamma with offline split/merge inference. Truth supplies the
oracle decoder and default noise scale; labels never enter model inference.

Logs, config, assignments, purity/ARI/NMI, image galleries, latent trajectories,
LDS plots and comparisons with known states/matrices go into separate run
folders under agents/synthetic_lds_clustering_runs. Saved figures are not shown.
See docs/synthetic_lds_images.md. This script does not run during generation.
"""
from hdpgpc.experiments.synthetic_image_clustering import main


if __name__ == '__main__':
    main()
