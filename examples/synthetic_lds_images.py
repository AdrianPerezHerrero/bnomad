"""Generate labeled images with known switching linear Gaussian dynamics.

python examples/synthetic_lds_images.py
python examples/synthetic_lds_images.py --state-mode per_regime
python examples/synthetic_lds_images.py --scenario appearance_dynamics
python examples/synthetic_lds_images.py --scenario dynamics_only --state-mode switching
python examples/synthetic_lds_images.py --segment-length 60 --seed 7 --no-gif

Default: overlapping physical wave/eddy fields with weak appearance differences,
six latent coordinates, a shared affine decoder and resumed per-regime histories.
Distinct tissue textures remain available with --scenario appearance_dynamics.
The previous four-state equal-appearance control is --scenario dynamics_only.
No model is fitted. Ground-truth data, matrices, labels, frame PNGs, a GIF and
diagnostic figures are saved under agents/synthetic_lds_images in distinct runs.
See docs/synthetic_lds_images.md for equations and fixed-affine integration.
"""
from hdpgpc.experiments.synthetic_images import main


if __name__ == '__main__':
    main()
