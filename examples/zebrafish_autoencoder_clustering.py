"""Zebrafish expression states with a trained, fixed affine autoencoder decoder.

python examples/zebrafish_autoencoder_clustering.py --download
python examples/zebrafish_autoencoder_clustering.py --epochs 250 --components 10 --resume

Use the package's tied linear orthonormal autoencoder and exact affine Gaussian
observation model. Train once on the selected time course, then freeze the
decoder for both independent and evolving latent-state configurations.
All genes enter preparation; optional preliminary PCA is controlled by
--projection-components (0 bypasses it). Targets are evaluation-only.
"""
from hdpgpc.experiments.zebrafish import build_parser, run_zebrafish

REPRESENTATION = 'autoencoder'
DEFAULTS = {}


def main():
    args = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION).parse_args()
    if any(report['status'] == 'error' for report in run_zebrafish(args)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
