"""CAS-Colon ordered images with a trained fixed affine autoencoder emission.

python examples/cas_colon_autoencoder_clustering.py --records 1
python examples/cas_colon_autoencoder_clustering.py --records 9 --image-size 24 24 --components 8 --max-series 150
python examples/cas_colon_autoencoder_clustering.py --records all --resume

Uses the same data, chronology, ambiguous-label handling and trajectory reports
as cas_colon_pca_clustering.py. Train a tied linear autoencoder on up to 400
frames per segment, then freeze its encoder/decoder. The existing HDP-GPC learns
cluster-specific A/Gamma using the exact affine Gaussian observation model.
Sigma defaults to fixed diagonal reconstruction noise; --sigma-mode can change
its supported policy. --sigma-prior multiplies calibrated Sigma; --gamma-prior
sets the process covariance center times identity. No nonlinear decoder or
approximate LDS inference is introduced. See docs/cas_colon_autoencoder.md.
"""
from hdpgpc.experiments.cas_colon import build_parser, make_config, run_cas_colon

DATASET = 'cas_colon'
REPRESENTATION = 'autoencoder'
# Ordinary CLI defaults, not another model configuration system.
DEFAULTS = {}


def main():
    parser = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION)
    args = parser.parse_args()
    reports = run_cas_colon(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
