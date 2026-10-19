"""CAS-Colon ordered images with a fixed PCA emission and learned LDS dynamics.

python examples/cas_colon_pca_clustering.py --data-dir agents/data/cas_colon/CAS-ColonDB --records 1
python examples/cas_colon_pca_clustering.py --records 1 --image-size 24 24 --components 8 --max-series 150
python examples/cas_colon_pca_clustering.py --records all --resume

Extract the official Frame.rar and Label.rar first; no videos need decoding.
Use --help for the common inference/prior controls and image-specific options.
Labels are used only for metrics and plots. PCA uses this video's calibration
frames; reported clustering metrics are in-sample, not held-out accuracy.

Outputs include image galleries (observed/PCA/LDS/one-step), latent trajectories,
assignments, metrics, logs, calibration and state arrays in separate run folders.
The first experiment uses grayscale images, offline inference, HMM transitions,
fixed PCA C/Sigma and cluster-learned A/Gamma. See docs/cas_colon_pca.md.
"""
from hdpgpc.experiments.cas_colon import build_parser, make_config, run_cas_colon

DATASET = 'cas_colon'
REPRESENTATION = 'pca'
# Ordinary CLI defaults, not another model configuration system.
DEFAULTS = {}


def main():
    parser = build_parser(__doc__, DEFAULTS)
    args = parser.parse_args()
    reports = run_cas_colon(args, parser)
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
