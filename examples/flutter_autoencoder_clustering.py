"""Learn flutter motion regimes from raw delay windows with a fixed linear AE.

python examples/flutter_autoencoder_clustering.py --download
python examples/flutter_autoencoder_clustering.py --records 0950 --download --resume
python examples/flutter_autoencoder_clustering.py --configurations independent dynamic --resume

All recordings share an unlabeled calibration; each is clustered separately.
Preserve amplitudes with one fixed channel scaling, no per-window normalization
and no preliminary PCA or engineered velocities. See docs/flutter_workflow.md.
"""
from hdpgpc.experiments.flutter import build_parser, run_flutter

DEFAULTS = {}


def main():
    reports = run_flutter(build_parser(__doc__, DEFAULTS).parse_args())
    if any(report['status'] == 'error' for report in reports):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
