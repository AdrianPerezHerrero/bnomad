"""Zebrafish expression states with a calibrated, fixed PCA decoder.

python examples/zebrafish_pca_clustering.py --download
python examples/zebrafish_pca_clustering.py --components 10 --projection-components 30 --resume

Retain all genes in dataset preparation. Both configurations use the same
fixed PCA decoder; independent states fix A=0 and learn Gamma per cluster,
while dynamic states learn A/Gamma. HMM assignments remain enabled by default.
Sigma is fixed calibrated reconstruction noise unless --sigma-mode changes it.
"""
from hdpgpc.experiments.zebrafish import build_parser, run_zebrafish

REPRESENTATION = 'pca'
DEFAULTS = {}


def main():
    args = build_parser(__doc__, DEFAULTS, representation=REPRESENTATION).parse_args()
    if any(report['status'] == 'error' for report in run_zebrafish(args)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
