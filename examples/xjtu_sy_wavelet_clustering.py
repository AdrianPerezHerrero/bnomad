"""Cluster XJTU-SY raw acquisition windows with separate sensor Haar bases.

python examples/xjtu_sy_wavelet_clustering.py --download
python examples/xjtu_sy_wavelet_clustering.py --records Bearing1_1 Bearing1_4 --components 16 --resume
python examples/xjtu_sy_wavelet_clustering.py --components 8 --configurations independent dynamic

Physical-clock is the experiment default. Components are selected once by
unlabeled calibration energy from the existing orthonormal Haar basis. The raw
window length must be a power of two. No PCA preprocessing or per-window
normalization is applied. See docs/xjtu_sy_workflow.md.
"""
from hdpgpc.experiments.xjtu_sy import build_parser, run_xjtu_sy

DEFAULTS = {}


def main():
    reports=run_xjtu_sy(build_parser(__doc__,DEFAULTS,representation='wavelet').parse_args())
    if any(report['status']=='error' for report in reports):
        raise SystemExit(1)


if __name__=='__main__':
    main()
