"""Cluster XJTU-SY bearing lifetimes using raw acquisition windows and fixed PCA.

python examples/xjtu_sy_pca_clustering.py --download
python examples/xjtu_sy_pca_clustering.py --records Bearing1_1 Bearing1_2 Bearing1_3 Bearing1_4 Bearing1_5 --components 10 --configurations independent dynamic --download --resume

Uses the same preparation, channel scaling, inference and artifacts as the
XJTU-SY autoencoder variant. See docs/xjtu_sy_workflow.md.
"""
from hdpgpc.experiments.xjtu_sy import build_parser, run_xjtu_sy

DEFAULTS = {}


def main():
    reports=run_xjtu_sy(build_parser(__doc__,DEFAULTS,representation='pca').parse_args())
    if any(report['status']=='error' for report in reports):
        raise SystemExit(1)


if __name__=='__main__':
    main()
