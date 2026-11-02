"""Cluster XJTU-SY run-to-failure bearings using raw windows and a linear AE.

python examples/xjtu_sy_autoencoder_clustering.py --download
python examples/xjtu_sy_autoencoder_clustering.py --configurations independent dynamic --download --resume
python examples/xjtu_sy_autoencoder_clustering.py --records Bearing1_1 Bearing1_4 --download --resume

Each observation is a window inside one acquisition; the LDS uses acquisition
order, preserving the once-per-minute gaps. See docs/xjtu_sy_workflow.md.
"""
from hdpgpc.experiments.xjtu_sy import build_parser, run_xjtu_sy

DEFAULTS = {}


def main():
    reports=run_xjtu_sy(build_parser(__doc__,DEFAULTS).parse_args())
    if any(report['status']=='error' for report in reports):
        raise SystemExit(1)


if __name__=='__main__':
    main()
