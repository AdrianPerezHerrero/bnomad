"""Offline alignment plus a nonidentity GP warp; records its fitted offsets."""
from hdpgpc.examples.reference import run_experiment

if __name__ == '__main__':
    report = run_experiment('warping')
    print(report['scientific_result']['labels'])
    print(report['scientific_result']['alignment_demo'])
