"""Reproduce a small CPU reference experiment from an installed package.

python -m hdpgpc.examples.reference --mode offline --check --output result.json
"""
import argparse
from contextlib import redirect_stdout
import hashlib
from importlib import metadata
from importlib.resources import files
import io
import json
import os
from pathlib import Path
import platform


def run_experiment(mode='offline'):
    """Generate data, configure, infer and snapshot public scientific results."""
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    from threadpoolctl import threadpool_limits
    from hdpgpc import HDPGPC
    from . import configuration, waveforms
    config = configuration(mode)
    X = waveforms(config.multi_output.n_outputs)
    model = HDPGPC(config)
    assignments = []
    # Legacy progress printing stays unchanged; the experiment emits JSON only.
    with threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
        if mode == 'online':
            X = X[:4]
            for x in X:
                result = model.update(x)
                assignments.append({'cluster': result.assigned_cluster, 'new_cluster': result.new_cluster})
        else:
            model.fit(X)
        clusters = []
        for cluster in model.cluster_ids_:
            prediction = model.predictive(int(cluster))
            posterior = model.posterior(int(cluster))
            clusters.append({'id': int(cluster), 'mean': prediction.mean.tolist(),
                             'covariance': prediction.covariance.tolist(),
                             'latent': [{'output': state.output,
                                         'filtered_mean': state.filtered_mean.tolist(),
                                         'smoothed_mean': state.smoothed_mean.tolist()} for state in posterior]})
    scientific = {'labels': model.labels_.tolist(), 'n_clusters': model.n_clusters_,
                  'clusters': clusters, 'online_updates': assignments}
    if mode == 'warping':
        # Also demonstrate a visibly nonidentity warp directly, independent of
        # which offline iterations decide to activate alignment for five-point data.
        import torch
        from hdpgpc.alignment import GaussianProcessWarp
        grid = torch.linspace(0., 4., 24)[:, None]
        template = torch.exp(-((grid-1.8)/.65)**2)
        target = torch.exp(-((grid-2.1)/.65)**2)
        warper = GaussianProcessWarp(grid, recursive=False)
        offsets, aligned, score, losses = warper.compute_warp(grid, target, template, train_iter=50)
        scientific['alignment_demo'] = {'offsets': offsets.tolist(), 'aligned': aligned.tolist(),
                                        'score': float(score), 'steps': len(losses['loss'])}
    versions = {name: metadata.version(name) for name in
                ('hdpgpc', 'numpy', 'scipy', 'torch', 'gpytorch', 'linear-operator', 'pyro-ppl', 'scikit-learn')}
    return {'mode': mode, 'config': config.to_dict(), 'scientific_result': scientific,
            'data_sha256': hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
            'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                            'versions': versions, 'seed': 42, 'device': 'cpu', 'dtype': 'float64', 'threads': 1}}


def check_reference(report):
    """Exact discrete values; original characterization tolerance for floats."""
    import numpy as np
    if report['mode'] != 'offline':
        raise ValueError('--check uses the offline reference experiment')
    expected = json.loads(files(__package__).joinpath('reference_offline.json').read_text())
    actual = report['scientific_result']
    if actual['labels'] != expected['labels'] or actual['n_clusters'] != expected['n_clusters']:
        raise AssertionError('Reference cluster assignments changed')
    if actual['online_updates'] != expected['online_updates'] or len(actual['clusters']) != len(expected['clusters']):
        raise AssertionError('Reference result structure changed')
    for a, e in zip(actual['clusters'], expected['clusters']):
        assert a['id'] == e['id']
        for key in ('mean', 'covariance'):
            np.testing.assert_allclose(a[key], e[key], rtol=1e-7, atol=1e-9)
        assert len(a['latent']) == len(e['latent'])
        for la, le in zip(a['latent'], e['latent']):
            assert la['output'] == le['output']
            for key in ('filtered_mean', 'smoothed_mean'):
                np.testing.assert_allclose(la[key], le[key], rtol=1e-7, atol=1e-9)


def main(argv=None):
    from . import MODES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, default='offline')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if args.output and args.output.exists():
        parser.error('Choose a new output path; existing experiment results are not overwritten')
    report = run_experiment(args.mode)
    if args.check:
        check_reference(report)
        report['reference_check'] = 'passed'
    text = json.dumps(report, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
        print(f'{args.mode}: labels={report["scientific_result"]["labels"]}; saved {args.output}')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
