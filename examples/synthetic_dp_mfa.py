"""Fit the independent variational DP-MFA using only public package APIs.

python examples/synthetic_dp_mfa.py --components 5 --iterations 150
Artifacts default to the ignored agents directory; no external data required.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from hdpgpc import HDPMFA, HDPMFAConfig
from hdpgpc.config import HDPConfig, NumericalConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components', type=int, default=5, help='Fixed variational support size K')
    parser.add_argument('--iterations', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('agents/experiments/mfa/dp_example'))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    n, d, rank = 80, 6, 2
    true_means = np.array([[-5., 0., 3., 0., 0., 0.], [5., 0., -3., 0., 0., 0.],
                          [0., 5., 0., -4., 0., 0.]])
    true_w = rng.normal(0, .45, (3, d, rank))
    true_noise = np.linspace(.08, .18, d)
    labels = np.repeat(np.arange(3), n)
    z = rng.normal(size=(3 * n, rank))
    values = (true_means[labels] + np.einsum('ndr,nr->nd', true_w[labels], z)
              + rng.normal(size=(3 * n, d)) * np.sqrt(true_noise))
    config = HDPMFAConfig(n_factors=rank,
        hdp=HDPConfig(M=args.components, hmm_switch=False, hdp_hyp='more'),
        numerics=NumericalConfig(seed=args.seed, num_threads=1),
        loading_prior_precision=.1, noise_initial=.4, max_iter=args.iterations,
        tol=1e-6, verbose='silent')
    model = HDPMFA(config).fit(values)
    coordinates = model.transform(values)
    summary = {'observations': len(values), 'dimension': d, 'factor_rank': rank,
        'represented_components': model.n_components_, 'map_occupied_components': model.n_clusters_,
        'occupancies': model.occupancies_.tolist(), 'weights_conditional_on_support': model.weights_.tolist(),
        'tail_mass': model.tail_mass_, 'iterations': model.n_iter_, 'converged': model.converged_,
        'initial_elbo': float(model.elbo_history_[0]), 'final_elbo': model.elbo_,
        'elbo_components': model.elbo_components_,
        'minimum_sweep_gain': float(np.diff(model.elbo_history_).min()),
        'minimum_coordinate_gain': min(phase['delta_elbo'] for record in model.history_ for phase in record['phases']),
        'ari': adjusted_rand_score(labels, model.labels_),
        'nmi': normalized_mutual_info_score(labels, model.labels_),
        'learned_noise': model.noise_.tolist(), 'true_noise': true_noise.tolist(),
        'mean_plugin_log_density': float(model.score_samples(values).mean()),
        'map_coordinate_shape': list(coordinates.shape)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    model.save(args.output_dir / 'model.json')
    np.savez_compressed(args.output_dir / 'results.npz', values=values, true_labels=labels,
        labels=model.labels_, responsibilities=model.responsibilities_, map_coordinates=coordinates,
        component_coordinates=model.transform_components(values).mean, elbo_history=model.elbo_history_)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
