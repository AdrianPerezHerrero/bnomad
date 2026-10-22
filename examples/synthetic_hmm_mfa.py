"""Boundary-aware HDP-HMM-MFA, with objective-gated births/pruning.

python examples/synthetic_hmm_mfa.py --components 1 --max-components 6
All generated artifacts default to the ignored agents directory.
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
    parser.add_argument('--components', type=int, default=1, help='Initial represented states')
    parser.add_argument('--max-components', type=int, default=6)
    parser.add_argument('--strategy', choices=('fixed', 'birth_prune'), default='birth_prune')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--sequence-count', type=int, default=3)
    parser.add_argument('--sequence-length', type=int, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', choices=('silent', 'progress', 'debug'), default='silent')
    parser.add_argument('--output-dir', type=Path, default=Path('agents/experiments/mfa/hmm_example'))
    args = parser.parse_args()
    if args.sequence_count < 1 or args.sequence_length < 1:
        parser.error('sequence count and length must be positive')
    rng = np.random.default_rng(args.seed)
    lengths = [args.sequence_length] * args.sequence_count
    means = np.array([[-5.,0.,3.,0.,0.,0.], [5.,0.,-3.,0.,0.,0.], [0.,5.,0.,-4.,0.,0.]])
    transition = np.full((3,3), .02) + np.eye(3) * .94
    labels = []
    for length in lengths:
        state = int(rng.integers(3))
        for _ in range(length):
            labels.append(state)
            state = int(rng.choice(3, p=transition[state]))
    labels = np.array(labels)
    loadings = rng.normal(0, .45, (3,6,2))
    z = rng.normal(size=(len(labels),2))
    values = (means[labels] + np.einsum('ndr,nr->nd',loadings[labels],z)
              + rng.normal(size=(len(labels),6)) * .3)
    config = HDPMFAConfig(n_factors=2,
        hdp=HDPConfig(M=args.components, max_models=args.max_components,
                      hmm_switch=True, hdp_hyp='max'),
        numerics=NumericalConfig(seed=args.seed, num_threads=1),
        loading_prior_precision=.1, noise_initial=.4,
        component_strategy=args.strategy, n_explore_steps=3,
        proposal_refine_steps=5, prune_threshold=1.,
        max_iter=args.iterations, verbose=args.verbose)
    model = HDPMFA(config).fit(values, sequence_lengths=lengths)
    coordinates = model.transform(values, sequence_lengths=lengths)
    matched = ((model.means_[:, None, :] - means[None, :, :])**2).sum(2).argmin(1)
    true_covariance = loadings @ loadings.transpose(0,2,1) + .09 * np.eye(6)
    fitted_w = model.loadings_
    fitted_covariance = fitted_w @ fitted_w.transpose(0,2,1) + np.diag(model.noise_)
    covariance_error = [float(np.linalg.norm(fitted_covariance[k]-true_covariance[reference])
        / np.linalg.norm(true_covariance[reference])) for k,reference in enumerate(matched)]
    accepted = [record for record in model.structure_history_ if record['accepted']]
    summary = dict(observations=len(values), sequence_lengths=lengths,
        expected_transition_count=len(values)-len(lengths),
        actual_transition_count=float(model.transition_counts_.sum()),
        actual_start_count=float(model.start_counts_.sum()),
        initial_components=args.components, final_components=model.n_components_,
        map_clusters=model.n_clusters_, occupancies=model.occupancies_.tolist(),
        accepted_moves=accepted, iterations=model.n_iter_, converged=model.converged_,
        initial_elbo=float(model.elbo_history_[0]), final_elbo=model.elbo_,
        elbo_components=model.elbo_components_,
        minimum_sweep_gain=float(np.diff(model.elbo_history_).min()),
        minimum_coordinate_gain=min(phase['delta_elbo'] for record in model.history_ for phase in record['phases']),
        ari=adjusted_rand_score(labels,model.labels_),
        nmi=normalized_mutual_info_score(labels,model.labels_),
        covariance_relative_errors=covariance_error,
        covariance_reference_matching='nearest known centroid; subspace rotations are not identified',
        map_coordinate_shape=list(coordinates.shape),
        mean_plugin_log_density=float(model.score_samples(values,lengths).mean()),
        transition_matrix=model.transition_matrix_.tolist(), tail_mass=model.tail_mass_)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    model.save(args.output_dir/'model.json')
    restored = HDPMFA.load(args.output_dir/'model.json')
    summary['roundtrip_max_assignment_error'] = float(np.max(np.abs(
        model.predict_proba(values,lengths)-restored.predict_proba(values,lengths))))
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    np.savez_compressed(args.output_dir/'results.npz',values=values, true_labels=labels,
        labels=model.labels_,responsibilities=model.responsibilities_,pairs=model.pair_responsibilities_,
        sequence_lengths=lengths,transition=model.transition_matrix_,elbo=model.elbo_history_,
        map_coordinates=coordinates, true_covariance=true_covariance,
        fitted_covariance=fitted_covariance, matched_reference=matched)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3,2,figsize=(13,10),layout='constrained')
    figure.suptitle(f'HDP-HMM-MFA | {len(lengths)} sequences | ARI={summary["ari"]:.3f} | NMI={summary["nmi"]:.3f}')
    axes[0,0].scatter(np.arange(len(labels)),labels,c=labels,s=8,cmap='tab10')
    axes[0,0].set_title('Known regimes')
    axes[0,1].scatter(np.arange(len(labels)),model.labels_,c=model.labels_,s=8,cmap='tab10')
    axes[0,1].set_title('Inferred regimes (IDs need not match truth)')
    for axis in axes[0]:
        axis.set_yticks(np.arange(max(3, model.n_components_)))
        for boundary in np.cumsum(lengths)[:-1]: axis.axvline(boundary-.5,c='k',ls='--',lw=.8)
        axis.set_xlabel('Observation index; dashed lines restart the HMM')
    im=axes[1,0].imshow(model.responsibilities_.T,aspect='auto',vmin=0,vmax=1,interpolation='nearest')
    axes[1,0].set_title('Soft state marginals'); axes[1,0].set_ylabel('Component')
    axes[1,0].set_yticks(np.arange(model.n_components_))
    figure.colorbar(im,ax=axes[1,0],label='Probability')
    im=axes[1,1].imshow(model.transition_matrix_,vmin=0,vmax=1,cmap='Blues')
    axes[1,1].set_title('Mean transitions, conditioned on represented support')
    axes[1,1].set_xlabel('Next state'); axes[1,1].set_ylabel('Current state')
    axes[1,1].set_xticks(np.arange(model.n_components_))
    axes[1,1].set_yticks(np.arange(model.n_components_))
    figure.colorbar(im,ax=axes[1,1],label='Probability')
    axes[2,0].plot(model.elbo_history_,marker='.',ms=3)
    axes[2,0].set_title('Complete variational bound'); axes[2,0].set_xlabel('Sweep')
    for move in accepted:
        axes[2,0].axvline(move['iteration'],ls=':',color='gray')
    axes[2,1].scatter(values[:,0],values[:,1],c=model.labels_,s=9,cmap='tab10',alpha=.7)
    axes[2,1].set_title('First two observed features, colored by inferred state')
    figure.savefig(args.output_dir/'summary.png',dpi=140)
    plt.close(figure)
    figure,axes=plt.subplots(model.n_components_,2,figsize=(9,3.5*model.n_components_),
                            squeeze=False,layout='constrained')
    figure.suptitle('Observation covariances: reference versus fitted plug-in MFA')
    for k,reference in enumerate(matched):
        limit=max(np.abs(true_covariance[reference]).max(),np.abs(fitted_covariance[k]).max())
        for column,matrix in enumerate((true_covariance[reference],fitted_covariance[k])):
            im=axes[k,column].imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit)
            figure.colorbar(im,ax=axes[k,column],fraction=.045)
        axes[k,0].set_title(f'Reference {reference}, matched by centroid')
        axes[k,1].set_title(f'Component {k}, relative error {covariance_error[k]:.3f}')
    figure.savefig(args.output_dir/'covariances.png',dpi=140)
    plt.close(figure)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
