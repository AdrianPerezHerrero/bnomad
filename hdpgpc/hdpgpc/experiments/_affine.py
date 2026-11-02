"""Shared native configuration for calibrated affine observation experiments."""
from dataclasses import replace
import numpy as np
from ..config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
    AlignmentConfig, InferenceConfig, NumericalConfig, MatrixPolicyConfig,
    NoisePolicyConfig, with_mfa_emission, with_fixed_observation_model)
from ._arguments import lds_update


def make_config(args, initialization, *, dynamic):
    """Shared configuration for learned MFA or fixed PCA/autoencoder emissions."""
    from sklearn.gaussian_process.kernels import RBF
    initialization = replace(initialization, Sigma=initialization.Sigma * args.sigma_prior)
    config = HDPGPCConfig(
        x_basis=np.arange(initialization.latent_dim, dtype=float)[:, None],
        verbose=args.verbose,
        emission=GPEmissionConfig(kernels=[RBF(1.)], fit_gp=False, ini_sigma=args.sigma_prior),
        dynamics=DynamicsConfig(ini_gamma=args.gamma_prior, annealing=False,
            bayesian_params=True, free_deg_MNIV=args.lds_degrees,
            lds_update=lds_update(args), lds_scoring='final', lds_step_size=args.lds_step_size),
        alignment=AlignmentConfig(policy='none'),
        hdp=HDPConfig(hdp_hyp=args.hdp_preset, hmm_switch=args.hmm_switch,
            max_models=args.max_clusters, transition_topology=args.transition_topology),
        inference=InferenceConfig(schedule='offline', warp=False, it_limit=args.iterations,
            offline_strategy=args.offline_strategy, n_explore_steps=args.exploration_steps,
            offline_representer_selection=args.offline_representer_selection,
            split_merge_refine_steps=args.split_merge_refine_steps,
            split_merge_stability_patience=args.split_merge_stability_patience,
            split_merge_score_rtol=args.split_merge_score_rtol,
            hard_assignments=args.hard_assignments,
            birth_lds_kl_per_dimension=args.birth_lds_kl_per_dimension),
        numerics=NumericalConfig(seed=args.seed, num_threads=1, dtype='float64', deterministic=True))
    if args.representation == 'mfa':
        config = with_mfa_emission(config, initialization, dynamic=dynamic,
            shared=args.shared_emission, mean_precision=args.mean_precision,
            loading_precision=args.loading_precision)
    else:
        config = with_fixed_observation_model(config, initialization,
            sigma_policy=NoisePolicyConfig(mode=args.sigma_mode))
        if not dynamic:
            config = replace(config, dynamics=replace(config.dynamics,
                A_policy=MatrixPolicyConfig(mode='fixed', value=np.zeros((initialization.latent_dim,)*2))))
    # The MFA convenience helper defaults static factors to IID assignments.
    # This experiment deliberately keeps the requested assignment process in
    # both arms: A=0/Gamma=I removes factor evolution, not HMM dependence.
    return replace(config, hdp=replace(config.hdp, hmm_switch=args.hmm_switch))

