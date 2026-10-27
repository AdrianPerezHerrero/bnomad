"""Numerical and lifecycle contracts for pooled MFA emissions."""
from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from sklearn.gaussian_process.kernels import RBF

from hdpgpc import HDPGPC, HDPGPCConfig
from hdpgpc.config import (GPEmissionConfig, DynamicsConfig, AlignmentConfig,
    InferenceConfig, HDPConfig, NumericalConfig, NoisePolicyConfig, with_mfa_emission, with_fixed_observation_model)
from hdpgpc.emissions._fixed_observation import AffineGaussianObservation
from hdpgpc.emissions._factor_analysis import FactorAnalysisPosterior, FactorAnalysisEmission
from hdpgpc.emissions._shared_factor_analysis import SharedFactorAnalysisPolicy
from hdpgpc.inference._scoring import full_LDS_elbo

DTYPE = torch.float64

def fixture():
    C = torch.tensor([[1.], [.3]], dtype=DTYPE)
    Sigma = torch.diag(torch.tensor([.4, .7], dtype=DTYPE))
    baseline = torch.tensor([[.2], [-.1]], dtype=DTYPE)
    Y = torch.tensor([[1., 2.], [2., 1.], [-1., 3.], [0., 4.]], dtype=DTYPE)
    responsibilities = torch.tensor([[.8, .2], [.6, .4], [.1, .9], [.3, .7]], dtype=DTYPE)
    means = ([.5, 1., -.2, .1], [-.4, .3, .9, .7])
    emissions = []
    for values in means:
        emissions.append(SimpleNamespace(
            observation_params=FactorAnalysisPosterior.prior(C, Sigma, 4., 2., 3.),
            factor_analysis=True, indexes=list(range(4)),
            y_train=[v[:, None] for v in Y],
            f_star_sm=[torch.zeros((1, 1), dtype=DTYPE)] + [torch.tensor([[v]], dtype=DTYPE) for v in values],
            cov_f_sm=[torch.eye(1, dtype=DTYPE)*.1 for _ in range(5)],
            direct_observation_grid=torch.arange(2., dtype=DTYPE)[:, None],
            direct_observation_offset=baseline, ini_cov_def=torch.eye(1, dtype=DTYPE),
            gp=SimpleNamespace(assign_alpha_ini=Mock()), Gamma=[torch.eye(1, dtype=DTYPE)],
            A=[torch.eye(1, dtype=DTYPE)], y_var=[]))
    policy = SharedFactorAnalysisPolicy(C, Sigma, 4., 2., 3.)
    for gp in emissions:
        policy.attach(gp)
    return policy, emissions, responsibilities, Y, baseline


class SharedMFAPosteriorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_pooled_posterior_matches_independent_augmented_regression(self):
        policy, gps, r, Y, baseline = fixture()
        policy.update(gps, r)
        # Independently build expected sufficient statistics without the
        # implementation's square-root-weighted design or residual formula.
        K0 = np.diag([2., 2., 3.]); K = K0.copy()
        M0 = np.array([[0., 0., 1.], [0., 0., .3]])
        info = M0 @ K0
        yy = np.zeros(2)
        for k, gp in enumerate(gps):
            for t in range(4):
                u = np.array([float(k == 0), float(k == 1), float(gp.f_star_sm[t+1][0, 0])])
                Euu = np.outer(u, u); Euu[2, 2] += .1
                y = (Y[t]-baseline[:, 0]).numpy(); w = float(r[t, k])
                K += w*Euu; info += w*np.outer(y, u); yy += w*y*y
        mean = np.linalg.solve(K, info.T).T
        scatter = yy + np.diag(M0@K0@M0.T) - np.diag(mean@K@mean.T)
        expected_scale = (4*np.array([.4, .7])+scatter)/8
        np.testing.assert_allclose(policy.posterior.m_mean, mean, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(policy.posterior.m_r_cov, K, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(policy.Sigma.diag(), expected_scale, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(policy.posterior.n0, 8.)  # Four total observations, not eight.
        for k, gp in enumerate(gps):
            self.assertIs(gp.C[-1], policy.C)
            self.assertIs(gp.Sigma[-1], policy.Sigma)
            np.testing.assert_allclose(gp.fixed_observation_model.offset, baseline[:, 0]+mean[:, k])
            slots = [k, 2]
            np.testing.assert_allclose(gp.fixed_observation_model.column_covariance,
                np.linalg.inv(K)[slots][:, slots], rtol=1e-12, atol=1e-12)
            snapshot = FactorAnalysisEmission.from_state_dict(gp.fixed_observation_model.state_dict())
            np.testing.assert_array_equal(snapshot.C, policy.C)
        before = policy.posterior.m_mean.clone()
        policy.update(gps, r)
        torch.testing.assert_close(before, policy.posterior.m_mean)  # No double-counting offline.
        for gp in gps:
            gp.return_LDS_param_likelihood = lambda: torch.tensor(0., dtype=DTYPE)
        value = full_LDS_elbo(None, gps, r.sum(0))
        torch.testing.assert_close(value, -policy.kl())

    def test_reordering_empty_cluster_and_proposal_copy_preserve_pooled_fit(self):
        policy, gps, r, _, _ = fixture(); policy.update(gps, r)
        C, S, kl = policy.C.clone(), policy.Sigma.clone(), policy.kl().clone()
        offsets = [g.fixed_observation_model.offset.copy() for g in gps]
        candidate = deepcopy(policy)
        reordered = deepcopy(gps[::-1])
        candidate.update(reordered, r.flip(1))
        torch.testing.assert_close(C, candidate.C)
        torch.testing.assert_close(S, candidate.Sigma)
        for k, gp in enumerate(reordered):
            np.testing.assert_allclose(gp.fixed_observation_model.offset, offsets[1-k])
        _, more, _, _, _ = fixture(); empty = more[0]; empty.indexes=[]
        policy.update(gps+[empty], torch.cat((r, r.new_zeros((4, 1))), 1))
        torch.testing.assert_close(C, policy.C)
        torch.testing.assert_close(S, policy.Sigma)
        torch.testing.assert_close(kl, policy.kl())  # An unused intercept retains its prior.
        self.assertEqual(policy.intercepts, 3)
        self.assertIs(empty.observation_policy, policy)
        with self.assertRaisesRegex(ValueError, 'offline'):
            policy.update(gps, r, online=True)


def recipe(shared=True, dynamic=True):
    data = np.concatenate((np.tile([-2., -1., -.5, -1.5], (6, 1)),
                           np.tile([2., 1., .5, 1.5], (6, 1))))
    data = data + np.random.default_rng(5).normal(0, .1, data.shape)
    grid = np.arange(4.)[:, None]
    initial = AffineGaussianObservation(np.array([[1.], [.5], [.2], [.7]]),
        np.eye(4)*.1, grid, np.zeros(4), np.eye(1))
    config = HDPGPCConfig(x_basis=np.zeros((1, 1)),
        emission=GPEmissionConfig(kernels=[RBF(1.)], fit_gp=False, ini_sigma=.1),
        dynamics=DynamicsConfig(ini_gamma=.1, lds_update='conjugate_batch', free_deg_MNIV=4),
        alignment=AlignmentConfig(policy='none'),
        hdp=HDPConfig(hmm_switch=True, transition_topology='left_to_right', max_models=3),
        inference=InferenceConfig(schedule='offline', offline_strategy='split_merge',
            it_limit=4, n_explore_steps=2, hard_assignments=False),
        numerics=NumericalConfig(seed=3, num_threads=1))
    return with_mfa_emission(config, initial, dynamic=dynamic, shared=shared), data


class SharedMFAIntegrationTests(unittest.TestCase):
    def test_fit_pickle_conversion_and_shared_snapshots(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                config, data = recipe(dynamic=dynamic)
                config = HDPGPCConfig.from_yaml(config.to_yaml())
                model = HDPGPC(config).fit(data)
                self.assertGreaterEqual(model.n_clusters_, 2 if dynamic else 1)
                backend = model._backend; owner = backend.observation_policies[0]
                self.assertEqual(model.observation_parameters_['family'], 'factor_analysis')
                self.assertEqual(model.observation_parameters_['scope'], 'shared')
                offsets = []
                for k in model.cluster_ids_:
                    gp = backend.gpmodels[0][int(k)]
                    self.assertIs(gp.observation_policy, owner)
                    torch.testing.assert_close(gp.C[-1], owner.C)
                    torch.testing.assert_close(gp.Sigma[-1], owner.Sigma)
                    np.testing.assert_array_equal(gp.fixed_observation_model.C, owner.C)
                    offsets.append(gp.fixed_observation_model.offset)
                    self.assertTrue(np.isfinite(model.predictive(int(k)).mean).all())
                if dynamic:
                    self.assertGreater(np.linalg.norm(offsets[0]-offsets[-1]), .1)
                reloaded = pickle.loads(pickle.dumps(model))
                np.testing.assert_array_equal(reloaded.labels_, model.labels_)
                reloaded._backend.full_model_to_numpy()
                reloaded._backend.full_model_to_torch()
                for k in model.cluster_ids_:
                    gp = reloaded._backend.gpmodels[0][int(k)]
                    self.assertIs(gp.observation_policy, reloaded._backend.observation_policies[0])
                    self.assertIsInstance(gp.observation_params, FactorAnalysisPosterior)
                    np.testing.assert_allclose(reloaded.predictive(int(k)).mean, model.predictive(int(k)).mean)

    def test_cli_shared_modes_and_existing_shared_affine_noise(self):
        from hdpgpc.experiments.cas_colon import build_parser, _validate, make_config
        initial = AffineGaussianObservation(np.ones((4, 1)), np.eye(4)*.1,
            np.arange(4.)[:, None], np.zeros(4), np.eye(1))
        for flags in (['--shared-emission'], ['--c-mode', 'shared', '--sigma-mode', 'shared_diagonal']):
            args = build_parser(representation='mfa').parse_args(flags)
            _validate(args)
            config = make_config(initial.grid, args, initial)
            self.assertEqual(config.observation.C_policy.mode, 'shared')
            self.assertEqual(config.observation.Sigma_policy.mode, 'shared_diagonal')
            self.assertEqual(config.dynamics.lds_update, 'conjugate_batch')
        args = build_parser(representation='mfa').parse_args(
            ['--shared-emission', '--lds-update', 'natural_gradient'])
        with self.assertRaisesRegex(ValueError, 'conjugate_batch'):
            _validate(args)
        config, data = recipe()
        linear = with_fixed_observation_model(config, initial,
            sigma_policy=NoisePolicyConfig(mode='shared_diagonal'))
        linear = replace(linear, inference=replace(linear.inference, it_limit=2))
        model = HDPGPC(linear).fit(data)
        self.assertEqual(model.observation_parameters_['scope'], 'shared')
        self.assertFalse(model._backend.factor_analysis)
        for gp in model._backend.gpmodels[0]:
            np.testing.assert_array_equal(gp.C[-1], initial.C)

    def test_old_local_mfa_and_unsupported_shared_combinations(self):
        config, data = recipe(shared=False)
        model = HDPGPC(config).fit(data)
        self.assertEqual(model.observation_parameters_['scope'], 'cluster')
        shared, _ = recipe()
        for bad in (lambda: replace(shared, dynamics=replace(shared.dynamics, lds_update='conjugate_sequential')),
                    lambda: replace(shared, inference=replace(shared.inference, schedule='online')),
                    lambda: replace(shared, observation=replace(shared.observation,
                        Sigma_policy=NoisePolicyConfig(mode='cluster_diagonal', value=np.eye(4))))):
            with self.assertRaisesRegex(ValueError, 'Shared MFA|Factor analysis'):
                HDPGPC(bad())


if __name__ == '__main__':
    unittest.main()
