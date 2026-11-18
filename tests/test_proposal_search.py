"""Noise-coordinate invariance and isolation of offline proposal screening."""
from copy import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.gaussian_process.kernels import RBF

from hdpgpc import HDPGPC
from hdpgpc.config import (HDPGPCConfig, HDPConfig, GPEmissionConfig, DynamicsConfig,
    AlignmentConfig, InferenceConfig, NumericalConfig, NoisePolicyConfig,
    with_fixed_observation_model)
from hdpgpc.emissions import AffineGaussianObservation
from hdpgpc.inference import _proposal_search as search, _split_merge


def observations(points=32, dimension=8):
    rng = np.random.default_rng(17)
    labels = np.repeat([0, 1, 2, 0, 1, 2], 20)
    means = np.array([[-4., -2.], [0., 4.], [4., -2.]])
    values = rng.normal(0, np.sqrt(.1), (120, 128))[:, :points].copy()
    values[:, :2] += means[labels]
    decoder = AffineGaussianObservation(np.eye(points, dimension), .1 * np.eye(points),
        np.arange(points, dtype=float)[:, None], np.zeros(points), np.eye(dimension))
    return values, decoder


def configuration(decoder, strategy='split_merge', clock='cluster-clock', hard=True):
    config = HDPGPCConfig(x_basis=np.arange(decoder.latent_dim, dtype=float)[:, None],
        verbose=False, hdp=HDPConfig(hdp_hyp='balanced', max_models=6),
        emission=GPEmissionConfig(kernels=[RBF()], fit_gp=False, ini_sigma=.1),
        dynamics=DynamicsConfig(ini_gamma=.1, annealing=False, constant_input=True,
            clock=clock, lds_update='conjugate_batch', free_deg_MNIV=5),
        alignment=AlignmentConfig(policy='none'),
        inference=InferenceConfig(schedule='offline', offline_strategy=strategy,
            offline_candidate_search='noise_aware',
            it_limit=1, n_explore_steps=0, hard_assignments=hard, split_merge_refine_steps=1),
        numerics=NumericalConfig(seed=42, num_threads=1, dtype='float64'))
    return with_fixed_observation_model(config, decoder,
                                       sigma_policy=NoisePolicyConfig(mode='fixed'))


def signatures(state):
    features = search.initialize(state, state._split_merge_y).features
    return [(description['kind'], description.get('seed'), resp.argmax(1).tolist())
            for description, resp, _pair, _sources in
            _split_merge._moves(state, features, features)]


class ProposalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        values, decoder = observations()
        cls.reference = HDPGPC(configuration(decoder)).fit(values, grid=decoder.grid)._backend

    def clone(self, points=32):
        values, decoder = observations(points)
        state = copy(self.reference)
        state.__dict__ = self.reference.__dict__.copy()
        state.gpmodels = [[copy(gp) for gp in output] for output in self.reference.gpmodels]
        for output in state.gpmodels:
            for gp in output:
                gp.fixed_observation_model = decoder
        state.fixed_observation_model = decoder
        state._split_merge_y = torch.tensor(values[..., None], dtype=torch.float64)
        state.n_explore_steps = 4
        return state

    def test_projected_scores_equal_full_likelihood_without_common_nuisance(self):
        state = self.clone()
        geometry = search.initialize(state, state._split_merge_y)
        projected = search.observation_scores(state, state.M)[:, 0, 0]
        self.assertEqual(geometry.metadata['outputs'][0]['retained_rank'], 2)
        reference = geometry.outputs[0]
        gp = state.gpmodels[0][0]
        rows = torch.arange(state.T)
        positions = torch.searchsorted(torch.tensor(gp.indexes), rows, right=True)
        means = torch.stack(gp.f_star_sm)[positions].reshape(state.T, -1)
        covariances = torch.stack(gp.cov_f_sm)[positions]
        y = state._split_merge_y[:, :, 0]
        full = gp.compute_sq_err_all(state.x_train, state._split_merge_y, no_first=True)
        # Use the full information subspace for the exact density identity;
        # the production spectral proxy additionally drops weak latent directions.
        C = torch.tensor(state.fixed_observation_model.C.copy())
        inverse_C = C / .1
        H = C.T @ inverse_C
        root = torch.linalg.cholesky(H).T
        residual = y @ inverse_C @ torch.linalg.inv(root) - means @ root.T
        canonical = -.5 * (residual.square().sum(1)
            + torch.einsum('ij,nji->n', H, covariances) + len(H) * np.log(2*np.pi))
        energy = ((y @ inverse_C) @ torch.linalg.inv(H) * (y @ inverse_C)).sum(1)
        nuisance = -.5 * (y.square().sum(1)/.1 - energy
            + y.shape[1]*np.log(.1) + (y.shape[1]-len(H))*np.log(2*np.pi))
        torch.testing.assert_close(full-canonical, nuisance, rtol=1e-10, atol=1e-10)
        expected = -.5 * ((reference.observation-means @ reference.transform).square().sum(1)
            + torch.einsum('ij,nji->n', reference.precision, covariances)
            + reference.transform.shape[1]*np.log(2*np.pi))
        torch.testing.assert_close(projected, expected)

    def test_both_candidate_paths_ignore_added_observation_noise(self):
        original = None
        for points in (16, 32, 64, 128):
            state = self.clone(points)
            moves = signatures(state)
            seeds = search.birth_seeds(state, state.resp_last, state.q_lat_last,
                                       state.snr_last, state.f_ind_old)
            scores = search.observation_scores(state, state.M)
            if original is None:
                original = moves, seeds, scores
            else:
                self.assertEqual(moves, original[0])
                self.assertEqual(seeds, original[1])
                torch.testing.assert_close(scores, original[2], atol=1e-10, rtol=1e-10)

    def test_rowwise_common_density_does_not_change_proposals(self):
        state = self.clone()
        before = signatures(state)
        seeds = search.birth_seeds(state, state.resp_last, state.q_lat_last,
                                   state.snr_last, state.f_ind_old)
        state.q_last = state.q_last + torch.linspace(-1e6, 1e6, state.T)[:, None, None]
        self.assertEqual(before, signatures(state))
        self.assertEqual(seeds, search.birth_seeds(state, state.resp_last, state.q_lat_last,
                                                  state.snr_last, state.f_ind_old))

    def test_existing_cluster_merge_and_split_candidates_ignore_nuisance(self):
        from hdpgpc.inference._assignments import from_labels
        original = None
        for points in (16, 64, 128):
            state = self.clone(points)
            state.M = 3
            state.gpmodels = [[copy(output[0]) for _ in range(3)] for output in state.gpmodels]
            state.resp_last, state.respPair_last = from_labels(
                torch.arange(state.T) // 40, state.M, state.q_last)
            for name in ('q_last', 'q_lat_last', 'snr_last'):
                setattr(state, name, getattr(state, name).expand(-1, 3, -1))
            actual = signatures(state)
            self.assertTrue(any(item[0] == 'merge' for item in actual))
            self.assertTrue(any(item[0] == 'split' for item in actual))
            if original is None:
                original = actual
            else:
                self.assertEqual(original, actual)

    def test_chunked_covariance_projection_matches_full_stack(self):
        precision = torch.eye(400, dtype=torch.float64)
        covariance = torch.diag(torch.linspace(.1, 2., 400))
        covariances = [covariance * (1 + index / 20) for index in range(11)]
        expected = torch.einsum('ij,nji->n', precision, torch.stack(covariances))
        torch.testing.assert_close(search._covariance_traces(covariances, precision), expected)

    def test_search_does_not_mutate_likelihoods_parameters_or_acceptance(self):
        state = self.clone()
        tensors = [state.q_last.clone(), state.q_lat_last.clone(), state._split_merge_y.clone(),
                   state.gpmodels[0][0].A[-1].clone(), state.gpmodels[0][0].Sigma[-1].clone()]
        _split_merge._score(state)
        before = float(state._split_merge_score), float(state.elbo_last)
        signatures(state)
        search.birth_seeds(state, state.resp_last, state.q_lat_last, state.snr_last, state.f_ind_old)
        _split_merge._score(state)
        self.assertEqual(before, (float(state._split_merge_score), float(state.elbo_last)))
        for before, after in zip(tensors, [state.q_last, state.q_lat_last, state._split_merge_y,
                                state.gpmodels[0][0].A[-1], state.gpmodels[0][0].Sigma[-1]]):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
        self.assertNotIn('weight_mean', state.__dict__)

    def test_correlated_orthogonal_nuisance_is_ignored(self):
        state = self.clone()
        search.initialize(state, state._split_merge_y)
        baseline = search.observation_scores(state, state.M)
        model = state.fixed_observation_model
        rng = np.random.default_rng(9)
        basis = rng.normal(size=(7, 7))
        noise = basis @ basis.T + np.eye(7)
        R = np.zeros((39, 39)); R[:32, :32] = model.Sigma; R[32:, 32:] = noise
        decoder = AffineGaussianObservation(np.vstack((model.C, np.zeros((7, 8)))), R,
            np.arange(39.)[:, None], np.zeros(39), np.eye(8))
        state.gpmodels[0][0].fixed_observation_model = decoder
        extra = torch.tensor(rng.multivariate_normal(np.zeros(7), noise, size=state.T))
        state._split_merge_y = torch.cat((state._split_merge_y, extra[..., None]), dim=1)
        search.initialize(state, state._split_merge_y)
        torch.testing.assert_close(search.observation_scores(state, state.M), baseline,
                                   rtol=1e-10, atol=1e-10)

    def test_multi_output_scores_and_degenerate_fallback_are_finite(self):
        state = self.clone()
        state.gpmodels = [state.gpmodels[0], state.gpmodels[0]]
        state.n_outputs = 2
        state._split_merge_y = state._split_merge_y.expand(-1, -1, 2)
        geometry = search.initialize(state, state._split_merge_y)
        scores = search.observation_scores(state, state.M)
        self.assertEqual(geometry.features.shape[1], 4)
        torch.testing.assert_close(scores[:, :, 0], scores[:, :, 1])
        state._split_merge_y = torch.zeros_like(state._split_merge_y)
        geometry = search.initialize(state, state._split_merge_y)
        self.assertTrue(all(item['fallback'] for item in geometry.metadata['outputs']))
        self.assertTrue(torch.isfinite(geometry.features).all())

    def test_rank_deficient_decoder_and_no_observable_direction(self):
        state = self.clone()
        original = state.fixed_observation_model
        C = original.C.copy(); C[:, 2:] = 0
        state.gpmodels[0][0].fixed_observation_model = replace(original, C=C)
        geometry = search.initialize(state, state._split_merge_y)
        self.assertEqual(geometry.metadata['outputs'][0]['observable_rank'], 2)
        self.assertTrue(torch.isfinite(search.observation_scores(state, state.M)).all())
        state.gpmodels[0][0].fixed_observation_model = replace(original, C=np.zeros_like(C))
        self.assertIsNone(search.initialize(state, state._split_merge_y))

    def test_birth_budget_and_representative_exclusions(self):
        state = self.clone()
        search.initialize(state, state._split_merge_y)
        for budget in (0, 1, 4, 200):
            state.n_explore_steps = budget
            seeds = search.birth_seeds(state, state.resp_last, state.q_lat_last,
                                       state.snr_last, state.f_ind_old)
            self.assertLessEqual(len(seeds), budget)
            self.assertEqual(len(set(seeds)), len(seeds))
            self.assertNotIn(int(state.f_ind_old[0]), seeds)

    def test_shared_partition_preserves_soft_adjacent_marginals(self):
        resp = torch.tensor([[.2, .8], [.7, .3], [.4, .6], [.9, .1]], dtype=torch.float64)
        pair = resp.new_zeros((4, 2, 2))
        pair[1:] = resp[:-1, :, None] * resp[1:, None, :]
        proposed, proposed_pair = search.split_partition(
            resp, pair, 0, resp.new_tensor([.1, .8, 1., 0.]))
        torch.testing.assert_close(proposed.sum(1), resp.sum(1))
        torch.testing.assert_close(proposed_pair[1:].sum(1), proposed[1:])
        torch.testing.assert_close(proposed_pair[1:].sum(2), proposed[:-1])
        self.assertEqual(float(proposed_pair[0].sum()), 0.)
        self.assertTrue((proposed_pair >= 0).all())

    def test_all_affine_noise_policies_keep_native_scoring_finite(self):
        values, decoder = observations(points=8, dimension=2)
        for strategy in ('legacy', 'split_merge'):
            for mode in ('fixed', 'cluster', 'cluster_diagonal', 'shared', 'shared_diagonal'):
                with self.subTest(strategy=strategy, sigma=mode):
                    cfg = configuration(decoder, strategy)
                    cfg = with_fixed_observation_model(cfg, decoder,
                        sigma_policy=NoisePolicyConfig(mode=mode))
                    cfg = replace(cfg, inference=replace(cfg.inference, n_explore_steps=2))
                    model = HDPGPC(cfg).fit(values[:12], grid=decoder.grid)
                    self.assertTrue(torch.isfinite(model._backend.q_last).all())
                    self.assertTrue(model.offline_search_diagnostics_['active'])
                    diagnostics = model.offline_search_diagnostics_
                    diagnostics['outputs'][0]['retained_rank'] = -1
                    self.assertGreater(model.offline_search_diagnostics_['outputs'][0]['retained_rank'], 0)

    def test_geometry_rebuilt_for_new_data_and_legacy_selection(self):
        state = self.clone()
        first = search.initialize(state, state._split_merge_y)
        changed = state._split_merge_y.flip(0)
        second = search.initialize(state, changed)
        self.assertIsNot(first, second)
        torch.testing.assert_close(first.features.flip(0), second.features, atol=1e-10, rtol=1e-10)
        state.offline_candidate_search = 'legacy'
        self.assertIsNone(search.initialize(state, changed))
        self.assertIsNone(search.birth_seeds(state, state.resp_last, state.q_lat_last,
                                             state.snr_last, state.f_ind_old))

    def test_setting_roundtrip_validation_and_record_arguments(self):
        from hdpgpc.experiments.records import build_parser, make_config
        _, decoder = observations()
        config = configuration(decoder)
        raw = config.to_dict()
        self.assertEqual(HDPGPCConfig.from_dict(raw).inference.offline_candidate_search, 'noise_aware')
        config = replace(config, inference=replace(config.inference, offline_candidate_search='legacy'))
        self.assertEqual(HDPGPCConfig.from_yaml(config.to_yaml()), config)
        with self.assertRaisesRegex(ValueError, 'offline_candidate_search'):
            replace(config, inference=replace(config.inference, offline_candidate_search='invalid'))
        for method in ('score_guided', 'noise_aware', 'dynamics', 'combined', 'legacy'):
            args = build_parser('mitbih', 'autoencoder').parse_args(['--offline-candidate-search', method])
            expected = 'score_guided' if method == 'legacy' else method
            self.assertEqual(make_config(decoder.grid, args, decoder=decoder).inference.offline_candidate_search, expected)

    def test_native_both_strategies_both_clocks_and_soft_assignments(self):
        values, decoder = observations(points=8, dimension=2)
        values = values[:12]
        values[6:, :2] -= 8.
        for strategy in ('legacy', 'split_merge'):
            for clock in ('cluster-clock', 'physical-clock'):
                for hard in (False, True):
                    with self.subTest(strategy=strategy, clock=clock, hard=hard):
                        cfg = configuration(decoder, strategy, clock, hard)
                        cfg = replace(cfg, inference=replace(cfg.inference, n_explore_steps=2, it_limit=2))
                        with patch.object(search, 'rank_splits', wraps=search.rank_splits) as screened:
                            model = HDPGPC(cfg).fit(values, grid=decoder.grid)
                        self.assertTrue(screened.called)
                        state = model._backend
                        self.assertTrue(state.offline_search_diagnostics['active'])
                        self.assertTrue(torch.isfinite(state.q_last).all())
                        torch.testing.assert_close(state.resp_last.sum(1), torch.ones(len(values)))
                        self.assertEqual(len(model.labels_), len(values))


if __name__ == '__main__':
    unittest.main()
