"""Constant-input filtering and gap smoothing against exact Gaussian conditioning."""
from types import SimpleNamespace
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import multivariate_normal

from hdpgpc.dynamics import _clock
from hdpgpc.dynamics._kalman import smooth_sequence, smooth_pair
from hdpgpc.config import AlignmentConfig, with_mfa_emission
from hdpgpc.emissions import AffineGaussianObservation
from test_dynamics_clock import emission
from test_public_api import config


def affine_emission():
    gp = emission()
    gp.constant_input = True
    gp.b = [torch.tensor([[.4]], dtype=torch.float64)]
    gp.b_def = gp.b[0].clone()
    return gp


class AffineClockTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_native_filter_learning_and_conversion_for_both_clocks(self):
        from hdpgpc.dynamics._trajectory import hold_lds_params
        for clock in ('cluster-clock', 'physical-clock'):
            with self.subTest(clock=clock):
                cfg = config('dynamic')
                cfg = replace(cfg, verbose=False, numerics=replace(cfg.numerics, cuda=True),
                    alignment=AlignmentConfig(policy='none'), emission=replace(cfg.emission, fit_gp=False),
                    dynamics=replace(cfg.dynamics, clock=clock, constant_input=True,
                                     lds_update='conjugate_batch'))
                gp = cfg.build_model().gpmodels[0][0]
                grid = gp.x_basis
                observation = torch.ones((len(grid), 1), dtype=torch.float64, device='cuda')
                gp.include_weighted_sample(0, grid, grid, observation, 1.)
                hold_lds_params(gp)
                self.assertEqual(gp.b_def.device.type, 'cuda')
                gp.b[-1] = torch.full_like(gp.f_star_sm[-1], .25)
                if clock == 'physical-clock':
                    first = _clock.trajectory(gp).filtered_means[0].clone()
                    expected = first
                    for _ in range(2):
                        expected = gp.A[-1] @ expected + gp.b[-1]
                    gp.include_weighted_sample(2, grid, grid, observation, 0.)
                    torch.testing.assert_close(_clock.trajectory(gp).means[2], expected)
                gp.include_weighted_sample(3, grid, grid, 1.1 * observation, 1.)
                hold_lds_params(gp)
                gp.include_weighted_sample(4, grid, grid, 1.2 * observation, 1.)
                hold_lds_params(gp)
                gp.backwards()
                gp.bayesian_batch_params(torch.ones(5, dtype=torch.float64, device='cuda'))
                self.assertTrue(torch.isfinite(gp.b[-1]).all())
                self.assertGreater(float(gp.b[-1].norm()), 0.)
                self.assertEqual(gp.internal_params.get_intercept().device.type, 'cuda')
                torch.testing.assert_close(gp.b[-1], gp.internal_params.get_intercept())
                learned = gp.b[-1].detach().cpu().clone()
                gp.model_to_cpu()
                torch.testing.assert_close(gp.b[-1], learned)
                gp.model_to_numpy()
                np.testing.assert_allclose(gp.b[-1], learned.numpy())
                gp.model_to_torch()
                torch.testing.assert_close(gp.b[-1], learned)
                gp.cuda = True
                gp.model_to_cuda()
                self.assertEqual(gp.b[-1].device.type, 'cuda')
                torch.testing.assert_close(gp.b[-1].cpu(), learned)
                if clock == 'physical-clock':
                    self.assertEqual(_clock.trajectory(gp).offsets[-1].device.type, 'cuda')

    def test_independent_factor_fast_path_matches_scalar_filter_and_latent_scores(self):
        decoder = AffineGaussianObservation(np.array([[1.], [.3]]), np.diag([.2, .4]),
            np.arange(2.)[:, None], np.zeros(2), np.eye(1))
        from test_dynamics_clock import recipe
        base = recipe('cluster-clock')
        base = replace(base, dynamics=replace(base.dynamics, constant_input=True))
        cfg = with_mfa_emission(base, decoder, dynamic=False)
        observations = torch.tensor([[[1.], [.3]], [[1.1], [.2]], [[.9], [.4]], [[1.2], [.25]]],
                                    dtype=torch.float64)
        weights = torch.ones(4, dtype=torch.float64)
        results = []
        for fast in (True, False):
            gp = cfg.build_model().gpmodels[0][0]
            gp.b[-1] = torch.full_like(gp.b[-1], .35)
            grids = gp.direct_observation_grid[None].repeat(4, 1, 1)
            if fast:
                scores = gp.full_pass_weighted(grids, observations, weights)
            else:
                with patch('hdpgpc.dynamics._independent.full_pass', return_value=False):
                    scores = gp.full_pass_weighted(grids, observations, weights)
            scalar = torch.stack([gp.log_lat_error(position, 1.) for position in range(gp.N)])
            from hdpgpc.dynamics._independent import latent_scores
            torch.testing.assert_close(latent_scores(gp, gp.N), scalar, rtol=1e-6, atol=1e-6)
            # MFA assignment scores use fresh local moments under the updated
            # observation parameters, rather than the retained E-step path.
            local_means, local_covariances = gp._factor_local_moments
            expected_means, _ = gp.fixed_observation_model.condition_batch(
                gp.b[-1], gp.ini_cov_def, observations[1:], responsibilities=weights[1:])
            torch.testing.assert_close(local_means[1:], expected_means)
            residuals = local_means-gp.b[-1]
            residuals[0] = local_means[0]
            precision = torch.linalg.inv(gp.ini_cov_def)
            expected = .5*(local_means.shape[1]+torch.linalg.slogdet(local_covariances).logabsdet
                - torch.linalg.slogdet(gp.ini_cov_def).logabsdet
                - torch.einsum('nix,ij,njx->n',residuals,precision,residuals)
                - torch.einsum('ij,nji->n',precision,local_covariances))
            torch.testing.assert_close(scores[1], expected, rtol=1e-6, atol=1e-6)
            results.append((torch.stack(gp.f_star_sm), torch.stack(gp.cov_f_sm), scores,
                            gp.b[-1], gp.C[-1], gp.Sigma[-1]))
        for fast, scalar in zip(results[0], results[1]):
            if isinstance(fast, tuple):
                for first, second in zip(fast, scalar):
                    torch.testing.assert_close(first, second, rtol=1e-6, atol=1e-6)
            else:
                torch.testing.assert_close(fast, scalar, rtol=1e-6, atol=1e-6)

    def test_prediction_only_ticks_and_read_only_future_include_constant_input(self):
        gp = affine_emission()
        state = _clock.advance(gp, 3)
        np.testing.assert_allclose([float(mean.item()) for mean in state.means],
                                   [0., .4, .72, .976])
        expected_covariance = .8**6 + .1 * (1 + .8**2 + .8**4)
        self.assertAlmostEqual(float(state.covariances[3]), expected_covariance)
        future_mean, _ = _clock.state_at(gp, 5)
        self.assertAlmostEqual(float(future_mean), .4 * sum(.8**j for j in range(5)))
        self.assertEqual(len(state.means), 4)
        self.assertEqual(gp.N, 0)

    def test_gap_smoother_matches_joint_affine_gaussian(self):
        gp = affine_emission()
        grid = gp.direct_observation_grid
        observations = np.array([[1., .3], [.5, -.2]])
        gp.include_weighted_sample(0, grid, grid, torch.tensor(observations[0, :, None]), 1.)
        gp.include_weighted_sample(3, grid, grid, torch.tensor(observations[1, :, None]), 1.)
        gp.backwards()
        state = _clock.trajectory(gp)
        prior_mean = np.array([.4 * sum(.8**j for j in range(t)) for t in range(4)])
        variance = [1.]
        for _ in range(3):
            variance.append(.8**2 * variance[-1] + .1)
        covariance = np.array([[variance[min(i, j)] * .8**abs(i-j)
                                for j in range(4)] for i in range(4)])
        H = np.zeros((4, 4))
        H[:2, 0] = [1., .3]
        H[2:, 3] = [1., .3]
        R = np.diag([.2, .4, .2, .4])
        S = H @ covariance @ H.T + R
        expected_mean = prior_mean + covariance @ H.T @ np.linalg.solve(
            S, observations.reshape(-1) - H @ prior_mean)
        expected_covariance = covariance - covariance @ H.T @ np.linalg.solve(S, H @ covariance)
        # The production smoother deliberately regularizes its Cholesky solves
        # at 1e-8, so compare with the unregularized joint Gaussian at that scale.
        np.testing.assert_allclose(torch.stack(state.means).numpy().reshape(-1), expected_mean,
                                   rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(torch.stack(state.covariances).numpy().reshape(-1),
                                   np.diag(expected_covariance), rtol=1e-7, atol=1e-7)
        # Scoring under the saved historical parameters must include the same b.
        gp.lds_scoring = 'historical'
        historical = _clock.latent_scores(gp)
        gp.lds_scoring = 'final'
        torch.testing.assert_close(_clock.latent_scores(gp), historical)
        score = sum(float(_clock.observation_score(gp, grid, torch.tensor(value[:, None]), tick))
                    for tick, value in zip((0, 3), observations))
        score += float(_clock.latent_scores(gp).sum())
        self.assertAlmostEqual(score, multivariate_normal.logpdf(
            observations.reshape(-1), mean=H @ prior_mean, cov=S), places=5)
        gp.b[-1] = gp.b[-1] + .2
        gp.lds_scoring = 'historical'
        torch.testing.assert_close(_clock.latent_scores(gp), historical)
        gp.lds_scoring = 'final'
        self.assertFalse(torch.allclose(_clock.latent_scores(gp), historical))
        _clock.convert(gp, 'numpy')
        self.assertIsInstance(state.offsets[0], np.ndarray)
        _clock.convert(gp, 'torch')
        self.assertIsInstance(state.offsets[0], torch.Tensor)
        torch.testing.assert_close(state.offsets[-1], torch.tensor([[.4]], dtype=torch.float64))

    def test_cluster_filter_adds_input_before_affine_measurement_conditioning(self):
        gp = affine_emission()
        mean = torch.tensor([[.6]], dtype=torch.float64)
        covariance = torch.tensor([[.7]], dtype=torch.float64)
        observation = torch.tensor([[.9], [.1]], dtype=torch.float64)
        actual = gp.gp.posterior(mean, covariance, observation, gp.A[-1], gp.Gamma[-1],
            gp.C[-1], gp.Sigma[-1], b=gp.b[-1], x_train=gp.direct_observation_grid)
        expected = gp.fixed_observation_model.condition(gp.A[-1] @ mean + gp.b[-1],
            gp.A[-1] @ covariance @ gp.A[-1].T + gp.Gamma[-1], observation,
            noise=gp.Sigma[-1])
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])

    def test_sampled_input_warp_conditioning_uses_one_affine_transition(self):
        cfg = config('dynamic')
        cfg = replace(cfg, verbose=False, emission=replace(cfg.emission, fit_gp=False),
                      dynamics=replace(cfg.dynamics, constant_input=True, clock='cluster-clock'))
        gp = cfg.build_model().gpmodels[0][0]
        grid = gp.x_basis
        observation = torch.linspace(.1, .5, len(grid), dtype=torch.float64).reshape(-1, 1)
        grids = torch.stack((grid, grid + .01))
        gp.include_weighted_sample_flow(0, grid, grids, observation, 1.)
        gp.b[-1] = torch.full_like(gp.f_star_sm[-1], .35)
        A, Q, C, R = gp.A[-1], gp.Gamma[-1], gp.C[-1], gp.Sigma[-1]
        prior_mean = A @ gp.f_star_sm[-1] + gp.b[-1]
        eye = torch.eye(len(grid), dtype=torch.float64)
        prior_covariance = A @ gp.cov_f_sm[-1] @ A.T + Q + 1e-6 * eye
        precision = torch.linalg.inv(prior_covariance)
        information = precision @ prior_mean
        basis_covariance = torch.tensor(gp.gp.kernel(grid, grid), dtype=torch.float64) + 1e-4 * eye
        noise = torch.diag(R).mean() * eye + 1e-6 * eye
        for warped in grids:
            cross = torch.tensor(gp.gp.kernel(grid, warped), dtype=torch.float64)
            H = torch.linalg.solve(basis_covariance, cross).T @ C
            precision += .5 * H.T @ torch.linalg.solve(noise, H)
            information += .5 * H.T @ torch.linalg.solve(noise, observation)
        expected_covariance = torch.linalg.inv(precision)
        expected_mean = expected_covariance @ information
        gp.include_weighted_sample_flow(1, grid, grids, observation, 1.)
        torch.testing.assert_close(gp.f_star_sm[-1], expected_mean, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(gp.cov_f_sm[-1], expected_covariance, rtol=1e-9, atol=1e-9)

    def test_cluster_smoothers_use_affine_predicted_mean(self):
        # Identical filtered/smoothed means at an exactly predicted second tick
        # should leave the first mean unchanged under either smoother wrapper.
        means = [torch.tensor([[1.]]), torch.tensor([[1.2]])]
        covariances = [torch.tensor([[.5]]), torch.tensor([[.3]])]
        A, Q, b = torch.tensor([[.8]]), torch.tensor([[.1]]), torch.tensor([[.4]])
        for sequence in (False, True):
            function = smooth_sequence if sequence else smooth_pair
            matrices = ([A], [Q], [b]) if sequence else (A, Q, b)
            smoothed, _ = function(SimpleNamespace(), matrices[0], matrices[1],
                [value.clone() for value in means], [value.clone() for value in covariances],
                b_prior=matrices[2])
            torch.testing.assert_close(smoothed[0], means[0])


if __name__ == '__main__':
    unittest.main()
