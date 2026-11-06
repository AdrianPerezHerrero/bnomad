"""Analytic affine-regression posterior checks independent of clustering quality."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from hdpgpc.dynamics._affine import (make_transition_prior,
    augment_transition_statistics, fit_transition_moments, AffineTransitionPosterior)
from hdpgpc.dynamics._parameter_posterior import make_pair_prior


def policy(mode='cluster', strength=1.):
    return SimpleNamespace(mode=mode, strength=strength)


class AffineTransitionPosteriorTests(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.A = torch.tensor([[.8, .1], [-.05, .9]])
        self.Q = torch.tensor([[.2, .03], [.03, .3]])
        self.x = torch.tensor([[.2, -.3, .5, 1.2], [-.2, .6, .7, -.8]])
        self.y = self.A @ self.x + torch.tensor([[.4], [-.3]])
        self.y += torch.tensor([[.01, -.03, .02, .04], [-.02, .01, .02, -.03]])
        self.weights = torch.tensor([.2, .7, 1., .4])
        # Joint latent uncertainty, including a nonzero lag cross covariance.
        self.joint = torch.tensor([[.03, .01, .01, .00], [.01, .04, .00, .01],
                                   [.01, .00, .05, .01], [.00, .01, .01, .06]])

    def tearDown(self):
        torch.set_default_dtype(self.previous_dtype)

    def prior(self, matrix='cluster', noise='cluster'):
        return make_transition_prior(self.A, self.Q, 8., policy(matrix, 2.),
            policy(noise), constant_input=True, intercept_strength=.7)

    def statistics(self, positions=None):
        positions = slice(None) if positions is None else positions
        weights = self.weights[positions]
        roots = weights.sqrt()
        mass = float(weights.sum())
        covariance = mass * self.joint
        return (mass, self.y[:, positions] * roots, self.x[:, positions] * roots,
                covariance[:2, :2], covariance[2:, 2:], covariance[:2, 2:]), roots

    def test_full_posterior_matches_augmented_normal_equations(self):
        prior = self.prior(matrix='shrinkage')
        stats, roots = self.statistics()
        mass, y, x, yy, xx, yx = stats
        posterior = prior.posterior(*stats, intercept_weights=roots)
        u, augmented_cov, cross = augment_transition_statistics(*stats, intercept_weights=roots)
        K = prior.m_r_cov + u @ u.T + augmented_cov
        information = prior.m_mean @ prior.m_r_cov + y @ u.T + cross
        expected_mean = torch.linalg.solve(K, information.T).T
        expected_scale = (prior.n0 * prior.scale + y @ y.T + yy
            + prior.m_mean @ prior.m_r_cov @ prior.m_mean.T - expected_mean @ K @ expected_mean.T)
        expected_scale /= prior.n0 + mass
        torch.testing.assert_close(posterior.m_mean, expected_mean)
        torch.testing.assert_close(posterior.scale, expected_scale)
        torch.testing.assert_close(posterior.get_mean(), expected_mean[:, :-1])
        torch.testing.assert_close(posterior.get_intercept(), expected_mean[:, -1:])

    def test_fixed_A_learns_intercept_and_counts_only_its_uncertainty(self):
        prior = self.prior(matrix='fixed')
        stats, roots = self.statistics()
        mass, y, x, yy, xx, yx = stats
        posterior = prior.posterior(*stats, intercept_weights=roots)
        expected_b = (y - self.A @ x) @ roots[:, None] / (.7 + mass)
        torch.testing.assert_close(posterior.get_mean(), self.A, rtol=0., atol=0.)
        torch.testing.assert_close(posterior.get_intercept(), expected_b)
        self.assertGreater(float(posterior.kl_divergence(prior)), 0.)
        self.assertAlmostEqual(float(prior.kl_divergence(prior)), 0., places=12)

    def test_sequential_updates_equal_batch_for_fixed_expected_statistics(self):
        for matrix in ('cluster', 'fixed'):
            for noise in ('cluster', 'cluster_diagonal', 'fixed'):
                with self.subTest(matrix=matrix, noise=noise):
                    prior = self.prior(matrix, noise)
                    all_stats, roots = self.statistics()
                    batch = prior.posterior(*all_stats, intercept_weights=roots)
                    sequential = prior
                    for position in range(4):
                        stats, weight = self.statistics([position])
                        sequential = sequential.posterior(*stats, intercept_weights=weight)
                    torch.testing.assert_close(batch.m_mean, sequential.m_mean)
                    torch.testing.assert_close(batch.scale, sequential.scale)
                    torch.testing.assert_close(batch.m_r_cov, sequential.m_r_cov)
                    self.assertGreater(float(batch.kl_divergence(prior)), 0.)
                    if noise == 'fixed':
                        torch.testing.assert_close(batch.scale, prior.scale, rtol=0., atol=0.)
                    if 'diagonal' in noise:
                        torch.testing.assert_close(batch.scale, torch.diag(batch.scale.diag()))

    def test_natural_gradient_scales_all_statistics_and_intercept_weights(self):
        prior = self.prior()
        stats, roots = self.statistics()
        fraction = .3
        posterior = prior.posterior_natural_gradient(*stats, step_size=fraction,
                                                      intercept_weights=roots)
        mass, y, x, yy, xx, yx = stats
        expected = prior.posterior(fraction * mass, fraction**.5 * y, fraction**.5 * x,
            fraction * yy, fraction * xx, fraction * yx,
            intercept_weights=fraction**.5 * roots)
        torch.testing.assert_close(posterior.m_mean, expected.m_mean)
        torch.testing.assert_close(posterior.scale, expected.scale)
        legacy = prior.posterior_legacy(*stats, intercept_weights=roots)
        batch = prior.posterior(*stats, intercept_weights=roots)
        torch.testing.assert_close(legacy.m_mean, batch.m_mean)

    def test_disabled_priors_are_original_types_and_values(self):
        for matrix in ('cluster', 'fixed', 'shrinkage'):
            for noise in ('cluster', 'cluster_diagonal', 'fixed'):
                pA, pQ = policy(matrix), policy(noise)
                disabled = make_transition_prior(self.A, self.Q, 8., pA, pQ)
                old = make_pair_prior(self.A, self.Q, 8., pA, pQ)
                self.assertIs(type(disabled), type(old))
                torch.testing.assert_close(disabled.m_mean, old.m_mean, rtol=0., atol=0.)
                torch.testing.assert_close(disabled.m_r_cov, old.m_r_cov, rtol=0., atol=0.)

    def test_conversions_and_weight_validation(self):
        prior = self.prior('fixed')
        stats, roots = self.statistics()
        posterior = prior.posterior(*stats, intercept_weights=roots)
        mean = posterior.get_intercept().clone()
        posterior.to_numpy()
        self.assertIsInstance(posterior.get_intercept(), np.ndarray)
        posterior.to_torch()
        torch.testing.assert_close(posterior.get_intercept(), mean)
        self.assertTrue(torch.isfinite(posterior.kl_divergence(prior)))
        with self.assertRaisesRegex(ValueError, 'responsibility mass'):
            prior.posterior(*stats, intercept_weights=torch.ones(4))

    def test_ml_expected_statistics_respect_fixed_policies(self):
        # With observed states, unrestricted affine ML recovers the exact input.
        zeros = torch.zeros((2, 2))
        b = torch.tensor([[.4], [-.3]])
        y = self.A @ self.x + b
        A, estimate_b, Q = fit_transition_moments(self.A, self.Q, 4., y, self.x,
            zeros, zeros, zeros, matrix_policy=policy(), noise_policy=policy('fixed'))
        torch.testing.assert_close(A, self.A)
        torch.testing.assert_close(estimate_b, b)
        torch.testing.assert_close(Q, self.Q, rtol=0., atol=0.)
        stats, roots = self.statistics()
        A, estimate_b, Q = fit_transition_moments(self.A, self.Q, *stats,
            matrix_policy=policy('fixed'), noise_policy=policy('cluster_diagonal'),
            intercept_weights=roots)
        torch.testing.assert_close(A, self.A, rtol=0., atol=0.)
        torch.testing.assert_close(estimate_b, b +
            ((self.y - self.A @ self.x - b) * self.weights).sum(1, keepdim=True) / self.weights.sum())
        torch.testing.assert_close(Q, torch.diag(Q.diag()))
        self.assertGreater(float(Q.diag().min()), 0.)

    def test_fixed_correlated_noise_KL_matches_vector_gaussian(self):
        # Fixed full Gamma requires a correlated matrix-normal, not independent rows.
        for matrix in ('cluster', 'fixed'):
            with self.subTest(matrix=matrix):
                prior = self.prior(matrix, 'fixed')
                stats, roots = self.statistics()
                posterior = prior.posterior(*stats, intercept_weights=roots)
                q, p = posterior._free_distribution(), prior._free_distribution()
                mean_q, mean_p = q.m_mean.numpy().reshape(-1), p.m_mean.numpy().reshape(-1)
                covariance_q = np.kron(self.Q.numpy(), np.linalg.inv(q.m_r_cov.numpy()))
                covariance_p = np.kron(self.Q.numpy(), np.linalg.inv(p.m_r_cov.numpy()))
                delta = mean_q-mean_p
                expected = .5*(np.trace(np.linalg.solve(covariance_p, covariance_q))
                    + delta @ np.linalg.solve(covariance_p, delta)-len(delta)
                    + np.linalg.slogdet(covariance_p)[1]-np.linalg.slogdet(covariance_q)[1])
                self.assertAlmostEqual(float(posterior.kl_divergence(prior)), expected, places=11)

    def test_full_noise_KL_matches_scipy_inverse_wishart_expectations(self):
        from scipy.special import digamma
        from scipy.stats import invwishart
        for matrix in ('cluster', 'fixed'):
            with self.subTest(matrix=matrix):
                prior = self.prior(matrix, 'cluster')
                stats, roots = self.statistics()
                posterior = prior.posterior(*stats, intercept_weights=roots)
                q, p = posterior._free_distribution(), prior._free_distribution()
                d, columns = q.m_mean.shape
                nu_q, nu_p = float(q.n0)+d+1, float(p.n0)+d+1
                psi_q, psi_p = float(q.n0)*q.scale.numpy(), float(p.n0)*p.scale.numpy()
                expected_precision = nu_q*np.linalg.inv(psi_q)
                expected_logdet = np.linalg.slogdet(psi_q)[1]-d*np.log(2.)
                expected_logdet -= digamma((nu_q+1-np.arange(1,d+1))/2).sum()
                log_p = invwishart.logpdf(np.eye(d), df=nu_p, scale=psi_p)
                expected_log_p = (log_p+.5*np.trace(psi_p)
                    -.5*(nu_p+d+1)*expected_logdet-.5*np.trace(psi_p @ expected_precision))
                log_q = invwishart.logpdf(np.eye(d), df=nu_q, scale=psi_q)
                expected_log_q = (log_q+.5*np.trace(psi_q)
                    -.5*(nu_q+d+1)*expected_logdet-.5*np.trace(psi_q @ expected_precision))
                noise_kl = expected_log_q-expected_log_p
                Kq, Kp = q.m_r_cov.numpy(), p.m_r_cov.numpy()
                delta = q.m_mean.numpy()-p.m_mean.numpy()
                matrix_kl = .5*(d*(np.linalg.slogdet(Kq)[1]-np.linalg.slogdet(Kp)[1]
                    + np.trace(np.linalg.solve(Kq,Kp))-columns)
                    + np.trace(expected_precision @ delta @ Kp @ delta.T))
                self.assertAlmostEqual(float(posterior.kl_divergence(prior)),
                                       noise_kl+matrix_kl, places=10)

    def test_log_likelihood_with_fixed_noise_survives_numpy_round_trip(self):
        for matrix in ('cluster', 'fixed'):
            with self.subTest(matrix=matrix):
                prior = self.prior(matrix, 'fixed')
                stats, roots = self.statistics()
                posterior = prior.posterior(*stats, intercept_weights=roots)
                A, b = posterior.get_mean().clone(), posterior.get_intercept().clone()
                likelihood = prior.log_likelihood_MNIW(torch.cat((A,b),1), self.Q)
                free = prior._free_distribution()
                value = torch.cat((A,b),1) if matrix != 'fixed' else b
                delta = value-free.m_mean
                expected = -.5*torch.trace(torch.linalg.solve(self.Q,
                    delta @ free.m_r_cov @ delta.T+free.scale))
                torch.testing.assert_close(likelihood, expected)
                posterior.to_numpy(); posterior.to_torch()
                torch.testing.assert_close(posterior.get_mean(), A)
                torch.testing.assert_close(posterior.get_intercept(), b)
                torch.testing.assert_close(prior.log_likelihood_MNIW(
                    torch.cat((posterior.get_mean(),posterior.get_intercept()),1),self.Q), likelihood)

    def test_radius_sampling_excludes_input_and_supports_covariance_constraints(self):
        from hdpgpc.diagnostics._affine_dynamics import transition_radius_samples
        for matrix in ('cluster', 'fixed'):
            for noise in ('cluster', 'cluster_diagonal', 'fixed'):
                with self.subTest(matrix=matrix, noise=noise):
                    prior = self.prior(matrix, noise)
                    radii = transition_radius_samples(prior, 30, 7)
                    np.testing.assert_array_equal(radii, transition_radius_samples(prior,30,7))
                    self.assertTrue(np.isfinite(radii).all())
                    if matrix == 'fixed':
                        np.testing.assert_allclose(radii, np.max(np.abs(np.linalg.eigvals(self.A.numpy()))))
        scalar = AffineTransitionPosterior(torch.tensor([[0.,99.]]),
            torch.tensor([[4.,1.],[1.,2.]]),8.,torch.tensor([[.2]]),noise='fixed')
        radii = transition_radius_samples(scalar, 10000, 7)
        # rho(A)=abs(A) in one dimension. The intercept uncertainty is correlated
        # through the joint column covariance, but its mean must not enter rho.
        expected = np.sqrt(.2*np.linalg.inv(scalar.m_r_cov.numpy())[0,0])*np.sqrt(2/np.pi)
        self.assertAlmostEqual(float(radii.mean()), expected, delta=.005)


if __name__ == '__main__':
    unittest.main()
