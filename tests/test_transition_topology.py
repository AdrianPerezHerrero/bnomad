"""Numerical regressions for the ordered transition prior."""
from dataclasses import replace
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import betaln, expit, roots_legendre

from hdpgpc.config import HDPConfig, HDPGPCConfig, InferenceConfig
from hdpgpc.hdp import _row_ratio, _topology
from hdpgpc.hdp._topology import row_moments, optimize_sticks


class OrderedRowMomentsTests(unittest.TestCase):
    def test_underflowing_ratio_weights_use_stable_moments(self):
        # An optimizer trial within the allowed Beta-shape bounds. The 1-D
        # rule has subnormal mass and previously returned a normalizer of -inf.
        shapes = (410.5906565964896, 7799.848693309597,
                  0.04142056733985268, 52.898888248057695)
        reference = (0.9885372186422609, -18.035488717587324,
                     0.011462781357739388)
        row_moments.cache_clear()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            coarse = row_moments(*shapes, 100., 0., order=128)
            fine = row_moments(*shapes, 100., 0., order=256)
        self.assertTrue(np.isfinite(coarse).all())
        np.testing.assert_allclose(coarse, reference, rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(fine, coarse, rtol=1e-8, atol=1e-8)
        self.assertAlmostEqual(coarse[0] + coarse[2], 1.)

    def test_tiny_normal_mass_also_uses_stable_moments(self):
        # Mass is about 4e-298: still normal float64, but the previous
        # endpoint subtraction produced a spurious O(1e282) normalizer.
        shapes = (448.14306549502146, 10560.924910829091,
                  0.022692155605089224, 2119.4817113270124)
        reference = (0.9997499542100672, -44.38562565737881,
                     0.0002500457899329862)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = row_moments(*shapes, 100., 0.)
        np.testing.assert_allclose(result, reference, rtol=1e-8, atol=1e-10)

    def test_integrable_endpoint_overflow_uses_direct_rule(self):
        # hyp2f1(1, 2, 3, r) spuriously returns inf extremely close to r=1.
        # For independent uniform v,w, E[v/(v+(1-v)*w)] = pi**2/6 - 1.
        row_moments.cache_clear()
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            coarse = row_moments(1., 1., 1., 1., 10., 0., order=128)
            fine = row_moments(1., 1., 1., 1., 10., 0., order=256)
        self.assertAlmostEqual(coarse[0], np.pi**2/6-1, places=10)
        np.testing.assert_allclose(coarse, fine, rtol=1e-7, atol=1e-8)
        self.assertAlmostEqual(fine[1], 3.3889486367124775, places=7)

    def test_optimizer_rejects_invalid_trials_without_invalid_differences(self):
        for invalid in ('exception', 'nan'):
            calls = []
            state = SimpleNamespace(rho=np.array([.5]), omega=np.array([2.]),
                                    device='cpu')
            def terms(state, counts, rho, omega, theta, **kwargs):
                if rho[0] > .5:
                    calls.append(invalid)
                    if invalid == 'exception':
                        raise FloatingPointError('Unreliable quadrature')
                    return np.nan
                return 1.
            with self.subTest(invalid=invalid), warnings.catch_warnings(), patch(
                    'hdpgpc.hdp._topology.linear_terms', side_effect=terms):
                warnings.simplefilter('error', RuntimeWarning)
                rho, omega, info = optimize_sticks(state, None)
            self.assertTrue(calls)  # Exercise rejected finite-difference probes.
            np.testing.assert_allclose(rho, state.rho)
            np.testing.assert_allclose(omega, state.omega)
            self.assertTrue(np.isfinite(info['objective']))
            self.assertEqual(info['objective'], -1.)

    def test_failed_fine_audit_keeps_initial_sticks(self):
        state = SimpleNamespace(rho=np.array([.5]), omega=np.array([2.]),
                                device='cpu')
        candidate = SimpleNamespace(x=np.log([2., 1.]), fun=-10.,
                                    success=True, nit=1)
        def terms(state, counts, rho, omega, theta, *, order=128):
            if order == 256:
                raise FloatingPointError('Failed fine audit')
            return 10. if rho[0] > .5 else 1.
        with warnings.catch_warnings(), patch(
                'hdpgpc.hdp._topology.linear_terms', side_effect=terms), patch(
                'hdpgpc.hdp._topology.minimize', return_value=candidate):
            warnings.simplefilter('error', RuntimeWarning)
            rho, omega, info = optimize_sticks(state, None)
        np.testing.assert_allclose(rho, state.rho)
        np.testing.assert_allclose(omega, state.omega)
        self.assertFalse(info['quadrature_accepted'])
        self.assertEqual(info['objective'], -1.)

    def test_regular_ratio_moments_remain_unchanged(self):
        for kappa, normalizer in ((0., 55.01189873987044),
                                  (10., 67.31761576678784)):
            with self.subTest(kappa=kappa), warnings.catch_warnings():
                warnings.simplefilter('error', RuntimeWarning)
                result = row_moments(2., 8., 4., 3., 100., kappa)
            np.testing.assert_allclose(result,
                (0.3061224489795924, normalizer, 0.6938775510204078),
                rtol=1e-10, atol=1e-10)


class RowRatioSolverTests(unittest.TestCase):
    def problem(self):
        a, b = np.array([2., 4.]), np.array([8., 3.])
        state = SimpleNamespace(rho=a/(a+b), omega=a+b, gamma=6., transAlpha=100.,
                                kappa=10., device='cpu')
        theta, _ = _topology.posterior_matrices(state, [[8., 1.], [0., 5.]])
        return state, theta

    def test_direct_gradient_matches_independent_quadrature(self):
        state, theta = self.problem()
        start, args = _row_ratio._context(state, theta)
        value, gradient = _row_ratio._value_gradient(start, *args)
        gamma, alpha, kappa, log_pi, entropy = args
        nodes, weights = roots_legendre(384)
        x = 45*nodes
        log_v, log_one_minus_v = -np.logaddexp(0., -x), -np.logaddexp(0., x)
        ratio_log_odds = x[:, None]-log_v[None, :]
        stay = alpha*expit(ratio_log_odds)+kappa
        advance = alpha*expit(-ratio_log_odds)

        def beta_mass(a, b):
            return 45*weights*np.exp(a*log_v+b*log_one_minus_v-betaln(a, b))

        def quadrature(point):
            # Integrate the two independent logit sticks directly. The old
            # hypergeometric quadrature has a numerical discontinuity near
            # the integer parameter reached by this test's b[0] perturbation.
            a, b = np.exp(point).reshape(2, -1)
            total = entropy.sum()-_topology._beta_kl(a, b, gamma)
            for i, (log_stay, log_advance) in enumerate(log_pi):
                c, d = (a[i+1], b[i+1]) if i+1 < len(a) else (1., gamma)
                row = -betaln(stay, advance)+(stay-1)*log_stay+(advance-1)*log_advance
                total += beta_mass(a[i], b[i]) @ row @ beta_mass(c, d)
            return total

        shifts = 1e-4*np.eye(len(start))
        reference = [(quadrature(start+shift)-quadrature(start-shift))/2e-4 for shift in shifts]
        np.testing.assert_allclose(value, quadrature(start), atol=1e-8, rtol=0)
        np.testing.assert_allclose(gradient, reference, atol=1e-6, rtol=0)
        _, _, info = _row_ratio.optimize_sticks(state, theta)
        self.assertFalse(info['used_fallback'])
        self.assertTrue(info['quadrature_accepted'])
        self.assertGreater(info['objective'], value)

    def test_failed_refined_integral_calls_current_solver_from_original_state(self):
        state, theta = self.problem()
        original = _row_ratio._value_gradient

        def fail_refinement(parameters, *args, **grid):
            if grid:
                raise FloatingPointError('failed test audit')
            return original(parameters, *args)

        with patch.object(_row_ratio, '_value_gradient', side_effect=fail_refinement), \
                patch.object(_topology, 'optimize_sticks', wraps=optimize_sticks) as fallback:
            rho, omega, info = _row_ratio.optimize_sticks(state, theta)
        fallback.assert_called_once_with(state, theta)
        expected_rho, expected_omega, _ = optimize_sticks(state, theta)
        np.testing.assert_array_equal(rho, expected_rho)
        np.testing.assert_array_equal(omega, expected_omega)
        self.assertEqual(info['solver'], 'quadrature')
        self.assertEqual(info['fallback_reason'], 'failed test audit')

    def test_solver_choice_roundtrips_and_requires_ordered_inference(self):
        inference = InferenceConfig(schedule='offline', offline_strategy='split_merge',
                                    ordered_hdp_solver='row_ratio')
        config = HDPGPCConfig(hdp=HDPConfig(transition_topology='left_to_right'), inference=inference)
        self.assertEqual(HDPGPCConfig.from_yaml(config.to_yaml()), config)
        with self.assertRaisesRegex(ValueError, 'requires transition_topology=left_to_right'):
            replace(config, hdp=HDPConfig())
        default = replace(config, inference=replace(inference, ordered_hdp_solver='quadrature'))
        self.assertNotIn('ordered_hdp_solver', default.to_dict()['inference'])
        self.assertEqual(HDPGPCConfig.from_dict(default.to_dict()).inference.ordered_hdp_solver, 'quadrature')


if __name__ == '__main__':
    unittest.main()
