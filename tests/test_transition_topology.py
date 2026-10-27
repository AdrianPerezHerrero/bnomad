"""Numerical regressions for the ordered transition prior."""
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
