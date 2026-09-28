"""Focused contracts for faithfully extracted calculations; no model training."""

from pathlib import Path
import math
import json
import sys
import unittest

import os
if not os.environ.get("HDPGPC_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdpgpc"))

import numpy as np
import torch

from hdpgpc.numerics._normalization import max_shift_log_scores
from hdpgpc.hdp._entropy import state_entropy_terms, pair_entropy_terms
from hdpgpc.alignment._covariance import warp_prior_rbf_covariance
from hdpgpc.alignment._grids import monotone_grid_from_expanded_controls, batch_monotone_grid


REFERENCES = json.loads((Path(__file__).parent / "fixtures/calculation_reference.json").read_text())['cases']


class MaxShiftTests(unittest.TestCase):
    def test_numpy_axes_dtype_and_no_mutation(self):
        for dtype in (np.float32, np.float64):
            x = np.array([[-5, -2], [-1, -4]], dtype=dtype)
            original = x.copy()
            for axis, expected, maxima in (
                (0, [[-4, 0], [0, -2]], [-1, -2]),
                (1, [[-3, 0], [0, -3]], [-2, -1]),
            ):
                actual, maximum = max_shift_log_scores(x, axis)
                np.testing.assert_array_equal(actual, np.array(expected, dtype=dtype))
                np.testing.assert_array_equal(maximum, np.array(maxima, dtype=dtype))
                self.assertEqual(actual.dtype, dtype)
            np.testing.assert_array_equal(x, original)

    def test_tensor_values_gradient_and_device(self):
        for device in (['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']):
            for dtype in (torch.float32, torch.float64):
                x = torch.tensor([[-5., -2.], [-1., -4.]], dtype=dtype,
                                 device=device, requires_grad=True)
                actual, maximum = max_shift_log_scores(x)
                torch.testing.assert_close(actual, x.new_tensor([[-3, 0], [0, -3]]), rtol=0, atol=0)
                torch.testing.assert_close(maximum, x.new_tensor([-2, -1]), rtol=0, atol=0)
                actual.sum().backward()
                torch.testing.assert_close(x.grad, x.new_tensor([[1, -1], [-1, 1]]), rtol=0, atol=0)
                self.assertEqual(actual.device, x.device)
                self.assertEqual(actual.dtype, x.dtype)

    def test_any_infinite_maximum_bypasses_entire_input(self):
        for value in (float('-inf'), float('inf')):
            array = np.array([[value, value], [-3., -1.]])
            for x in (array, torch.from_numpy(array.copy())):
                result, maximum = max_shift_log_scores(x)
                self.assertIs(result, x)
                self.assertEqual(float(maximum[1]), -1.)

    def test_nonfinite_nan_is_not_an_infinite_maximum(self):
        x = np.array([[np.nan, 1.], [2., 4.]])
        result, maximum = max_shift_log_scores(x)
        self.assertIsNot(result, x)
        self.assertTrue(np.isnan(result[0]).all())
        np.testing.assert_array_equal(result[1], [-2., 0.])

    def test_other_axis_returns_original_and_reduction(self):
        for x in (np.arange(8.).reshape(2, 2, 2), torch.arange(8.).reshape(2, 2, 2)):
            result, maximum = max_shift_log_scores(x, axis=2)
            self.assertIs(result, x)
            np.testing.assert_array_equal(np.asarray(maximum), [[1, 3], [5, 7]])

    def test_one_dimensional_numpy_scalar_special_case(self):
        result, maximum = max_shift_log_scores(np.array([2., 5.], dtype=np.float64), axis=0)
        np.testing.assert_array_equal(result, [[-3., 0.]])
        np.testing.assert_array_equal(maximum, [5.])
        # The legacy branch does not generalize this shape repair to float32.
        with self.assertRaises(IndexError):
            max_shift_log_scores(np.array([2., 5.], dtype=np.float32), axis=0)

    def test_legacy_method_remains_callable_without_model_state(self):
        from hdpgpc.model._state import ModelState

        result, maximum = ModelState.LogLik(None, np.array([[-3., -1.]]))
        np.testing.assert_array_equal(result, [[-2., 0.]])
        np.testing.assert_array_equal(maximum, [-1.])


class EntropyTests(unittest.TestCase):
    def test_state_term_sums_all_rows_without_normalizing(self):
        for dtype, tol in ((np.float32, 1e-7), (np.float64, 1e-15)):
            resp = np.array([[.25, .75], [.5, .5]], dtype=dtype)
            original = resp.copy()
            expected = [-.25 * math.log(.25) - .5 * math.log(.5),
                        -.75 * math.log(.75) - .5 * math.log(.5)]
            actual = state_entropy_terms(resp)
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
            self.assertEqual(actual.dtype, resp.dtype)
            np.testing.assert_array_equal(resp, original)

    def test_pair_terms_preserve_joint_weights_and_input(self):
        for dtype, tol in ((np.float32, 1e-7), (np.float64, 1e-15)):
            # First row has mass .5; result must use joint weights, not just
            # the entropy of normalized conditional probabilities.
            pair = np.array([[[.25, .25], [0., 0.]],
                             [[0., 0.], [.75, .25]]], dtype=dtype)
            original = pair.copy()
            expected = [[-.25 * math.log(.5), -.25 * math.log(.5)],
                        [-.75 * math.log(.75), -.25 * math.log(.25)]]
            actual = pair_entropy_terms(pair)
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)
            self.assertEqual(actual.dtype, pair.dtype)
            np.testing.assert_array_equal(pair, original)

    def test_custom_epsilon_is_used_in_both_pair_positions(self):
        pair = np.array([[[1., 0.], [0., 0.]]], dtype=np.float64)
        actual = pair_entropy_terms(pair, eps=.5)
        np.testing.assert_allclose(actual, [[-math.log(7/6), 0.], [0., 0.]],
                                   rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(state_entropy_terms(np.array([[1., 0.]]), eps=.5),
                                   [-math.log(1.5), 0.], rtol=0, atol=0)

    def test_zero_and_empty_counts_retain_shapes_and_finite_results(self):
        for n in (0, 2):
            np.testing.assert_array_equal(state_entropy_terms(np.zeros((n, 3))), np.zeros(3))
            np.testing.assert_array_equal(pair_entropy_terms(np.zeros((n, 3, 3))), np.zeros((3, 3)))

    def test_legacy_objective_calls_terms_without_model_state(self):
        from hdpgpc.model._state import ModelState

        resp = np.array([[.5, .5]])
        pair = np.array([[[.25, .25], [.25, .25]]])
        np.testing.assert_array_equal(ModelState.calc_Hstart(None, resp), state_entropy_terms(resp))
        np.testing.assert_array_equal(ModelState.calc_Htable(None, pair), pair_entropy_terms(pair))


class WarpCovarianceTests(unittest.TestCase):
    def test_original_covariance_and_gradients_exactly(self):
        for dtype in (torch.float32, torch.float64):
            for normalized in (False, True):
                name = 'cov_' + str(dtype).split('.')[-1] + '_' + str(normalized)
                case = REFERENCES[name]
                with self.subTest(case=name):
                    x = torch.tensor(case['x'], dtype=dtype, requires_grad=True)
                    noise = torch.tensor(case['noise'], dtype=dtype, requires_grad=True)
                    original = x.detach().clone()
                    actual = warp_prior_rbf_covariance(x, case['rho'], case['omega'], noise,
                                                       normalized, case['jitter'])
                    gx, gn = torch.autograd.grad(actual.sum(), (x, noise))
                    torch.testing.assert_close(actual, x.new_tensor(case['result']), rtol=0, atol=0)
                    torch.testing.assert_close(gx, x.new_tensor(case['gradient_x']), rtol=0, atol=0)
                    torch.testing.assert_close(gn, noise.new_tensor(case['gradient_noise']), rtol=0, atol=0)
                    torch.testing.assert_close(x, original, rtol=0, atol=0)
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.device, x.device)

    def test_degenerate_grid_retains_finite_constant_kernel(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.tensor([2., 2., 2.], dtype=dtype)
            actual = warp_prior_rbf_covariance(x, .5, 2., x.new_tensor(.125), True, 0.)
            expected = x.new_tensor([[4.125, 4., 4.], [4., 4.125, 4.], [4., 4., 4.125]])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_original_prior_wrapper_and_cache_still_own_factorization(self):
        from hdpgpc.alignment._amtgp import WarpPriorAMTGP

        prior = WarpPriorAMTGP(.03, jitter=2e-5, normalize_x=False)
        case = REFERENCES['cov_float64_False']
        x = torch.tensor(case['x'], dtype=torch.float64)
        noise = x.new_tensor(case['noise'])
        actual = prior._rbf_cov(x, .8, 1.4, noise)
        torch.testing.assert_close(actual, x.new_tensor(case['result']), rtol=0, atol=0)
        self.assertIsNone(prior._cache_L)
        prior._ensure_cache(x, .8, 1.4, noise)
        factor = prior._cache_L
        prior._ensure_cache(x, .8, 1.4, noise)
        self.assertIs(prior._cache_L, factor)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable; CPU goldens are the reference')
    def test_cuda_covariance_retains_device_and_gradient(self):
        x = torch.tensor([0., 1., 3.], device='cuda', dtype=torch.float64, requires_grad=True)
        actual = warp_prior_rbf_covariance(x, .8, 1.4, x.new_tensor(.03), True, 2e-5)
        self.assertEqual(actual.device, x.device)
        self.assertTrue(torch.isfinite(actual).all())
        actual.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class MonotoneGridTests(unittest.TestCase):
    def test_scalar_original_values_and_control_gradients_exactly(self):
        from hdpgpc.alignment import GaussianProcessWarp

        case = REFERENCES['scalar_grid']
        x = torch.tensor(case['x'], dtype=torch.float64)
        controls = torch.tensor(case['controls'], dtype=torch.float64, requires_grad=True)
        warper = GaussianProcessWarp(x, n_ctrl=4, recursive=False)
        expanded = warper._expand_ctrl_to_T(controls)
        grid, displacement = monotone_grid_from_expanded_controls(expanded, x)
        gradient, = torch.autograd.grad((grid * grid).sum(), (controls,))
        for actual, expected in ((grid, case['grid']), (displacement, case['displacement']),
                                 (gradient, case['gradient'])):
            torch.testing.assert_close(actual, x.new_tensor(expected), rtol=0, atol=0)
        # The legacy method must still provide coordinate-dependent expansion.
        wrapper_grid, wrapper_displacement = warper._monotonic_grid(controls)
        torch.testing.assert_close(wrapper_grid, grid, rtol=0, atol=0)
        torch.testing.assert_close(wrapper_displacement, displacement, rtol=0, atol=0)
        self.assertIsNone(warper._u_ctrl_prev)

    def test_batch_original_values_dtype_and_gradients_exactly(self):
        for dtype in (torch.float32, torch.float64):
            case = REFERENCES['batch_grid_' + str(dtype).split('.')[-1]]
            x = torch.tensor(case['x'], dtype=dtype)
            controls = torch.tensor(case['controls'], dtype=dtype, requires_grad=True)
            original = controls.detach().clone()
            grid, displacement = batch_monotone_grid(controls, x, len(x))
            gradient, = torch.autograd.grad((grid * grid).sum(), (controls,))
            for actual, expected in ((grid, case['grid']), (displacement, case['displacement']),
                                     (gradient, case['gradient'])):
                torch.testing.assert_close(actual, x.new_tensor(expected), rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, x.device)
            torch.testing.assert_close(controls, original, rtol=0, atol=0)
            self.assertTrue(torch.all(torch.diff(grid, dim=1) > 0))

    def test_zero_controls_on_irregular_grid_are_not_identity(self):
        x = torch.tensor([0., .25, 1., 2., 4.], dtype=torch.float64)
        grid, displacement = monotone_grid_from_expanded_controls(torch.zeros_like(x), x)
        torch.testing.assert_close(grid, x.new_tensor([0., 1., 2., 3., 4.]), rtol=0, atol=2e-12)
        self.assertFalse(torch.allclose(displacement, torch.zeros_like(x)))
        self.assertEqual(grid[0].item(), 0.)
        # Preserve the denominator epsilon, including the tiny endpoint deficit.
        self.assertLess(grid[-1].item(), 4.)

    def test_scalar_and_batch_interpolation_remain_distinct(self):
        from hdpgpc.alignment import GaussianProcessWarp

        case = REFERENCES['scalar_grid']
        x = torch.tensor(case['x'], dtype=torch.float64)
        controls = torch.tensor(case['controls'], dtype=torch.float64)
        scalar, _ = GaussianProcessWarp(x, n_ctrl=4)._monotonic_grid(controls)
        batch, _ = batch_monotone_grid(controls[None, :], x, len(x))
        self.assertFalse(torch.allclose(scalar, batch[0]))


if __name__ == "__main__":
    unittest.main()
