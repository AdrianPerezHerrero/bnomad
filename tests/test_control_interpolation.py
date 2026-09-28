"""Exact control values/gradients and per-call/fallback semantics for P01."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import os
if not os.environ.get('HDPGPC_TEST_INSTALLED'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hdpgpc'))
import torch
from hdpgpc.alignment._amtgp import GaussianProcessWarp, _lin_interp_1d
from hdpgpc.alignment._control_interpolation import prepare_control_interpolation, expand_controls


class ControlInterpolationTests(unittest.TestCase):
    def check_grid(self, dtype, irregular=False, device='cpu'):
        x = torch.linspace(0., 4., 17, dtype=dtype, device=device)
        if irregular:
            x = x ** 1.3
        first = torch.linspace(-1., 1.5, 7, dtype=dtype, device=device, requires_grad=True)
        second = first.detach().clone().requires_grad_(True)
        x_ctrl = torch.linspace(float(x[0]), float(x[-1]), steps=7, device=device, dtype=dtype)
        original = _lin_interp_1d(x_ctrl, first[:, None], x)[:, 0]
        prepared = prepare_control_interpolation(x, 7, device, dtype)
        actual = expand_controls(second, prepared)
        self.assertTrue(torch.equal(original, actual))
        weights = torch.linspace(.3, 1.7, 17, dtype=dtype, device=device)
        (original * weights).sum().backward()
        (actual * weights).sum().backward()
        self.assertTrue(torch.equal(first.grad, second.grad))
        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(actual.device, x.device)
        self.assertTrue(all(not item.requires_grad for item in prepared))

    def test_exact_uniform_and_irregular_values_and_gradients(self):
        for dtype in (torch.float32, torch.float64):
            for irregular in (False, True):
                with self.subTest(dtype=dtype, irregular=irregular):
                    self.check_grid(dtype, irregular)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_values_and_gradients(self):
        self.check_grid(torch.float64, irregular=True, device='cuda')

    def test_one_preparation_per_call_and_changed_grid(self):
        x = torch.linspace(0., 4., 9, dtype=torch.float64)
        warper = GaussianProcessWarp(x, n_ctrl=4, recursive=True)
        with patch('hdpgpc.alignment._amtgp.prepare_control_interpolation', wraps=prepare_control_interpolation) as prepare:
            for grid in (x, x, torch.linspace(0., 4., 11)):
                y = torch.sin(grid)
                warper.compute_warp(grid, y, y + .01, train_iter=3)
            self.assertEqual(prepare.call_count, 3)
            self.assertEqual(prepare.call_args.args[0].numel(), 11)
        self.assertFalse(any('interpolation' in name for name in vars(warper)))

    def test_zero_steps_and_coordinate_gradient_use_original_path(self):
        for needs_grad, steps in ((False, 0), (True, 1)):
            x = torch.linspace(0., 4., 9, requires_grad=needs_grad)
            warper = GaussianProcessWarp(x, n_ctrl=4)
            y = torch.sin(x.detach())
            with patch('hdpgpc.alignment._amtgp.prepare_control_interpolation', wraps=prepare_control_interpolation) as prepare:
                warper.compute_warp(x, y, y + .01, train_iter=steps)
                prepare.assert_not_called()

    def test_custom_control_expansion_is_honored(self):
        class CustomWarp(GaussianProcessWarp):
            calls = 0

            def _expand_ctrl_to_T(self, controls):
                self.calls += 1
                return super()._expand_ctrl_to_T(controls)

        x = torch.linspace(0., 4., 9)
        warper = CustomWarp(x, n_ctrl=4)
        with patch('hdpgpc.alignment._amtgp.prepare_control_interpolation', wraps=prepare_control_interpolation) as prepare:
            warper.compute_warp(x, torch.sin(x), torch.cos(x), train_iter=3)
            prepare.assert_not_called()
        self.assertEqual(warper.calls, 4)


if __name__ == '__main__':
    unittest.main()
