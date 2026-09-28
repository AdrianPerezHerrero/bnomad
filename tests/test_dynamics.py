"""Pre-extraction regression values and focused static/LDS component contracts."""

import json
from pathlib import Path
import unittest

from dynamics_cases import CASES, TASKS, capture, core, setup
from emission_cases import reset, np, torch
from hdpgpc.dynamics import DynamicsModel, LinearDynamics, StaticDynamics


REFERENCE = json.loads((Path(__file__).parent / "fixtures/dynamics_reference.json").read_text())["cases"]


def assert_snapshot(test, actual, expected, path="state"):
    if isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected), path)
        if set(expected) == {"dtype", "shape", "values"}:
            test.assertEqual(actual["dtype"], expected["dtype"], path)
            test.assertEqual(actual["shape"], expected["shape"], path)
            np.testing.assert_allclose(actual["values"], expected["values"],
                                       rtol=1e-7, atol=1e-9, err_msg=path)
        else:
            for key in expected:
                assert_snapshot(test, actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected), path)
        for index, (a, e) in enumerate(zip(actual, expected)):
            assert_snapshot(test, a, e, f"{path}[{index}]")
    else:
        test.assertEqual(actual, expected, path)


class DynamicsRegressionTests(unittest.TestCase):
    """Actual legacy entry points must retain all recorded values and failures."""

    def test_reference_inventory_and_known_failures_are_explicit(self):
        self.assertEqual(set(CASES), set(REFERENCE))
        self.assertEqual({name for name, result in REFERENCE.items() if "exception_type" in result}, {
            "trajectory_static_full_smoother", "trajectory_static_full_data",
            "trajectory_static_reset_likelihood", "trajectory_linear_reset_likelihood",
        })


def regression(name):
    def test(self):
        assert_snapshot(self, capture(name), REFERENCE[name], name)
    test.__name__ = "test_" + name
    return test


for _name in CASES:
    setattr(DynamicsRegressionTests, "test_" + _name, regression(_name))


class DynamicsComponentTests(unittest.TestCase):
    def setUp(self):
        reset()

    def test_public_static_and_linear_operations_match_pre_extraction_reference(self):
        for mode, component in (("static", StaticDynamics()), ("linear", LinearDynamics())):
            self.assertIsInstance(component, DynamicsModel)
            self.assertEqual(vars(component), {})
            for task in TASKS:
                name = f"{mode}_{task}"
                with self.subTest(name=name):
                    assert_snapshot(self, core(mode, task, component), REFERENCE[name]["value"], name)

    def test_static_uses_real_filter_and_reduces_posterior_uncertainty(self):
        model, x, y, params, means, covars = setup("static")
        mean, cov = StaticDynamics().filter(model.gp, means[0], covars[0], y, *params, x_train=x)
        self.assertFalse(torch.equal(mean, means[0]))
        self.assertFalse(torch.equal(cov, covars[0]))
        self.assertTrue(torch.all(torch.diag(cov) < torch.diag(covars[0])))
        self.assertIs(StaticDynamics.filter, LinearDynamics.filter)
        # Static means no state evolution, not a frozen observation posterior.
        torch.testing.assert_close(params[0], torch.eye(len(x)), rtol=0, atol=0)
        torch.testing.assert_close(params[1], torch.zeros(len(x), len(x)), rtol=0, atol=0)

    def test_static_ml_sets_identity_and_zero_process_covariance(self):
        model, _, y, params, means, covars = setup("static")
        A, Gamma, C, Sigma = StaticDynamics().estimate_parameters(model.gp, *params, [y] * 3, means, covars)
        torch.testing.assert_close(A, torch.eye(len(A)), rtol=0, atol=0)
        torch.testing.assert_close(C, torch.eye(len(C)), rtol=0, atol=0)
        torch.testing.assert_close(Gamma, torch.zeros_like(Gamma), rtol=0, atol=0)
        self.assertFalse(torch.equal(Sigma, params[3]))

    def test_initializers_follow_existing_global_dtype_and_do_not_change_it(self):
        try:
            for dtype in (torch.float32, torch.float64):
                torch.set_default_dtype(dtype)
                x = torch.arange(3)[:, None]
                for component in (LinearDynamics(), StaticDynamics()):
                    params = component.initial_parameters(x)
                    self.assertTrue(all(value.dtype == dtype for value in params))
                    self.assertTrue(all(value.device.type == "cpu" for value in params))
                    self.assertEqual(torch.get_default_dtype(), dtype)
        finally:
            torch.set_default_dtype(torch.float64)

    def test_smoothing_retains_list_mutation_and_last_tensor_identity(self):
        for component in (LinearDynamics(), StaticDynamics()):
            model, _, _, params, means, covars = setup("linear")
            final_mean, final_cov = means[-1], covars[-1]
            original_first = means[0]
            original_values = original_first.clone()
            result = component.smooth(model.gp, [params[0]], [params[1]], means, covars)
            self.assertIs(result[0], means)
            self.assertIs(result[1], covars)
            self.assertIs(means[-1], final_mean)
            self.assertIs(covars[-1], final_cov)
            self.assertIsNot(means[0], original_first)
            torch.testing.assert_close(original_first, original_values, rtol=0, atol=0)

    def test_pair_smoothing_preserves_float32_and_autograd(self):
        model, _, _, params, means, covars = setup("linear")
        A, Gamma = (value.float() for value in params[:2])
        leaves = [value.float().requires_grad_() for value in means]
        covars = [value.float() for value in covars]
        result = LinearDynamics().smooth_pair(model.gp, A, Gamma, list(leaves), covars)
        self.assertTrue(all(value.dtype == torch.float32 for value in result[0] + result[1]))
        sum(value.sum() for value in result[0]).backward()
        self.assertTrue(all(value.grad is not None and torch.isfinite(value.grad).all() for value in leaves))


if __name__ == "__main__":
    unittest.main()
