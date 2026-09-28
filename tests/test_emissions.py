"""Contracts for the extracted GP component and the old emission API."""

import base64
import copy
import json
import math
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from emission_cases import MODES, make_emission, reset, run_case, snapshot, np, torch
from hdpgpc.emissions import EmissionModel, GaussianProcessEmission
from hdpgpc.emissions import _gaussian_process as legacy


REFERENCE = json.loads((Path(__file__).parent / "fixtures/emission_reference.json").read_text())


class EmissionTests(unittest.TestCase):
    def setUp(self):
        reset()

    def compare(self, actual, expected, path="state"):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            if set(expected) == {"dtype", "shape", "values"}:
                self.assertEqual(actual["dtype"], expected["dtype"], path)
                self.assertEqual(actual["shape"], expected["shape"], path)
                # Same pinned CPU stack: retain characterization tolerances for
                # portability, with exact integer/discrete and dtype contracts.
                np.testing.assert_allclose(actual["values"], expected["values"],
                                           rtol=1e-7, atol=1e-9, err_msg=path)
            else:
                for key in expected:
                    self.compare(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.compare(a, e, f"{path}[{i}]")
        else:
            self.assertEqual(actual, expected, path)

    def test_real_training_predictions_states_and_optimizer_match_original(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model, state = run_case(mode, GaussianProcessEmission)
                self.assertIs(type(model), legacy.GaussianProcessEmission)
                self.compare(state, REFERENCE["cases"][mode], mode)

    def test_public_identity_and_structural_interface(self):
        self.assertIs(GaussianProcessEmission, GaussianProcessEmission)
        self.assertIs(GaussianProcessEmission, legacy.GaussianProcessEmission)
        model, _, _ = make_emission(cls=GaussianProcessEmission)
        self.assertIsInstance(model, EmissionModel)
        for cls in (GaussianProcessEmission, legacy.matrix_normal_inv_wishart, legacy.inv_wishart):
            self.assertTrue(cls.__module__.startswith(("hdpgpc.emissions.", "hdpgpc.dynamics.")))
            self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_native_instance_pickle_preserves_predictions(self):
        model, expected = run_case('exact', GaussianProcessEmission)
        _, x, y = make_emission()
        clone = pickle.loads(pickle.dumps(model))
        self.assertIs(type(clone), GaussianProcessEmission)
        self.compare(snapshot(clone, x, y, expected['optimizer']), expected)

    def test_prior_white_noise_and_static_dynamic_initialization(self):
        for mode in ("exact", "static"):
            with self.subTest(mode=mode):
                model, x, _ = make_emission(mode, GaussianProcessEmission)
                kernel = model.gp.kernel
                np.testing.assert_array_equal(model.K.numpy(), kernel(x, x))
                np.testing.assert_array_equal(model.gp.K_X_X.numpy(), kernel(x))
                self.assertFalse(torch.equal(model.K, model.gp.K_X_X))
                self.assertIs(model.cov_f[0], model.K)
                self.assertIs(model.cov_f_sm[0], model.K)
                self.assertEqual(model.N, 0)
                self.assertEqual(model.indexes, [])
                torch.testing.assert_close(model.f_star[0], torch.zeros(5, 1), rtol=0, atol=0)
                self.assertEqual(type(model.observation_params),
                                 legacy.inv_wishart if mode == "static" else legacy.matrix_normal_inv_wishart)

    def test_responsibility_and_snr_gates_preserve_history_aliases(self):
        model, x, y = make_emission(cls=GaussianProcessEmission)
        with patch.object(model, "fit_kernel_params", wraps=model.fit_kernel_params) as fit:
            model.include_weighted_sample(0, x, x, y, .999)
            fit.assert_not_called()
        self.assertEqual(model.N, 0)
        self.assertEqual(model.indexes, [])
        # Isolate posterior gating from training, which is covered by real fits.
        model.fitted = True
        previous = model.f_star_sm[-1]
        model.include_weighted_sample(1, x, x, y, 1., snr=.5)
        self.assertEqual(model.N, 0)
        self.assertEqual(model.indexes, [1])
        self.assertIs(model.f_star_sm[-1], previous)
        model.include_weighted_sample(2, x, x, y, 1., snr=.51)
        self.assertEqual(model.N, 1)
        self.assertEqual(model.indexes, [1, 2])
        self.assertIs(model.f_star[-1], model.f_star_sm[-1])
        self.assertIs(model.cov_f[-1], model.cov_f_sm[-1])
        clone = copy.deepcopy(model)
        self.assertIs(clone.cov_f[-1], clone.cov_f_sm[-1])
        self.assertIsNot(clone.cov_f[-1], model.cov_f[-1])

    def test_exact_grid_returns_covariance_by_identity_and_dtype(self):
        model, x, _ = make_emission(cls=GaussianProcessEmission)
        mean, covariance = model.observe_last(x)
        self.assertIs(covariance, model.Sigma[-1])
        self.assertEqual(mean.dtype, torch.float64)
        self.assertEqual(mean.device.type, "cpu")
        mean, covariance = model.resample_latent_mean(x)
        self.assertIs(mean, model.f_star_sm[-1])
        self.assertIs(covariance, model.cov_f_sm[-1])

    def test_score_omits_log_determinant(self):
        model, x, _ = make_emission(cls=GaussianProcessEmission)
        zero = torch.zeros_like(x)
        expected = -.5 * len(x) * math.log(2. * math.pi)
        for variance in (.04, .8):
            model.Sigma[-1] = variance * torch.eye(len(x))
            self.assertEqual(model.log_sq_error(x, zero, i=-1).item(), expected)

    def test_prediction_equal_diagonal_shortcut_ignores_off_diagonals(self):
        model, x, _ = make_emission(cls=GaussianProcessEmission)
        query = torch.linspace(0., 4., 7)[:, None]
        sigma = .03 * torch.ones(5, 5) + .01 * torch.eye(5)
        mean, cov = model.gp.pred_dist(query, x, torch.zeros_like(x), sigma)
        torch.testing.assert_close(cov, .04 * torch.eye(7), rtol=0, atol=0)
        torch.testing.assert_close(mean, torch.zeros_like(query), rtol=0, atol=0)

    def test_identity_cache_reuse_dtype_separation_and_batch_scores(self):
        model, x, y = make_emission(cls=GaussianProcessEmission)
        eye = model._eye_cached(5, x.device, x.dtype)
        self.assertIs(eye, model._eye_cached(5, x.device, x.dtype))
        eye32 = model._eye_cached(5, x.device, torch.float32)
        self.assertIsNot(eye, eye32)
        self.assertEqual(eye32.dtype, torch.float32)
        batch = torch.stack((y, y * .9))
        scores = model._gaussian_score_shared_cov(batch, model.f_star[0], model.Sigma[0])
        scalar = torch.stack([model.log_sq_error(x, observation, i=-1) for observation in batch])
        torch.testing.assert_close(scores, scalar, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
