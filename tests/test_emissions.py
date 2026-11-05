"""Contracts for the extracted GP component and the old emission API."""

import base64
import copy
import json
import math
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

from emission_cases import MODES, make_emission, reset, run_case, snapshot, encode, np, torch
from hdpgpc.emissions import EmissionModel, GaussianProcessEmission
from hdpgpc.emissions import _gaussian_process as legacy


REFERENCE = json.loads((Path(__file__).parent / "fixtures/emission_reference.json").read_text())


class EmissionTests(unittest.TestCase):
    def setUp(self):
        reset()

    def compare(self, actual, expected, path="state", *, fitted_noise_delta=0.):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected), path)
            if path == 'reduced':
                fitted_noise_delta = (math.exp(actual['kernel_theta']['values'][-1])
                                      - math.exp(expected['kernel_theta']['values'][-1]))
            if set(expected) == {"dtype", "shape", "values"}:
                self.assertEqual(actual["dtype"], expected["dtype"], path)
                self.assertEqual(actual["shape"], expected["shape"], path)
                # Adam's final log-kernel parameters vary slightly across CPU
                # builds. An absolute 1e-6 log-space bound permits about 1 ppm kernel
                # parameter variation. Its derived reduced-grid alpha uses the
                # same relative noise bound. Predictions, posteriors, shapes,
                # dtype, optimizer settings and counts retain their checks.
                optimized_kernel = path.endswith('.kernel_theta')
                optimized_noise = path == 'reduced.alpha_ini'
                values = actual['values']
                if path == 'reduced.query[1]':
                    # pred_dist adds WhiteKernel noise exactly once to the
                    # query diagonal. Compare at the reference fitted noise;
                    # keep all remaining covariance entries at original bounds.
                    values = np.asarray(values) - fitted_noise_delta*np.eye(actual['shape'][0])
                np.testing.assert_allclose(values, expected["values"],
                    rtol=0 if optimized_kernel else (np.expm1(1e-6) if optimized_noise else 1e-7),
                    atol=1e-6 if optimized_kernel else (0. if optimized_noise else 1e-9), err_msg=path)
            else:
                for key in expected:
                    self.compare(actual[key], expected[key], f"{path}.{key}",
                                 fitted_noise_delta=fitted_noise_delta)
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected), path)
            for i, (a, e) in enumerate(zip(actual, expected)):
                self.compare(a, e, f"{path}[{i}]", fitted_noise_delta=fitted_noise_delta)
        else:
            self.assertEqual(actual, expected, path)

    def compare_fitted_snapshot(self, model, state, expected, mode):
        comparable = state
        if mode == 'reduced':
            # A log density also depends on the fitted query-noise diagonal.
            # Compare it at common noise, keeping the actual learned posterior.
            # Verify the original captured score before re-evaluating it.
            _, x, y = make_emission(mode)
            self.assertEqual(state['score'], encode(model.log_sq_error(x, y, i=-1)))
            noise = model.gp.kernel.k2.noise_level
            comparable = dict(state)
            try:
                model.gp.kernel.k2.noise_level = math.exp(expected['kernel_theta']['values'][-1])
                comparable['score'] = encode(model.log_sq_error(x, y, i=-1))
            finally:
                model.gp.kernel.k2.noise_level = noise
        self.compare(comparable, expected, mode)

    def test_real_training_predictions_states_and_optimizer_match_original(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                model, state = run_case(mode, GaussianProcessEmission)
                self.assertIs(type(model), legacy.GaussianProcessEmission)
                self.compare_fitted_snapshot(model, state, REFERENCE["cases"][mode], mode)
                if mode == 'reduced':
                    # alpha is the fitted kernel noise, not a free tolerance.
                    np.testing.assert_array_equal(state['alpha_ini']['values'],
                        model.gp.kernel.k2.noise_level * np.eye(len(model.x_basis)))

    def test_public_identity_and_structural_interface(self):
        self.assertIs(GaussianProcessEmission, GaussianProcessEmission)
        self.assertIs(GaussianProcessEmission, legacy.GaussianProcessEmission)
        model, _, _ = make_emission(cls=GaussianProcessEmission)
        self.assertIsInstance(model, EmissionModel)
        for cls in (GaussianProcessEmission, legacy.matrix_normal_inv_wishart, legacy.inv_wishart):
            self.assertTrue(cls.__module__.startswith(("hdpgpc.emissions.", "hdpgpc.dynamics.")))
            self.assertIs(pickle.loads(pickle.dumps(cls)), cls)

    def test_optimizer_roundoff_does_not_relax_scientific_output_checks(self):
        expected = REFERENCE['cases']['reduced']
        actual = copy.deepcopy(expected)
        actual['kernel_theta']['values'][2] += 4e-7
        noise = math.exp(actual['kernel_theta']['values'][2])
        actual['alpha_ini']['values'] = (noise * np.eye(len(actual['alpha_ini']['values']))).tolist()
        covariance = np.asarray(actual['query'][1]['values'])
        delta = noise - math.exp(expected['kernel_theta']['values'][2])
        actual['query'][1]['values'] = (covariance + delta*np.eye(len(covariance))).tolist()
        self.compare(actual, expected, 'reduced')
        actual['kernel_theta']['values'][2] += 1e-3
        with self.assertRaises(AssertionError):
            self.compare(actual, expected, 'reduced')
        actual = copy.deepcopy(expected)
        actual['query'][1]['values'][0][0] += 1e-5
        with self.assertRaises(AssertionError):
            self.compare(actual, expected, 'reduced')
        actual = copy.deepcopy(expected)
        actual['query'][1]['values'][0][1] += 1e-5
        with self.assertRaises(AssertionError):
            self.compare(actual, expected, 'reduced')
        actual = copy.deepcopy(expected)
        actual['alpha_ini']['values'][0][0] += 1e-5
        with self.assertRaises(AssertionError):
            self.compare(actual, expected, 'reduced')
        actual = copy.deepcopy(expected)
        actual['alpha_ini']['values'][0][1] = 1e-12
        with self.assertRaises(AssertionError):
            self.compare(actual, expected, 'reduced')
        actual = copy.deepcopy(expected)
        actual['K_inv']['values'][0][0] += .001
        with self.assertRaises(AssertionError):
            self.compare(actual, expected, 'reduced')

    def test_native_instance_pickle_preserves_predictions(self):
        model, expected = run_case('exact', GaussianProcessEmission)
        _, x, y = make_emission()
        clone = pickle.loads(pickle.dumps(model))
        self.assertIs(type(clone), GaussianProcessEmission)
        self.compare(snapshot(clone, x, y, expected['optimizer']), expected)

    def test_reduced_snapshot_handles_consistent_fitted_noise_roundoff(self):
        from hdpgpc.emissions._prior import GaussianProcessPrior
        original = GaussianProcessPrior.fit_torch

        def shifted_noise(prior, *args, **kwargs):
            result = original(prior, *args, **kwargs)
            # Match the hosted fitted-noise drift after actual optimization.
            prior.kernel.k2.noise_level *= math.exp(3.71128256e-7)
            identity = torch.eye(len(prior.x_basis), dtype=prior.alpha_ini.dtype,
                                 device=prior.alpha_ini.device)
            prior.assign_alpha_ini(prior.kernel.k2.noise_level*identity, prior.gamma_ini)
            return result

        with patch.object(GaussianProcessPrior, 'fit_torch', shifted_noise):
            model, state = run_case('reduced', GaussianProcessEmission)
        self.compare_fitted_snapshot(model, state, REFERENCE['cases']['reduced'], 'reduced')
        corrupted = copy.deepcopy(state)
        corrupted['score']['values'] += 1e-5
        with self.assertRaises(AssertionError):
            self.compare_fitted_snapshot(model, corrupted, REFERENCE['cases']['reduced'], 'reduced')

    def test_reduced_query_covariance_adds_fitted_white_noise_on_the_diagonal(self):
        model, _ = run_case('reduced', GaussianProcessEmission)
        _, x, y = make_emission('reduced')
        query = torch.linspace(0., 4., 7)[:, None]
        mean, covariance = model.observe_last(query)
        noise = model.gp.kernel.k2.noise_level
        try:
            model.gp.kernel.k2.noise_level = noise*1.01
            changed_mean, changed_covariance = model.observe_last(query)
            torch.testing.assert_close(changed_mean, mean, rtol=0, atol=0)
            torch.testing.assert_close(changed_covariance-covariance,
                (model.gp.kernel.k2.noise_level-noise)*torch.eye(len(query)), rtol=0, atol=1e-12)
            # Check the raw score with independent NumPy Gaussian algebra at
            # both fitted-noise values, including latent uncertainty/jitter.
            for variance in (noise, noise*1.01):
                model.gp.kernel.k2.noise_level = variance
                m, R = model.observe(x, -1, final_lds=True, smoothed=True)
                basis = model.x_basis.numpy()
                K = model.gp.kernel(basis, basis)
                K += 1e-8*np.mean(np.abs(np.diag(K)))*np.eye(len(K))
                H = np.linalg.solve(K, model.gp.kernel(basis, x.numpy())).T
                C, P = model.C[-1].numpy(), model.cov_f_sm[-1].numpy()
                latent = H @ C @ P @ C.T @ H.T
                R = R.numpy()
                R = .5*(R+R.T) + 1e-8*np.mean(np.abs(np.diag(R)))*np.eye(len(R))
                residual = y.numpy()-m.numpy()
                expected = -.5*(float((residual.T @ np.linalg.solve(R, residual)).item())
                    + np.trace(np.linalg.solve(R, latent)) + np.linalg.slogdet(R)[1]
                    + len(R)*np.log(2*np.pi))
                self.assertAlmostEqual(model.log_sq_error(x, y, i=-1).item(), expected, places=10)
        finally:
            model.gp.kernel.k2.noise_level = noise

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
            model.include_weighted_sample(0, x, x, y, 0.)
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

    def test_fractional_responsibility_trains_with_less_precision(self):
        model, x, y = make_emission(cls=GaussianProcessEmission)
        model.fitted = True
        full = copy.deepcopy(model)
        model.include_weighted_sample(0, x, x, y, .25)
        full.include_weighted_sample(0, x, x, y, 1.)
        self.assertEqual(model.N, 1)
        self.assertEqual(model.indexes, [0])
        self.assertGreater(torch.trace(model.cov_f[-1]).item(),
                           torch.trace(full.cov_f[-1]).item())

    def test_exact_grid_returns_covariance_by_identity_and_dtype(self):
        model, x, _ = make_emission(cls=GaussianProcessEmission)
        mean, covariance = model.observe_last(x)
        self.assertIs(covariance, model.Sigma[-1])
        self.assertEqual(mean.dtype, torch.float64)
        self.assertEqual(mean.device.type, "cpu")
        mean, covariance = model.resample_latent_mean(x)
        self.assertIs(mean, model.f_star_sm[-1])
        self.assertIs(covariance, model.cov_f_sm[-1])

    def test_score_includes_log_determinant_and_latent_uncertainty(self):
        model, x, _ = make_emission(cls=GaussianProcessEmission)
        zero = torch.zeros_like(x)
        latent = model.C[-1] @ model.cov_f_sm[0] @ model.C[-1].T
        for variance in (.04, .8):
            model.Sigma[-1] = variance * torch.eye(len(x))
            # Analytic E_q[log N(0 | C f, variance I)], including Cholesky jitter.
            effective_variance = variance * (1. + 1e-8)
            expected = -.5 * (len(x) * math.log(2. * math.pi * effective_variance)
                              + torch.trace(latent).item() / effective_variance)
            self.assertAlmostEqual(model.log_sq_error(x, zero, i=-1).item(), expected, places=10)

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
        latent = model.C[0] @ model.cov_f_sm[0] @ model.C[0].T
        scores = model._gaussian_score_shared_cov(batch, model.f_star[0], model.Sigma[0], latent)
        scalar = torch.stack([model.log_sq_error(x, observation, i=-1) for observation in batch])
        torch.testing.assert_close(scores, scalar, rtol=1e-14, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
