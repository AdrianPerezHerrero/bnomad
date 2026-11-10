"""Screening validates unseen values, bounds work, and leaves inference untouched."""
from contextlib import redirect_stdout
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
import torch

from hdpgpc import HDPGPC
from hdpgpc.config import AlignmentConfig
from hdpgpc.dynamics import _clock
from hdpgpc.experiments import records
from hdpgpc.experiments._runs import experiment_settings
from hdpgpc.model_selection import NoisePriorScreenConfig, screen_noise_priors
from hdpgpc.model_selection._predictive import GaussianNoise, array, validate_predictive
from test_dynamics_clock import recipe


def factory(clock='physical-clock'):
    base = recipe(clock, fixed=False)
    def make(sigma, gamma):
        policy = replace(base.observation.Sigma_policy, value=sigma*np.diag([1., 2.]))
        return replace(base, observation=replace(base.observation, Sigma_policy=policy),
                       dynamics=replace(base.dynamics, ini_gamma=gamma))
    return make


def data(n=30):
    rng = np.random.default_rng(8)
    x = np.sin(np.arange(n)/4)
    return np.c_[x, .3*x]+rng.normal(scale=.025, size=(n, 2))


def screen(make=None, values=None, **options):
    defaults = dict(train_size=12, validation_size=4, blocks=2, max_fits=4, seconds=20., iterations=1)
    defaults.update(options)
    return screen_noise_priors(data() if values is None else values, factory() if make is None else make,
        grid=np.arange(2.), sigma_prior=.2, gamma_prior=.1, options=NoisePriorScreenConfig(**defaults))


class PriorScreenTests(unittest.TestCase):
    def test_plot_colors_stay_on_the_first_clip_after_confirmation(self):
        from hdpgpc.experiments._prior_screen import _plot_screen
        result = {'options': {'seed': 42}, 'clips': [{}, {}], 'fit_calls': 5, 'seconds': 1.,
            'working_pairs': [{'sigma_prior': .1, 'gamma_prior': 3.6}],
            'recommendation': {'candidate': 1, 'sigma_prior': .1, 'gamma_prior': 3.6,
                               'confirmation_complete': True},
            'candidates': [
                {'candidate': 0, 'sigma_prior': .001, 'gamma_prior': 10.,
                 'mean_nll_per_feature': 2.828, 'fits': [
                    {'block': 0, 'seed': 42, 'stage': 'coarse', 'status': 'ok', 'nll_per_feature': 2.828}]},
                {'candidate': 1, 'sigma_prior': .1, 'gamma_prior': 3.6,
                 'mean_nll_per_feature': 2.942, 'nll_standard_error': .071,
                 'seed_assignment_agreement_ari': 1., 'fits': [
                    {'block': block, 'seed': seed, 'stage': stage, 'status': 'ok', 'nll_per_feature': nll}
                    for block, seed, stage, nll in ((0, 42, 'refine', 2.818), (1, 42, 'confirm', 3.066),
                                                  (0, 43, 'confirm', 2.818), (1, 43, 'confirm', 3.066))]}]}
        captured = []
        def capture(fig, path, **kwargs):
            grid, details = fig.axes[:2]
            captured.append((np.asarray(grid.collections[0].get_array()).copy(),
                [cell.get_text().get_text() for table in details.tables for cell in table.get_celld().values()]))
        with patch('matplotlib.figure.Figure.savefig', new=capture):
            _plot_screen(result, 'unused.png')
            # Extra validation changes the confirmation average and table only.
            result['candidates'][1]['fits'][1]['nll_per_feature'] = 4.066
            result['candidates'][1]['mean_nll_per_feature'] = 3.192
            _plot_screen(result, 'unused.png')
        np.testing.assert_array_equal(captured[0][0], [2.828, 2.818])
        np.testing.assert_array_equal(captured[1][0], captured[0][0])
        self.assertIn('2.9420 ± 0.0710', captured[0][1])
        self.assertIn('3.1920 ± 0.0710', captured[1][1])
        self.assertIn('4/4', captured[0][1])

    def test_plot_handles_unconfirmed_and_missing_first_clip_scores(self):
        from hdpgpc.experiments._prior_screen import _plot_screen
        result = {'options': {'seed': 42}, 'clips': [{}], 'fit_calls': 1, 'seconds': .1,
            'working_pairs': [], 'recommendation': None,
            'candidates': [{'candidate': 0, 'sigma_prior': .1, 'gamma_prior': .1,
                # A score on another seed must never substitute for the common first stage.
                'mean_nll_per_feature': 2., 'fits': [
                    {'block': 0, 'seed': 43, 'stage': 'coarse', 'status': 'ok', 'nll_per_feature': 2.}]}]}
        captured = []
        def capture(fig, path, **kwargs):
            grid, details = fig.axes[:2]
            captured.append(([collection.get_array() for collection in grid.collections],
                             '\n'.join(text.get_text() for text in details.texts)))
        with patch('matplotlib.figure.Figure.savefig', new=capture):
            _plot_screen(result, 'unused.png')
        self.assertTrue(all(array is None for array in captured[0][0]))
        self.assertIn('No successful confirmation', captured[0][1])
        self.assertIn('No valid recommendation', captured[0][1])

    def test_affine_density_matches_dense_gaussian_for_full_and_diagonal_noise(self):
        rng = np.random.default_rng(23)
        C = rng.normal(size=(17, 3)); U = rng.normal(size=(3, 3))
        P = U @ U.T+.1*np.eye(3); y = rng.normal(size=17)
        for diagonal in (True, False):
            V = rng.normal(size=(17, 17))
            R = np.diag(rng.uniform(.1, .5, size=17)) if diagonal else V @ V.T+np.eye(17)
            actual = GaussianNoise(R, C).score(y, P)
            expected = multivariate_normal.logpdf(y, cov=R+C @ P @ C.T)
            self.assertAlmostEqual(actual, expected, places=10)

    def test_large_diagonal_emissions_only_factor_latent_matrices(self):
        rng = np.random.default_rng(4)
        noise = GaussianNoise(np.eye(512)*1e-5, rng.normal(size=(512, 4)))
        from hdpgpc.model_selection import _predictive
        operation = _predictive.cholesky
        with patch.object(_predictive, 'cholesky', wraps=operation) as chol:
            value = noise.score(rng.normal(size=512), np.eye(4))
        self.assertTrue(np.isfinite(value))
        self.assertTrue(all(call.args[0].shape == (4, 4) for call in chol.call_args_list))

    def test_first_validation_density_is_pre_update_and_future_values_do_not_leak(self):
        for clock in ('cluster-clock', 'physical-clock'):
            with self.subTest(clock=clock), redirect_stdout(io.StringIO()):
                cfg = factory(clock)(.2, .1)
                model = HDPGPC(cfg).fit(data(12), grid=np.arange(2.))
                untouched = model.labels_.copy()
                states = {int(k): [(np.array(s.filtered_mean), np.array(s.filtered_covariance))
                                  for s in model.posterior(int(k))] for k in model.cluster_ids_}
                y = np.array([[[2.], [.6]], [[.5], [.2]]])
                ids = model.cluster_ids_.astype(int)
                transition = array(model._backend.transTheta)[np.ix_(ids, ids)]
                transition /= transition.sum(1, keepdims=True)
                weights = transition[list(ids).index(int(untouched[-1]))]
                expected = []
                for k in ids:
                    gp = model._backend.gpmodels[0][int(k)]
                    m, P = states[int(k)][0]
                    A, Q, C, R = map(array, (gp.A[-1], gp.Gamma[-1], gp.C[-1], gp.Sigma[-1]))
                    mean = C @ (A @ m)
                    covariance = C @ (A @ P @ A.T+Q) @ C.T+R
                    expected.append(multivariate_normal.logpdf(y[0, :, 0], mean=mean, cov=covariance))
                expected = -logsumexp(np.log(weights)+expected)/2
                actual = validate_predictive(model, y, np.arange(2.)[:, None])
                self.assertAlmostEqual(actual['losses'][0], expected, places=9)
                y[1] = 100.
                self.assertEqual(validate_predictive(model, y, np.arange(2.)[:, None])['losses'][0], actual['losses'][0])
                np.testing.assert_array_equal(model.labels_, untouched)
                for k in ids:
                    for (mean, covariance), state in zip(states[int(k)], model.posterior(int(k))):
                        np.testing.assert_array_equal(mean, state.filtered_mean)
                        np.testing.assert_array_equal(covariance, state.filtered_covariance)

    def test_budget_is_bounded_and_training_clips_are_contiguous(self):
        values = data(60)
        original_fit = HDPGPC.fit
        seen = []
        def fit(model, X, **kwargs):
            seen.append(np.array(X))
            return original_fit(model, X, **kwargs)
        with patch.object(HDPGPC, 'fit', fit):
            result = screen(values=values)
        self.assertEqual(result['fit_calls'], 4)
        self.assertTrue(result['budget_exhausted'])
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(len(x) == 12 for x in seen))
        for X in seen:
            self.assertTrue(any(np.array_equal(X[..., 0], values[start:start+12]) for start in range(49)))
        self.assertIsNotNone(result['recommendation'])
        self.assertFalse(any('purity' in item for item in result['candidates']))

    def test_explicit_ranges_are_respected_during_refinement_and_confirmation(self):
        result = screen(max_fits=24, sigma_range=(.01, .3), gamma_range=(.001, .2))
        self.assertLessEqual(result['fit_calls'], 24)
        for item in result['candidates']:
            self.assertTrue(.01 <= item['sigma_prior'] <= .3)
            self.assertTrue(.001 <= item['gamma_prior'] <= .2)
        self.assertEqual(result['recommendation']['confirmation'], 'promoted')
        self.assertTrue(any(item['seed_assignment_agreement_ari'] is not None for item in result['candidates']))
        self.assertTrue(any(f.get('gap_diagnostics') for item in result['candidates'] for f in item['fits']))
        json.dumps(result, allow_nan=False)

    def test_default_bounds_constrain_refinement_and_anchor(self):
        for supplied in ((.2, .1), (20., 30.)):
            with self.subTest(supplied=supplied):
                def validation(model, *args):
                    # Favor the upper corner to exercise clipping during refinement.
                    sigma = model.config.observation.Sigma_policy.value[0, 0]
                    gamma = model.config.dynamics.ini_gamma
                    loss = -float(np.log(sigma)+np.log(gamma))
                    return {'nll_per_feature': loss, 'losses': [loss]}
                with patch('hdpgpc.model_selection._noise_screen._calibrate',
                           return_value=(.01, .001, {'active_gamma': True})), \
                     patch('hdpgpc.model_selection._noise_screen.validate_predictive', side_effect=validation):
                    result = screen_noise_priors(data(), factory(), grid=np.arange(2.),
                        sigma_prior=supplied[0], gamma_prior=supplied[1],
                        options=NoisePriorScreenConfig(train_size=12, validation_size=4,
                            grid_points=3, max_fits=24, iterations=1, seconds=30.))
                self.assertEqual(result['initial_ranges']['sigma'], [.001, 10.])
                self.assertEqual(result['initial_ranges']['gamma'], [.001, 10.])
                for name in ('sigma_prior', 'gamma_prior'):
                    self.assertEqual(max(item[name] for item in result['candidates']), 10.)
                    self.assertEqual(min(item[name] for item in result['candidates']), .001)
                self.assertTrue(any(f['stage'] == 'refine'
                                    for item in result['candidates'] for f in item['fits']))

    def test_explicit_ranges_override_default_bounds(self):
        result = screen(max_fits=24, sigma_range=(.1, 20.), gamma_range=(.01, 30.))
        self.assertEqual(result['initial_ranges'], {'sigma': [.1, 20.], 'gamma': [.01, 30.]})
        self.assertTrue(any(item['sigma_prior'] == 20. and item['fits'] for item in result['candidates']))
        self.assertTrue(any(item['gamma_prior'] == 30. and item['fits'] for item in result['candidates']))
        self.assertTrue(all(.1 <= item['sigma_prior'] <= 20. and .01 <= item['gamma_prior'] <= 30.
                            for item in result['candidates']))

    def test_dense_grid_override_spans_both_axes_and_reserves_confirmation_work(self):
        result = screen(grid_points=10, max_fits=100, seconds=30.)
        self.assertLessEqual(result['fit_calls'], 100)
        self.assertLessEqual(len(result['candidates']), 100)
        self.assertGreaterEqual(result['fit_calls'], 90)
        self.assertEqual(result['grid']['combinations'], 100)
        for name in ('sigma', 'gamma'):
            axis = np.asarray(result['grid'][f'{name}_values'])
            self.assertEqual(len(axis), 10)
            self.assertEqual((axis[0], axis[-1]), (.001, 10.))
            np.testing.assert_allclose(np.diff(np.log10(axis)), 4/9)
        pairs = {(item['sigma_prior'], item['gamma_prior']) for item in result['candidates'] if item['fits']}
        self.assertTrue({(.001, .001), (.001, 10.), (10., .001), (10., 10.)}.issubset(pairs))
        self.assertTrue(result['recommendation']['confirmation_complete'])
        self.assertTrue(any(f['stage'] == 'refine' for item in result['candidates'] for f in item['fits']))
        json.dumps(result, allow_nan=False)

    def test_default_sampling_spans_the_record_and_uses_a_coarser_grid(self):
        from hdpgpc.model_selection._noise_screen import _clips
        options = NoisePriorScreenConfig()
        self.assertEqual((options.train_size, options.validation_size, options.blocks), (128, 32, 5))
        self.assertEqual((options.grid_points, options.max_fits, options.seconds), (5, 60, 120.))
        clips = _clips(2187, options)
        self.assertEqual(len(clips), 5)
        self.assertEqual(clips[0][0], 0)
        self.assertEqual(clips[-1][-1], 2187)
        covered = set()
        for start, cut, end in clips:
            self.assertEqual(cut-start, 128)
            self.assertEqual(end-cut, 32)
            covered.update(range(start, end))
        self.assertEqual(len(covered), 800)
        # Use smaller local clips to verify work allocation without making this
        # regression test depend on the runtime of a long native fit.
        result = screen(values=data(100), blocks=options.blocks,
                        grid_points=options.grid_points, max_fits=options.max_fits, seconds=30.)
        self.assertEqual(len(result['clips']), 5)
        self.assertEqual(result['grid']['combinations'], 25)
        self.assertLessEqual(result['fit_calls'], 60)
        self.assertTrue(result['recommendation']['confirmation_complete'])
        self.assertEqual(result['recommendation']['evaluated_blocks'], list(range(5)))

    def test_screen_restores_rngs_and_numeric_settings(self):
        random.seed(111); np.random.seed(222); torch.manual_seed(333)
        py, np_state, state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        threads, dtype = torch.get_num_threads(), torch.get_default_dtype()
        deterministic = torch.are_deterministic_algorithms_enabled()
        first = screen(max_fits=2)
        self.assertEqual(random.getstate(), py)
        self.assertEqual(np.random.get_state()[0], np_state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(torch.get_default_dtype(), dtype)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        second = screen(max_fits=2)
        self.assertEqual(first['recommendation'], second['recommendation'])
        self.assertEqual(first['working_pairs'], second['working_pairs'])

    def test_changing_the_frozen_decoder_is_rejected(self):
        make = factory()
        def changing(sigma, gamma):
            cfg = make(sigma, gamma)
            if sigma != .2:
                policy = replace(cfg.observation.C_policy, value=cfg.observation.C_policy.value*2)
                cfg = replace(cfg, observation=replace(cfg.observation, C_policy=policy))
            return cfg
        result = screen(changing, max_fits=4)
        failures = [f for item in result['candidates'] for f in item['fits'] if f['status'] == 'failed']
        self.assertTrue(failures)
        self.assertTrue(all('scientific settings' in f['error'] for f in failures))

    def test_validation_rejects_bad_budgets_inputs_and_warp(self):
        for kwargs in ({'seconds': 0}, {'max_fits': 0}, {'train_size': 5},
                       {'grid_points': 1}, {'grid_points': True}, {'sigma_range': (-1, 2)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                NoisePriorScreenConfig(**kwargs)
        with self.assertRaisesRegex(ValueError, 'eight observations'):
            screen(values=data(7))
        make = factory()
        def warped(sigma, gamma):
            cfg = make(sigma, gamma)
            # The entry point rejects the density interpretation before fitting.
            return replace(cfg, alignment=AlignmentConfig(policy='dtw_gp'))
        with self.assertRaises(ValueError):
            screen(warped)

    def test_disabled_cli_settings_keep_the_previous_identity(self):
        args = records.build_parser('mitbih', 'pca').parse_args([])
        settings = experiment_settings(args)
        self.assertFalse(any(key.startswith('screen_') for key in settings))
        args.screen_priors = True
        self.assertTrue(experiment_settings(args)['screen_priors'])
        from hdpgpc.experiments._prior_screen import options_from_args
        options = options_from_args(args)
        self.assertEqual((options.train_size, options.validation_size, options.blocks), (128, 32, 5))
        self.assertEqual((options.grid_points, options.max_fits, options.seconds), (5, 60, 120.))

    def test_cli_is_screen_only_calibrates_once_exports_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = np.linspace(0, 2*np.pi, 12)
            waves = np.sin(x)[None, :, None]*data(24)[:, :1, None]
            np.save(root/'203.npy', waves)
            np.save(root/'203_labels.npy', np.repeat(['N', 'V'], 12))
            args = records.build_parser('mitbih', 'pca').parse_args([
                '--data-dir', str(root), '--records', '203', '--output-dir', str(root/'runs'),
                '--screen-priors', '--screen-samples', '12', '--screen-max-fits', '3',
                '--screen-iterations', '1', '--components', '2', '--calibration-samples', '20',
                '--no-plots', '--verbose', 'silent'])
            with patch.object(records, 'estimate_pca_observation', wraps=records.estimate_pca_observation) as calibration, redirect_stdout(io.StringIO()):
                report, = records.run_records(args)
            self.assertEqual(calibration.call_count, 1)
            self.assertEqual(report['mode'], 'prior_screen')
            self.assertEqual(report['status'], 'ok', report)
            self.assertEqual(report['fitted'], 0)
            out = Path(report['output_directory'])
            self.assertTrue((out/'prior_screen.json').is_file())
            self.assertFalse((out/'assignments.csv').exists())
            args.resume = True
            with patch.object(HDPGPC, 'fit', side_effect=AssertionError('Refit during resume')), redirect_stdout(io.StringIO()):
                resumed, = records.run_records(args)
            self.assertEqual(resumed['recommendation'], report['recommendation'])


if __name__ == '__main__':
    unittest.main()
