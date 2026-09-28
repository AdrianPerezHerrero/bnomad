"""End-to-end research usage: only public hdpgpc names and NumPy inputs.

No backend fields, private helpers, monkeypatched numerical routines or test-only
model factories are used in these usage tests.
"""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import sys
import unittest

import os
if not os.environ.get('HDPGPC_TEST_INSTALLED'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hdpgpc'))
import numpy as np
from hdpgpc import HDPGPC, HDPGPCConfig, UpdateResult, Prediction, LatentState
from hdpgpc.config import InferenceConfig

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples/configurations'


def config(name='dynamic'):
    return HDPGPCConfig.from_yaml((EXAMPLES / (name + '.yaml')).read_text())


def waves(outputs=1):
    a = np.array([0., .4, 1., .4, 0.])
    b = np.array([0., -.6, -1.2, -.6, 0.])
    data = np.stack([a, a + .01, b, b - .01, a - .015, b + .015])
    return data if outputs == 1 else np.stack([data, .7 * data[:, ::-1] + .02], axis=-1)


class PublicUsageTests(unittest.TestCase):
    def setUp(self):
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)

    def test_minimal_static_fit_and_labels(self):
        model = HDPGPC(config('static'))
        self.assertIs(model.fit(waves()[:4]), model)
        np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1])
        np.testing.assert_array_equal(model.cluster_ids_, [0, 1])
        self.assertEqual(model.n_clusters_, 2)
        self.assertFalse(model.labels_.flags.writeable)

    def test_static_streaming_birth_reuse_and_optional_results(self):
        cfg = config('static')
        cfg = replace(cfg, inference=InferenceConfig(schedule='online', with_warp=False, n_explore_steps=2))
        model = HDPGPC(cfg)
        results = [model.update(sample, include_prediction=True, include_posterior=True) for sample in waves()[:4]]
        self.assertEqual([r.assigned_cluster for r in results], [0, 0, 1, 1])
        self.assertEqual([r.new_cluster for r in results], [False, False, True, False])
        self.assertEqual([r.observation_index for r in results], [0, 1, 2, 3])
        for r in results:
            self.assertIsInstance(r, UpdateResult)
            self.assertIsNone(r.assignment_probabilities)
            self.assertEqual(r.responsibilities.sum(), 1.)
            self.assertEqual(r.cluster_ids[np.argmax(r.responsibilities)], r.assigned_cluster)
            self.assertIsInstance(r.prediction, Prediction)
            self.assertEqual(r.prediction.mean.shape, (5, 1))
            self.assertEqual(r.prediction.covariance.shape, (1, 5, 5))
            self.assertIsInstance(r.posterior[0], LatentState)
            self.assertEqual(r.posterior[0].filtered_mean.shape, (5,))
            self.assertFalse(r.prediction.mean.flags.writeable)
        np.testing.assert_array_equal(results[0].responsibilities, [1.])
        np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1])

    def test_dynamic_fit_and_interpolated_prediction(self):
        model = HDPGPC(config()).fit(waves())
        cluster = int(model.labels_[0])
        prediction = model.predictive(cluster, grid=np.linspace(0, 4, 7))
        self.assertEqual(prediction.grid.shape, (7, 1))
        self.assertEqual(prediction.mean.shape, (7, 1))
        self.assertEqual(prediction.covariance.shape, (1, 7, 7))
        self.assertTrue(np.isfinite(prediction.mean).all())
        self.assertTrue(np.isfinite(prediction.covariance).all())
        posterior = model.posterior(cluster)
        self.assertEqual(len(posterior), 1)
        self.assertEqual(posterior[0].smoothed_covariance.shape, (5, 5))

    def test_online_fit_can_continue_streaming_and_refit_resets(self):
        model = HDPGPC(config('online')).fit(waves()[:2])
        saved = model.labels_
        result = model.update(waves()[2])
        self.assertEqual(result.observation_index, 2)
        self.assertIsNone(result.prediction)
        self.assertIsNone(result.posterior)
        self.assertEqual(len(model.labels_), 3)
        self.assertEqual(len(saved), 2)
        model.fit(waves()[:1])
        self.assertEqual(len(model.labels_), 1)
        self.assertEqual(model.update(waves()[1]).observation_index, 1)

    def test_fast_online(self):
        cfg = config('online')
        cfg = replace(cfg, inference=replace(cfg.inference, schedule='online_fast'))
        model = HDPGPC(cfg)
        for index, sample in enumerate(waves()[:4]):
            result = model.update(sample)
            self.assertEqual(result.observation_index, index)
            self.assertEqual(len(model.labels_), index + 1)
        self.assertEqual(model.n_clusters_, 1)

    def test_warping_and_multi_output_use_public_surface(self):
        for name, outputs in (('dynamic_warping', 1), ('multi_output', 2)):
            with self.subTest(name=name):
                model = HDPGPC(config(name)).fit(waves(outputs))
                self.assertEqual(model.labels_.shape, (6,))
                prediction = model.predictive(int(model.labels_[0]))
                self.assertEqual(prediction.mean.shape, (5, outputs))
                self.assertEqual(prediction.covariance.shape, (outputs, 5, 5))
                self.assertEqual(len(model.posterior(int(model.labels_[0]))), outputs)

    def test_single_sample_and_early_offline_exit_have_labels(self):
        cfg = config('static')
        cfg = replace(cfg, inference=replace(cfg.inference, it_limit=1))
        model = HDPGPC(cfg).fit(waves()[:1])
        np.testing.assert_array_equal(model.labels_, [0])
        model.fit(waves()[:4])
        np.testing.assert_array_equal(model.labels_, [0, 0, 1, 1])

    def test_unspecified_schedule_selects_by_operation(self):
        cfg = config('online')
        cfg = replace(cfg, inference=replace(cfg.inference, schedule=None, it_limit=1))
        offline = HDPGPC(cfg).fit(waves())
        with self.assertRaisesRegex(ValueError, 'offline'):
            offline.update(waves()[0])
        online = HDPGPC(cfg)
        self.assertEqual(online.update(waves()[0]).observation_index, 0)

    def test_input_and_snapshot_isolation(self):
        cfg = config('online')
        model = HDPGPC(cfg)
        original_grid = cfg.x_basis.copy()
        cfg.x_basis[:] = 99.
        exported = model.config
        exported.x_basis[:] = -99.
        np.testing.assert_array_equal(model.config.x_basis, original_grid)
        data = waves()[0]
        saved = data.copy()
        result = model.update(data, include_prediction=True)
        np.testing.assert_array_equal(data, saved)
        prediction = result.prediction.mean.copy()
        with self.assertRaises(ValueError):
            result.prediction.mean[0, 0] = 999.
        with self.assertRaises(AttributeError):
            result.assigned_cluster = 999
        model.update(waves()[1])
        np.testing.assert_array_equal(result.prediction.mean, prediction)

    def test_grid_validation_and_failed_validation_keep_fitted_results(self):
        model = HDPGPC(config('online'))
        with self.assertRaises(RuntimeError):
            _ = model.labels_
        for value in ([], np.zeros((5, 2)), [0., 1., np.nan, 1., 0.]):
            with self.assertRaises(ValueError):
                model.update(value)
        result = model.update(waves()[0], grid=np.arange(5))
        self.assertEqual(result.observation_index, 0)
        with self.assertRaises(ValueError):
            model.update(waves()[1], grid=np.arange(4))
        self.assertEqual(len(model.labels_), 1)
        with self.assertRaises(ValueError):
            model.predictive(100)
        with self.assertRaises(TypeError):
            model.update(waves()[1], include_prediction='yes')

    def test_no_silent_switch_from_explicit_offline_or_classification(self):
        with self.assertRaises(ValueError):
            HDPGPC(config('offline')).update(waves()[0])
        cfg = config('online')
        cfg = replace(cfg, inference=replace(cfg.inference, classify=True))
        with self.assertRaisesRegex(ValueError, 'classify=False'):
            HDPGPC(cfg).fit(waves())

    def test_existing_online_multi_output_failure_is_not_hidden(self):
        cfg = config('multi_output')
        cfg = replace(cfg, inference=InferenceConfig(schedule='online', with_warp=False, n_explore_steps=2))
        model = HDPGPC(cfg)
        with self.assertRaises(IndexError):
            for sample in waves(2)[:2]:
                model.update(sample)
        with self.assertRaisesRegex(RuntimeError, 'last inference call failed'):
            _ = model.labels_
