"""The ECG recipe must learn diagonal noise through its static fitting path."""
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if not os.environ.get('HDPGPC_TEST_INSTALLED'):
    sys.path.insert(0, str(ROOT/'hdpgpc'))
import numpy as np
from hdpgpc import HDPGPC, HDPGPCConfig

spec = importlib.util.spec_from_file_location('ecg_quickstart', ROOT/'examples/ecg_quickstart.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class ECGQuickstartNoiseTests(unittest.TestCase):
    def test_static_fit_learns_distinct_diagonal_noise_with_fixed_identity_C(self):
        grid = np.linspace(0., 4., 12)[:, None]
        config = example.make_config(grid)
        config = HDPGPCConfig.from_dict(config.to_dict())
        rng = np.random.default_rng(17)
        peak = np.exp(-((grid[:, 0]-1.8)/.55)**2)
        x = np.concatenate([
            peak + rng.normal(size=(6, len(grid)))*np.linspace(.01, .04, len(grid)),
            -peak + rng.normal(size=(6, len(grid)))*np.linspace(.04, .12, len(grid))])[:, :, None]
        with redirect_stdout(io.StringIO()):
            model = HDPGPC(config).fit(x, grid=grid)
        parameters = model.observation_parameters_
        self.assertEqual(parameters['scope'], 'cluster')
        self.assertGreaterEqual(model.n_clusters_, 2, 'Workload must exercise separate noise posteriors')
        diagonals = []
        for cluster in model.cluster_ids_:
            with self.subTest(cluster=int(cluster)):
                observation = parameters['outputs'][0][int(cluster)]
                sigma = observation['Sigma']
                diagonal = np.diag(sigma)
                np.testing.assert_array_equal(sigma, np.diag(diagonal))
                np.testing.assert_array_equal(observation['C'], np.eye(len(grid)))
                self.assertTrue(np.isfinite(diagonal).all() and (diagonal > 0).all())
                self.assertFalse(np.allclose(diagonal, config.emission.ini_sigma),
                                 'Static execution silently kept the initial noise')
                self.assertGreater(np.ptp(diagonal), 1e-6, 'Noise must vary along the waveform')
                np.testing.assert_array_equal(model.predictive(int(cluster), grid=grid).covariance[0], sigma)
                diagonals.append(diagonal)
        self.assertTrue(any(not np.allclose(diagonals[0], other) for other in diagonals[1:]),
                        'Clusters must own separate, data-dependent noise estimates')


if __name__ == '__main__':
    unittest.main()
