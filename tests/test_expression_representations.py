"""Fixed expression decoders and paired PCA/autoencoder experiment contracts."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hdpgpc.emissions import AffineGaussianObservation
from hdpgpc.experiments._expression import make_config
from hdpgpc.experiments.zebrafish import build_parser as zebrafish_parser, run_zebrafish
from hdpgpc.experiments.xenopus import build_parser as xenopus_parser, run_xenopus
from tests.test_zebrafish import write_fixture as zebrafish_fixture
from tests.test_xenopus import write_fixture as xenopus_fixture


class ExpressionRepresentationTests(unittest.TestCase):
    def test_fixed_decoder_and_independent_transition_policy(self):
        decoder = AffineGaussianObservation(np.ones((4, 1)), np.eye(4)*.2,
            np.arange(4.)[:, None], np.zeros(4), np.array([[2.]]))
        for representation in ('pca', 'autoencoder'):
            with self.subTest(representation=representation):
                args = zebrafish_parser(representation=representation).parse_args([])
                self.assertEqual(args.sigma_prior, 1.)
                for dynamic in (False, True):
                    config = make_config(args, decoder, dynamic=dynamic)
                    config.validate(for_build=True)
                    self.assertEqual(config.observation.family, 'linear')
                    self.assertEqual(config.observation.C_policy.mode, 'fixed')
                    self.assertEqual(config.observation.Sigma_policy.mode, 'fixed')
                    np.testing.assert_array_equal(config.observation.C_policy.value, decoder.C)
                    self.assertTrue(config.hdp.hmm_switch)
                    if dynamic:
                        self.assertEqual(config.dynamics.A_policy.mode, 'cluster')
                    else:
                        np.testing.assert_array_equal(config.dynamics.A_policy.value, [[0.]])
                        self.assertEqual(config.dynamics.Gamma_policy.mode, 'cluster')
                    np.testing.assert_array_equal(config.observation.fixed_latent_covariance, [[2.]])

    def test_both_datasets_and_representations_save_fixed_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for dataset, parser, runner in (
                    ('zebrafish', zebrafish_parser, run_zebrafish),
                    ('xenopus', xenopus_parser, run_xenopus)):
                source = root/dataset; source.mkdir()
                zebrafish_fixture(source) if dataset == 'zebrafish' else xenopus_fixture(source, 'A')
                for representation in ('pca', 'autoencoder'):
                    with self.subTest(dataset=dataset, representation=representation):
                        flags = ['--data-dir', str(source), '--output-dir', str(source/representation),
                            '--components', '1', '--projection-components', '4', '--iterations', '2',
                            '--exploration-steps', '1', '--max-clusters', '3', '--verbose', 'silent',
                            '--no-trajectory-plots', '--no-projection-plots']
                        if representation == 'autoencoder':
                            flags += ['--epochs', '3']
                        args = parser(representation=representation).parse_args(flags)
                        reports = runner(args)
                        self.assertEqual([report['status'] for report in reports], ['ok', 'ok'])
                        calibrations = []
                        for report in reports:
                            self.assertEqual(report['metadata']['representation'], representation)
                            self.assertTrue(report['metadata']['decoder_is_fixed'])
                            self.assertEqual(report['experiment'], f'{dataset}_{representation}_'+report['metadata']['configuration'])
                            self.assertNotIn('mfa_posteriors.npz', report['artifacts'])
                            path = Path(report['output_directory'])
                            with np.load(path/'observation_calibration.npz', allow_pickle=False) as saved:
                                calibrations.append({key: saved[key] for key in ('scores','decoder_C','decoder_Sigma')})
                            if representation == 'autoencoder':
                                self.assertIn('autoencoder_calibration.png', report['artifacts'])
                                with np.load(path/'training.npz') as training:
                                    self.assertEqual(training['losses'].shape, (3,))
                                    self.assertTrue(np.isfinite(training['losses']).all())
                            else:
                                self.assertIn('pca_calibration.npz', report['artifacts'])
                        for key in calibrations[0]:
                            np.testing.assert_array_equal(calibrations[0][key], calibrations[1][key])
                        args.resume = True
                        with patch('hdpgpc.experiments._expression._calibrate_observation',
                                   side_effect=AssertionError('recalibrated on resume')):
                            resumed = runner(args)
                        self.assertEqual([report['run_id'] for report in resumed],
                                         [report['run_id'] for report in reports])


if __name__ == '__main__':
    unittest.main()
