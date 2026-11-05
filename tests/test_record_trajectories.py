"""Record trajectory figures share the ordinary calibration and report path."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hdpgpc.experiments.records import build_parser, run_records


class RecordTrajectoryTests(unittest.TestCase):
    def test_shared_flag_preserves_cas_colon_defaults_and_overrides(self):
        from hdpgpc.experiments.cas_colon import build_parser as cas_parser
        for representation in ('pca', 'autoencoder', 'sfa', 'linear_prediction', 'mfa'):
            with self.subTest(representation=representation):
                parser = cas_parser(representation=representation)
                self.assertTrue(parser.parse_args([]).trajectory_plots)
                self.assertFalse(parser.parse_args(['--no-trajectory-plots']).trajectory_plots)
                parser = cas_parser(representation=representation, defaults={'trajectory_plots': False})
                self.assertFalse(parser.parse_args([]).trajectory_plots)
                self.assertTrue(parser.parse_args(['--trajectory-plots']).trajectory_plots)

    def test_mitbih_representations_clocks_and_calibration_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(8)
            x = np.linspace(0., 2*np.pi, 8)
            values = np.stack([np.sin(x), np.cos(x)], axis=1)[None].repeat(12, axis=0)
            values[6:] *= -1
            values += rng.normal(scale=.02, size=values.shape)
            np.save(root/'207.npy', values)
            np.save(root/'207_labels.npy', np.repeat(['N', 'V'], 6))
            for representation, clock, channels in (
                    ('waveform', 'cluster-clock', ['0', '1']),
                    ('pca', 'cluster-clock', ['0']),
                    ('autoencoder', 'physical-clock', ['0'])):
                with self.subTest(representation=representation, clock=clock):
                    flags = ['--data-dir', str(root), '--records', '207',
                        '--output-dir', str(root/representation), '--channels', *channels,
                        '--components', '2', '--calibration-samples', '12', '--epochs', '3',
                        '--iterations', '1', '--exploration-steps', '1', '--max-clusters', '2',
                        '--offline-strategy', 'split_merge', '--lds-update', 'conjugate_batch',
                        '--dynamics-clock', clock, '--trajectory-plots', '--no-fit-gp',
                        '--alignment', 'none', '--no-projection-plots', '--no-lds-plots',
                        '--no-lds-dynamics-plots', '--verbose', 'silent']
                    parser = build_parser('mitbih', representation)
                    report, = run_records(parser.parse_args(flags))
                    self.assertEqual(report['status'], 'ok', report.get('error'))
                    names = (['latent_trajectories.png'] if len(channels) == 1 else
                             [f'latent_trajectories_output_{output}.png' for output in range(2)])
                    for name in names:
                        self.assertIn(name, report['artifacts'])
                        self.assertTrue((Path(report['output_directory'])/name).is_file())
                    if representation == 'autoencoder':
                        self.assertIn('autoencoder_calibration.png', report['artifacts'])
                    if representation == 'pca':
                        # Master plotting switch suppresses the new diagnostic too.
                        args = parser.parse_args([*flags, '--no-plots'])
                        with patch('hdpgpc.diagnostics.plot_latent_trajectories',
                                   side_effect=AssertionError('Plot requested with --no-plots')):
                            quiet, = run_records(args)
                        self.assertEqual(quiet['status'], 'ok', quiet.get('error'))
                        self.assertFalse(any(name.endswith('.png') for name in quiet['artifacts']))
                        self.assertFalse(parser.parse_args([*flags, '--no-trajectory-plots']).trajectory_plots)

    def test_online_trajectory_request_fails_before_loading_data(self):
        for schedule in ('online', 'online_fast'):
            with self.subTest(schedule=schedule):
                parser = build_parser('mitbih', 'autoencoder')
                self.assertFalse(parser.parse_args([]).trajectory_plots)
                args = parser.parse_args(['--trajectory-plots', '--schedule', schedule])
                with patch('hdpgpc.experiments.records._record_jobs',
                           side_effect=AssertionError('Loaded data before validation')):
                    with self.assertRaisesRegex(ValueError, 'trajectory-plots.*offline'):
                        run_records(args)


if __name__ == '__main__':
    unittest.main()
