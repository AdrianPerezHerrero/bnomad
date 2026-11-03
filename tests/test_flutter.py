"""Pinned flutter loading, amplitude preservation, unlabeled inference and diagnostics."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from hdpgpc.datasets import flutter
from hdpgpc.experiments.flutter import build_parser, run_flutter
from hdpgpc.diagnostics._flutter import transition_radius_samples


def write_fixture(root, *, bad_time=False):
    records = {}
    with zipfile.ZipFile(root/flutter.FLUTTER_ARCHIVE, 'w') as bundle:
        for name, growth in (('0700_smallIC', -.3), ('0950', .4)):
            times = np.arange(300)*.005
            if bad_time:
                times[100] = times[99]
            phase = 2*np.pi*3*times
            motion = np.column_stack((np.sin(phase)*np.exp(growth*times),
                                      .02*np.cos(phase)*np.exp(growth*times)))
            buffer = io.StringIO()
            np.savetxt(buffer, np.column_stack((times, motion)))
            raw = buffer.getvalue().encode()
            bundle.writestr(flutter.FLUTTER_PREFIX+f'timeRespData_{name}.txt', raw)
            speed, initial, reference, _ = flutter.FLUTTER_RECORDS[name]
            records[name] = (speed, initial, reference, hashlib.sha256(raw).hexdigest())
    digest = hashlib.sha256((root/flutter.FLUTTER_ARCHIVE).read_bytes()).hexdigest()
    return records, digest


class FlutterDataTests(unittest.TestCase):
    def test_pinned_archive_causal_windows_and_preserved_amplitude(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, digest = write_fixture(root)
            with patch.object(flutter, 'FLUTTER_RECORDS', records), patch.object(flutter, 'FLUTTER_SHA256', digest):
                self.assertEqual(flutter.download_flutter(root), root/flutter.FLUTTER_ARCHIVE)
                record = flutter.load_flutter_record(root, '0950')
                windows = flutter.window_flutter_record(record, window_samples=12, step=6)
                np.testing.assert_array_equal(windows.values[0].reshape(12, 2), record.motion[:12])
                np.testing.assert_array_equal(windows.times, record.times[windows.ends])
                np.testing.assert_array_equal(windows.starts, np.arange(0, 289, 6))
                doubled = flutter.FlutterRecord(record.name, record.times, record.motion*2,
                    record.airflow_speed, record.initial_condition, record.reference,
                    record.file_sha256, record.archive_sha256)
                np.testing.assert_array_equal(flutter.window_flutter_record(doubled,
                    window_samples=12, step=6).values, windows.values*2)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                flutter.load_flutter_record(root)

    def test_duplicate_times_and_unknown_records_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, digest = write_fixture(root, bad_time=True)
            with patch.object(flutter, 'FLUTTER_RECORDS', records), patch.object(flutter, 'FLUTTER_SHA256', digest):
                with self.assertRaisesRegex(ValueError, 'timestamps'):
                    flutter.load_flutter_record(root, '0950')
                with self.assertRaisesRegex(ValueError, 'Unknown'):
                    flutter.load_flutter_record(root, 'invalid')


class FlutterExperimentTests(unittest.TestCase):
    def test_common_calibration_unlabeled_results_and_lazy_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records, digest = write_fixture(root)
            with patch.object(flutter, 'FLUTTER_RECORDS', records), patch.object(flutter, 'FLUTTER_SHA256', digest):
                args = build_parser().parse_args(['--data-dir', str(root), '--output-dir', str(root/'runs'),
                    '--records', '0700_smallIC', '0950', '--configurations', 'independent', 'dynamic',
                    '--window-samples', '12', '--step', '6', '--components', '2', '--calibration-samples', '40',
                    '--epochs', '4', '--iterations', '2', '--exploration-steps', '1', '--max-clusters', '2',
                    '--stability-samples', '20', '--verbose', 'silent'])
                reports = run_flutter(args)
                self.assertEqual([report['status'] for report in reports], ['ok']*4)
                decoders = []
                for report in reports:
                    self.assertFalse(report['reference_labels_available'])
                    self.assertEqual(report['metrics'], {})
                    self.assertEqual(report['classes'], 0)
                    path = Path(report['output_directory'])
                    with (path/'assignments.csv').open(newline='') as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertTrue(all(row['true_label'] == '' for row in rows))
                    self.assertTrue(all(row['record_reference'] for row in rows))
                    with np.load(path/'autoencoder_calibration.npz', allow_pickle=False) as saved:
                        decoders.append(saved['C'])
                        self.assertEqual(saved['channel_scale'].shape, (2,))
                        self.assertEqual(saved['losses'].shape, (4,))
                        self.assertLess(saved['losses'][-1], saved['losses'][0])
                    self.assertIn('clusters_output_0.png', report['artifacts'])
                    self.assertIn('latent_embedding.png', report['artifacts'])
                    self.assertIn('latent_trajectories.png', report['artifacts'])
                    self.assertIn('dynamics.json', report['artifacts'])
                    self.assertIn('Results saved to:', (path/'clustering.log').read_text())
                    diagnostics = report['metadata']['diagnostics']
                    self.assertTrue(np.isfinite(diagnostics['autoencoder_explained_variance']))
                    if report['metadata']['configuration'] == 'independent':
                        self.assertTrue(all(row['latent_spectral_radius'] == 0 for row in diagnostics['clusters']))
                for decoder in decoders[1:]:
                    np.testing.assert_array_equal(decoder, decoders[0])
                self.assertTrue((root/'runs/comparison.png').is_file())
                self.assertTrue((root/'runs/autoencoder_structure.png').is_file())
                args.resume = True
                with patch('hdpgpc.experiments.flutter.calibrate_windows',
                           side_effect=AssertionError('Resume must skip training')), \
                     patch('hdpgpc.experiments.flutter.HDPGPC', side_effect=AssertionError('Resume must skip fitting')):
                    resumed = run_flutter(args)
                self.assertEqual([r['run_id'] for r in resumed], [r['run_id'] for r in reports])

    def test_invalid_scales_and_sample_normalization_fail(self):
        from hdpgpc.experiments.flutter import _validate
        for flags in (['--normalize-samples'], ['--epochs', '0'], ['--gamma-prior', 'nan'],
                      ['--window-samples', '1'], ['--stability-samples', '-1']):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                _validate(build_parser().parse_args(flags))


class FlutterStabilityTests(unittest.TestCase):
    def test_mniw_sampling_preserves_posterior_and_distinguishes_stable_modes(self):
        from hdpgpc.dynamics._distributions import matrix_normal_inv_wishart
        for radius in (.8, 1.2):
            posterior = matrix_normal_inv_wishart(np.eye(2)*radius, np.eye(2)*1e6,
                                                 50., np.eye(2)*1e-4)
            before = posterior.m_mean.clone()
            samples = transition_radius_samples(posterior, 50, 3)
            self.assertEqual(samples.shape, (50,))
            self.assertTrue(np.isfinite(samples).all())
            np.testing.assert_array_equal(posterior.m_mean.numpy(), before.numpy())
            self.assertTrue(np.all(samples < 1) if radius < 1 else np.all(samples > 1))


if __name__ == '__main__':
    unittest.main()
