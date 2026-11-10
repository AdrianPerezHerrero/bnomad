"""The screen-only path uses shared calibration and reporting across domains."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from hdpgpc.datasets import flutter, xjtu_sy
from hdpgpc.experiments import flutter as flutter_run, xjtu_sy as xjtu_run, ucr
from hdpgpc.experiments import zebrafish, synthetic_image_clustering
from test_flutter import write_fixture as flutter_fixture
from test_xjtu_sy import fixture as bearing_fixture
from test_zebrafish import write_fixture as fish_fixture


FLAGS = ['--screen-priors', '--screen-samples', '8', '--screen-validation', '2',
         '--screen-max-fits', '2', '--screen-iterations', '1', '--verbose', 'silent']


class PriorScreenWorkflowTests(unittest.TestCase):
    def check_report(self, report):
        self.assertEqual(report['status'], 'ok', report.get('error'))
        self.assertEqual(report['mode'], 'prior_screen')
        self.assertEqual(report['fitted'], 0)
        path = Path(report['output_directory'])
        self.assertTrue((path/'prior_screen.json').is_file())
        self.assertFalse((path/'assignments.csv').exists())
        content = json.loads((path/'prior_screen.json').read_text())
        self.assertLessEqual(content['fit_calls'], 2)
        self.assertTrue(content['recommendation'])

    def test_flutter_screen_and_resume_skip_full_comparison_figures(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            records, digest = flutter_fixture(root)
            with patch.object(flutter, 'FLUTTER_RECORDS', records), patch.object(flutter, 'FLUTTER_SHA256', digest):
                args = flutter_run.build_parser().parse_args([
                    '--data-dir', str(root), '--output-dir', str(root/'runs'),
                    '--records', '0700_smallIC', '--configurations', 'dynamic',
                    '--window-samples', '12', '--step', '6', '--components', '2',
                    '--calibration-samples', '20', '--epochs', '2', *FLAGS])
                report, = flutter_run.run_flutter(args)
                self.check_report(report)
                self.assertTrue((Path(report['output_directory'])/'prior_screen.png').is_file())
                self.assertFalse((root/'runs/comparison.png').exists())
                args.resume = True
                resumed, = flutter_run.run_flutter(args)
                self.assertEqual(resumed['recommendation'], report['recommendation'])

    def test_bearing_fourier_screen_and_resume(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); records = bearing_fixture(root)
            with patch.object(xjtu_sy, 'XJTU_RECORDS', records), patch.object(xjtu_sy, 'XJTU_SAMPLES', 64):
                args = xjtu_run.build_parser(representation='fourier').parse_args([
                    '--data-dir', str(root), '--output-dir', str(root/'runs'),
                    '--records', 'Bearing1_1', '--configurations', 'dynamic', '--window-samples', '16',
                    '--components', '4', '--calibration-samples', '10', '--no-plots', *FLAGS])
                report, = xjtu_run.run_xjtu_sy(args)
                self.check_report(report)
                args.resume = True
                resumed, = xjtu_run.run_xjtu_sy(args)
                self.assertEqual(resumed['recommendation'], report['recommendation'])

    def test_expression_screens_both_arms_without_reference_metrics(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory); fish_fixture(root)
            args = zebrafish.build_parser(representation='pca').parse_args([
                '--data-dir', str(root), '--output-dir', str(root/'runs'), '--components', '1',
                '--projection-components', '4', '--no-plots', *FLAGS])
            reports = zebrafish.run_zebrafish(args)
            self.assertEqual(len(reports), 2)
            for report in reports:
                self.check_report(report)
                self.assertEqual(report['metrics'], {})

    def test_ucr_fixed_decoder_prior_multipliers(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            with zipfile.ZipFile(root/'archive.zip', 'w') as archive:
                for split in ('TRAIN', 'TEST'):
                    y = np.tile([1, 2], 12)
                    x = np.linspace(0, 2*np.pi, 12)
                    values = y[:, None]*np.sin(x)[None]+np.random.default_rng(7).normal(scale=.1, size=(24, 12))
                    buffer = io.StringIO()
                    np.savetxt(buffer, np.c_[y, values], delimiter='\t')
                    archive.writestr(f'UCRArchive_2018/Coffee/Coffee_{split}.tsv', buffer.getvalue())
            args = ucr.build_parser('pca').parse_args([
                '--archive', str(root/'archive.zip'), '--datasets', 'Coffee',
                '--output-dir', str(root/'runs'), '--components', '2', '--max-points', '12',
                '--no-plots', *FLAGS])
            report, = ucr.run_ucr(args)
            self.check_report(report)

    def test_synthetic_screen_uses_explicit_or_default_gamma_not_truth(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            args = synthetic_image_clustering.build_parser().parse_args([
                '--output-dir', directory, '--image-size', '8', '8', '--regime-order', '0', '1',
                '--segment-length', '12', '--no-plots', *FLAGS])
            report = synthetic_image_clustering.run_synthetic_clustering(args)
            self.check_report(report)
            saved = json.loads((Path(report['output_directory'])/'prior_screen.json').read_text())
            self.assertEqual(saved['candidates'][0]['gamma_prior'], .1)


if __name__ == '__main__':
    unittest.main()
