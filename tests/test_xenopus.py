"""Xenopus measured-expression loading, annotation masks, and shared execution."""
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hdpgpc.datasets.xenopus import load_xenopus_expression, XENOPUS_FILES, download_xenopus


def write_fixture(root, clutch):
    hours = np.array([0., .5, 1., 2., 3., 4.5, 6.5, 8., 10.5, 12., 15.,
                      17.5, 21.5, 22.5, 24., 33., 51., 52., 56., 60.])
    if clutch == 'B':
        hours = hours[hours <= 24]
    rng = np.random.default_rng(9)
    values = rng.uniform(0, 100, (len(hours), 6))
    values[:, 0] = 0
    order = np.arange(len(hours))[::-1]
    with gzip.open(root/XENOPUS_FILES[clutch], 'wt', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['', '', *[f'sample_{i}' for i in order]])
        writer.writerow(['', '', *hours[order]])
        for i in range(values.shape[1]):
            writer.writerow(['Gene', f'gene_{i}|name_{i}', *values[order, i]])
        writer.writerow(['Isoform', 'gene_0|isoform', *([9999]*len(hours))])
        writer.writerow(['Gene', 'ERCC-00002|', *([100]*len(hours))])
        writer.writerow(['Gene', 'AC_EC2|', *([200]*len(hours))])
    return hours, values


class XenopusDataTests(unittest.TestCase):
    def test_measured_genes_chronology_qc_and_explicit_target_masks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hours, values = write_fixture(root, 'A')
            data = load_xenopus_expression(root)
            keep = ~np.isin(hours, [21.5, 33., 52.])
            np.testing.assert_array_equal(data.hours, hours[keep])
            np.testing.assert_allclose(data.values, np.log1p(values[keep]))
            self.assertEqual(len(data.gene_ids), 6)
            np.testing.assert_array_equal(data.values[:, 0], 0)
            self.assertEqual(data.prepared_arrays['control_ids'].tolist(), ['ERCC-00002|', 'AC_EC2|'])
            self.assertEqual(data.time_arrays['control_tpm'].shape, (keep.sum(), 2))
            self.assertFalse(data.metadata['author_smoothed_expression'])
            self.assertEqual(data.targets['stage'][data.hours == 15].item(), 'NF16/17')
            self.assertEqual(data.targets['stage'][data.hours == .5].item(), 'unannotated')
            self.assertFalse(data.evaluation_masks['stage'][data.hours == .5].item())
            self.assertFalse(data.evaluation_masks['stage'][data.hours == 60].item())
            self.assertEqual(data.targets['broad'][data.hours == 6.5].item(), 'NF10-12.5 interval')
            self.assertIn('time-derived', data.metadata['target_source'])
            unfiltered = load_xenopus_expression(root, exclude_qc_failures=False)
            self.assertEqual(len(unfiltered.hours), len(hours))
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download_xenopus(root)

    def test_nonfinite_duplicate_and_unknown_inputs_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_fixture(root, 'A')
            path = root/XENOPUS_FILES['A']
            with gzip.open(path, 'rt') as stream:
                original = stream.read()
            for text, message in (
                (original.replace('gene_0|name_0\t0.0', 'gene_0|name_0\tnan', 1), 'finite'),
                (original.replace('gene_1|name_1', 'gene_0|name_0'), 'unique gene'),
                (original.replace('sample_1\t', 'sample_0\t', 1), 'unique samples')):
                with gzip.open(path, 'wt') as stream:
                    stream.write(text)
                with self.assertRaisesRegex(ValueError, message):
                    load_xenopus_expression(root)
            with self.assertRaisesRegex(ValueError, 'clutch'):
                load_xenopus_expression(root, 'C')


class XenopusExperimentTests(unittest.TestCase):
    def test_autoencoder_clutch_noise_defaults_and_explicit_override_are_saved(self):
        import runpy
        from hdpgpc.experiments.xenopus import build_parser, run_xenopus
        script = runpy.run_path(str(Path(__file__).resolve().parents[1]/
                                    'examples/xenopus_autoencoder_clustering.py'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for clutch in ('A', 'B'):
                write_fixture(root, clutch)
            run_ids = []
            for override in (None, 0.01):
                with self.subTest(override=override):
                    flags = ['--data-dir', str(root), '--output-dir', str(root/'runs'),
                        '--components', '1', '--projection-components', '4', '--epochs', '3',
                        '--iterations', '2', '--exploration-steps', '1', '--max-clusters', '2',
                        '--no-plots', '--verbose', 'silent', '--resume']
                    if override is not None:
                        flags += ['--sigma-prior', str(override)]
                    parser = build_parser(defaults=script['DEFAULTS'],
                                          representation=script['REPRESENTATION'])
                    args = parser.parse_args(flags)
                    reports = run_xenopus(args, sigma_prior_defaults=script['SIGMA_PRIOR_DEFAULTS'])
                    self.assertEqual(args.sigma_prior, override)  # No cross-clutch namespace mutation.
                    self.assertEqual([report['dataset'] for report in reports],
                                     ['xenopus_clutch_A', 'xenopus_clutch_B'])
                    self.assertEqual([report['status'] for report in reports], ['ok', 'ok'])
                    run_ids.append([report['run_id'] for report in reports])
                    for clutch, report in zip(('A', 'B'), reports):
                        settings = json.loads((Path(report['output_directory'])/'settings.json').read_text())['settings']
                        self.assertEqual(settings['sigma_prior'],
                                         override if override is not None else script['SIGMA_PRIOR_DEFAULTS'][clutch])
            self.assertEqual(run_ids[0][0], run_ids[1][0])  # Identical effective A settings resume.
            self.assertNotEqual(run_ids[0][1], run_ids[1][1])  # Explicit 0.01 overrides B's 0.005.

    def test_clutches_run_separately_and_preserve_stage_masks_and_prefix(self):
        from hdpgpc.experiments.xenopus import build_parser, run_xenopus
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for clutch in ('A', 'B'):
                write_fixture(root, clutch)
            args = build_parser().parse_args(['--data-dir', str(root), '--output-dir', str(root/'runs'),
                '--clutches', 'A', 'B', '--components', '1', '--projection-components', '4',
                '--iterations', '2', '--exploration-steps', '1', '--max-clusters', '3',
                '--max-series', '12', '--no-plots', '--verbose', 'silent', '--target', 'stage'])
            reports = run_xenopus(args)
            self.assertEqual(len(reports), 4)
            self.assertTrue(all(report['status'] == 'ok' for report in reports))
            self.assertEqual({report['dataset'] for report in reports}, {'xenopus_clutch_A', 'xenopus_clutch_B'})
            for report in reports:
                self.assertEqual(report['fitted'], 12)
                self.assertLess(report['metadata']['evaluated_observations']['stage'], report['fitted'])
                self.assertEqual(report['metrics']['evaluation_frames'],
                                 report['metadata']['evaluated_observations']['stage'])
                with np.load(Path(report['output_directory'])/'prepared_data.npz') as saved:
                    self.assertEqual(saved['log1p_tpm'].shape, (12, 6))
                    self.assertEqual(saved['control_tpm'].shape, (12, 2))
            for clutch in ('A', 'B'):
                comparison = json.loads((root/'runs'/f'comparison_clutch_{clutch}.json').read_text())
                self.assertEqual(comparison['dataset'], f'xenopus_clutch_{clutch}')
                self.assertEqual(len(comparison['runs']), 2)


if __name__ == '__main__':
    unittest.main()
