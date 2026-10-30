"""Unfiltered expression loading, chronology, and paired experiment contracts."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hdpgpc.datasets.zebrafish import (ZEBRAFISH_FILES, ZEBRAFISH_STAGES,
    download_zebrafish, load_zebrafish_expression)
from hdpgpc.preprocessing import fit_pca_projection


def write_fixture(root):
    rng = np.random.default_rng(7)
    records, columns = [], []
    # Reversed metadata and count column order make positional joins incorrect.
    for i, (stage, _) in enumerate(ZEBRAFISH_STAGES):
        category = 'early' if i < 9 else 'late'
        for replicate in range(2):
            sample = f'sample_{i}_{replicate}'
            records.append(dict(sample=sample, stageName=stage, stage=f'{category}:{stage}', sequencing='RNASeq'))
            columns.append(sample)
    records.append(dict(sample='other_assay', stageName='Dome', stage='early:Dome', sequencing='DeTCT'))
    with (root / ZEBRAFISH_FILES[0]).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter='\t')
        writer.writeheader(); writer.writerows(records[::-1])
    counts = rng.integers(1, 1000, (len(columns), 6)).astype(float)
    counts[:, 0] = 0  # Zero genes must survive, not be filtered before PCA.
    with (root / ZEBRAFISH_FILES[1]).open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['Gene ID', 'Name', *columns[::-1]])
        for i in range(counts.shape[1]):
            writer.writerow([f'gene{i}', f'name{i}', *counts[::-1, i]])
        writer.writerow(['__no_feature', 'NA', *([1000000]*len(columns))])
    return counts


class ZebrafishDataTests(unittest.TestCase):
    def test_join_chronology_all_genes_and_simultaneous_replicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); counts = write_fixture(root)
            data = load_zebrafish_expression(root)
            np.testing.assert_array_equal(data.counts, counts)
            self.assertEqual(data.counts.shape, (36, 6))
            np.testing.assert_array_equal(data.stages, [stage for stage, _ in ZEBRAFISH_STAGES])
            self.assertTrue(np.all(np.diff(data.hours) > 0))
            values, sd, n = data.stage_values()
            transformed = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 1e6)
            np.testing.assert_allclose(values, transformed.reshape(18, 2, 6).mean(axis=1))
            np.testing.assert_allclose(sd, transformed.reshape(18, 2, 6).std(axis=1, ddof=1))
            np.testing.assert_array_equal(n, np.repeat(2, 18))
            np.testing.assert_array_equal(values[:, 0], 0)
            self.assertIn('__no_feature', data.counting_summary)
            self.assertEqual(len(data.source_sha256), 2)

    def test_nonfinite_missing_samples_and_checksum_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_fixture(root)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download_zebrafish(root)
            path = root / ZEBRAFISH_FILES[1]
            text = path.read_text()
            path.write_text(text.replace('sample_0_0', 'missing_sample', 1))
            with self.assertRaisesRegex(ValueError, 'match metadata'):
                load_zebrafish_expression(root)
            path.write_text(text.replace('gene0\tname0\t0.0', 'gene0\tname0\tnan', 1))
            with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
                load_zebrafish_expression(root)

    def test_thin_projection_preserves_columns_and_reconstructs_rank(self):
        rng = np.random.default_rng(5)
        values = rng.normal(size=(8, 2)) @ rng.normal(size=(2, 100))
        values[:, 0] = 0
        projection = fit_pca_projection(values, components=6)
        self.assertEqual(projection.loadings.shape, (100, 2))
        self.assertEqual(projection.numerical_rank, 2)
        self.assertAlmostEqual(projection.explained_variance_ratio.sum(), 1.)
        reconstructed = projection.transform(values) * projection.scale @ projection.loadings.T + projection.mean
        np.testing.assert_allclose(reconstructed, values, atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            fit_pca_projection(np.zeros((4, 10)), components=2)


class ZebrafishExperimentTests(unittest.TestCase):
    def test_zero_projection_bypasses_pca_and_keeps_gene_space_artifacts(self):
        from hdpgpc.experiments.zebrafish import build_parser, run_zebrafish, _validate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_fixture(root)
            expected = load_zebrafish_expression(root).stage_values()[0]
            args = build_parser().parse_args(['--data-dir', str(root), '--output-dir', str(root/'runs'),
                '--components', '1', '--projection-components', '0', '--iterations', '2',
                '--exploration-steps', '1', '--max-clusters', '3', '--verbose', 'silent',
                '--no-trajectory-plots', '--no-projection-plots'])
            _validate(args)
            with patch('hdpgpc.experiments._expression.fit_pca_projection',
                       side_effect=AssertionError('PCA must be bypassed')):
                reports = run_zebrafish(args)
            self.assertEqual([report['status'] for report in reports], ['ok', 'ok'])
            for report in reports:
                metadata = report['metadata']
                self.assertFalse(metadata['projection_applied'])
                self.assertEqual(metadata['projection_components'], 0)
                self.assertEqual(metadata['observation_dimension'], 6)
                self.assertIsNone(metadata['projection_explained_variance'])
                path = Path(report['output_directory'])
                self.assertIn('clusters_output_0.png', report['artifacts'])
                with np.load(path/'observation_calibration.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['scores'], expected)
                    self.assertFalse(saved['projection_applied'])
                    self.assertNotIn('gene_loadings', saved.files)
                    self.assertNotIn('explained_variance_ratio', saved.files)
                    self.assertEqual(saved['mfa_C'].shape, (6, 1))
            comparison = json.loads((root/'runs/comparison.json').read_text())
            self.assertFalse(comparison['projection_applied'])
            self.assertEqual(comparison['projection_components'], 0)
            self.assertEqual(comparison['observation_dimension'], 6)
            self.assertIsNone(comparison['projection_explained_variance'])
            for invalid in (-1, 1):
                args.projection_components = invalid
                with self.assertRaisesRegex(ValueError, 'projection-components'):
                    _validate(args)

    def test_configuration_changes_only_factor_evolution(self):
        from hdpgpc.experiments.zebrafish import build_parser, make_config
        from hdpgpc.emissions import AffineGaussianObservation
        args = build_parser().parse_args([])
        initial = AffineGaussianObservation(np.ones((4, 1)), np.eye(4),
            np.arange(4.)[:, None], np.zeros(4), np.eye(1))
        independent = make_config(args, initial, dynamic=False)
        dynamic = make_config(args, initial, dynamic=True)
        independent.validate(for_build=True); dynamic.validate(for_build=True)
        self.assertTrue(independent.hdp.hmm_switch and dynamic.hdp.hmm_switch)
        self.assertEqual(independent.hdp, dynamic.hdp)
        np.testing.assert_array_equal(independent.dynamics.A_policy.value, np.zeros((1, 1)))
        np.testing.assert_array_equal(independent.dynamics.Gamma_policy.value, np.eye(1))
        self.assertEqual(dynamic.dynamics.A_policy.mode, 'cluster')
        self.assertEqual(dynamic.dynamics.Gamma_policy.mode, 'cluster')
        np.testing.assert_array_equal(independent.observation.C_policy.value, dynamic.observation.C_policy.value)
        np.testing.assert_array_equal(independent.observation.Sigma_policy.value, dynamic.observation.Sigma_policy.value)

    def test_both_runs_artifacts_target_metrics_and_resume(self):
        from hdpgpc.experiments.zebrafish import build_parser, run_zebrafish
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_fixture(root)
            args = build_parser().parse_args(['--data-dir', str(root), '--output-dir', str(root/'runs'),
                '--components', '1', '--projection-components', '4', '--iterations', '2',
                '--exploration-steps', '1', '--max-clusters', '3', '--no-plots', '--verbose', 'silent'])
            reports = run_zebrafish(args)
            self.assertEqual([r['status'] for r in reports], ['ok', 'ok'])
            self.assertNotEqual(reports[0]['run_id'], reports[1]['run_id'])
            for report in reports:
                path = Path(report['output_directory'])
                for name in ('settings.json', 'config.yaml', 'clustering.log', 'assignments.csv',
                             'lds_parameters.npz', 'mfa_posteriors.npz', 'mfa_shared_posterior.npz',
                             'prepared_data.npz', 'observation_calibration.npz', 'label_metrics.json'):
                    self.assertIn(name, report['artifacts'])
                self.assertFalse(report['metadata']['selected_genes'])
                self.assertEqual(report['metadata']['gene_count'], 6)
                self.assertEqual(report['fitted'], 18)
                self.assertEqual(set(report['metadata']['label_metrics']), {'broad', 'stage'})
                with np.load(path/'prepared_data.npz', allow_pickle=False) as prepared:
                    self.assertEqual(prepared['mean_log1p_cpm'].shape, (18, 6))
                with (path/'assignments.csv').open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 18)
            comparison = json.loads((root/'runs/comparison.json').read_text())
            self.assertEqual(len(comparison['runs']), 2)
            args.resume = True
            with patch('hdpgpc.experiments._expression.HDPGPC', side_effect=AssertionError('refitted on resume')):
                resumed = run_zebrafish(args)
            self.assertEqual([r['run_id'] for r in reports], [r['run_id'] for r in resumed])
            with (root/'runs/results.csv').open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)


if __name__ == '__main__':
    unittest.main()
