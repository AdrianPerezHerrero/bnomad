"""Optional source positions preserve beat identity in corrected MIT-BIH arrays."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hdpgpc.datasets import load_mitbih_record
from hdpgpc.experiments.records import _record_jobs, build_parser


class MITBIHAnnotationSamplesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.values = np.arange(5*90*2, dtype=float).reshape(5, 90, 2)
        self.labels = np.array(['N', 'V', 'N', 'Q', 'N'])
        self.samples = np.array([99, 431, 660, 862, 1002], dtype=np.int64)
        np.save(self.root/'203.npy', self.values)
        np.save(self.root/'203_labels.npy', self.labels)

    def job(self):
        args = build_parser('mitbih', 'pca').parse_args([
            '--data-dir', str(self.root), '--records', 'all'])
        job, = _record_jobs(args)
        return job

    def test_legacy_files_keep_tail_alignment_and_absent_positions(self):
        np.save(self.root/'203_labels.npy', np.append(self.labels, 'V'))
        record = load_mitbih_record(self.root, '203')
        np.testing.assert_array_equal(record.waveforms, self.values)
        np.testing.assert_array_equal(record.annotations, self.labels)
        self.assertEqual(record.omitted_trailing_annotations, 1)
        self.assertIsNone(record.annotation_samples)
        job = self.job()
        self.assertEqual(set(job[5]), {'source_row'})
        self.assertNotIn('annotation_samples_available', job[4])

    def test_positions_are_loaded_and_propagated_without_changing_observations(self):
        before = self.job()
        np.save(self.root/'203_annotation_samples.npy', self.samples)
        record = load_mitbih_record(self.root, '203')
        np.testing.assert_array_equal(record.annotation_samples, self.samples)
        np.testing.assert_array_equal(record.annotations, self.labels)
        after = self.job()
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(before[2], after[2])
        np.testing.assert_array_equal(after[5]['annotation_sample'], self.samples)
        self.assertTrue(after[4]['annotation_samples_available'])
        self.assertNotEqual(before[6], after[6])
        # A timestamp correction must invalidate resume even when labels/data match.
        np.save(self.root/'203_annotation_samples.npy', self.samples+1)
        self.assertNotEqual(after[6], self.job()[6])

    def test_invalid_positions_fail_before_a_record_can_be_used(self):
        invalid = [self.samples[:-1], self.samples[:, None], self.samples.astype(float),
                   np.array([-1, 431, 660, 862, 1002]),
                   np.array([99, 431, 431, 862, 1002]),
                   np.array([99, 660, 431, 862, 1002]),
                   np.array([99, 431, 660, 862, 2**63], dtype=np.uint64)]
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                np.save(self.root/'203_annotation_samples.npy', value)
                with self.assertRaisesRegex(ValueError, 'annotation sample positions'):
                    load_mitbih_record(self.root, '203')


if __name__ == '__main__':
    unittest.main()
