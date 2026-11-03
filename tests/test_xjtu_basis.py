"""Channel isolation, paired Fourier ranks, full Haar round trips and XJTU runs."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hdpgpc.datasets import xjtu_sy
from hdpgpc.experiments.xjtu_sy import build_parser, _validate, run_xjtu_sy
from hdpgpc.experiments._xjtu_basis import estimate_basis_observation
from tests.test_xjtu_sy import fixture


class XJTUBasisMathTests(unittest.TestCase):
    def test_fourier_recovers_distinct_channel_frequencies_with_whole_pairs(self):
        args=build_parser(representation='fourier').parse_args([
            '--window-samples','32','--components','4'])
        _validate(args)
        phase=2*np.pi*np.arange(20)/20
        time=2*np.pi*np.arange(32)/32
        values=np.stack((2*np.cos(3*time[None,:]+phase[:,None]),
                         np.sin(7*time[None,:]-phase[:,None])),axis=-1).reshape(20,-1)
        estimate=estimate_basis_observation(values,args)
        C=estimate.model.C
        np.testing.assert_allclose(C.T @ C,np.eye(4),atol=1e-12)
        np.testing.assert_array_equal(C[1::2,:2],0.)
        np.testing.assert_array_equal(C[0::2,2:],0.)
        coefficients=estimate.coefficient_metadata
        self.assertEqual([(c['channel'],c['frequency_bin'],c['kind']) for c in coefficients],
                         [(0,3,'cos'),(0,3,'sin'),(1,7,'cos'),(1,7,'sin')])
        decoded=estimate.model.offset+(values-estimate.model.offset) @ C @ C.T
        np.testing.assert_allclose(decoded,values,atol=1e-12)
        self.assertAlmostEqual(estimate.explained_variance_ratio,1.)

    def test_full_haar_retains_both_channels_without_acquisition_rank_limit(self):
        args=build_parser(representation='wavelet').parse_args([
            '--window-samples','16','--components','32','--max-series','4',
            '--calibration-samples','4'])
        _validate(args)
        values=np.random.default_rng(3).normal(size=(4,32))
        estimate=estimate_basis_observation(values,args)
        C=estimate.model.C
        np.testing.assert_allclose(C.T @ C,np.eye(32),atol=1e-12)
        np.testing.assert_allclose(estimate.model.offset+(values-estimate.model.offset) @ C @ C.T,
                                   values,atol=1e-12)
        self.assertTrue(np.all(np.linalg.eigvalsh(estimate.model.latent_covariance)>0))
        self.assertTrue(np.all(np.diag(estimate.model.Sigma)>0))

    def test_truncated_haar_learns_one_common_coordinate_selection(self):
        args=build_parser(representation='wavelet').parse_args([
            '--window-samples','16','--components','5'])
        values=np.zeros((8,16,2))
        values[:,0,0]=np.arange(8)
        values[:,15,1]=np.arange(8)[::-1]
        first=estimate_basis_observation(values.reshape(8,-1),args)
        second=estimate_basis_observation(values[::-1].reshape(8,-1),args)
        np.testing.assert_array_equal(first.model.C,second.model.C)
        self.assertEqual(first.model.C.shape,(32,5))
        self.assertEqual([len(x) for x in first.selected_columns],[3,2])
        self.assertGreater(first.explained_variance_ratio,.5)

    def test_windowed_fourier_metadata_and_invalid_ranks(self):
        args=build_parser(representation='fourier').parse_args([
            '--window-samples','16','--window-size','8','--frequency-bins','1',
            '--components','8'])
        _validate(args)
        estimate=estimate_basis_observation(np.random.default_rng(1).normal(size=(10,32)),args)
        self.assertEqual({x['start_sample'] for x in estimate.coefficient_metadata},{0,8})
        self.assertEqual({x['frequency_hz'] for x in estimate.coefficient_metadata},{3200.})
        for representation,flags in [
            ('fourier',['--window-size','0']),
            ('fourier',['--frequency-bins','1','--components','6']),
            ('fourier',['--frequency-bins','1','--components','8']),
            ('wavelet',['--window-samples','30']),
            ('wavelet',['--components','1'])]:
            with self.subTest(representation=representation,flags=flags),self.assertRaises(ValueError):
                _validate(build_parser(representation=representation).parse_args(flags))

    def test_only_new_runners_default_to_physical_clock(self):
        for representation in ['pca','autoencoder','fourier','wavelet']:
            args=build_parser(representation=representation).parse_args([])
            self.assertEqual(args.dynamics_clock,'physical-clock' if representation in ('fourier','wavelet')
                             else 'cluster-clock')


class XJTUBasisExperimentTests(unittest.TestCase):
    def test_shared_basis_artifacts_labels_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=fixture(root)
            for representation in ['fourier','wavelet']:
                with self.subTest(representation=representation), \
                     patch.object(xjtu_sy,'XJTU_RECORDS',records),patch.object(xjtu_sy,'XJTU_SAMPLES',64):
                    args=build_parser(representation=representation).parse_args([
                        '--data-dir',str(root),'--output-dir',str(root/representation),
                        '--records','Bearing1_1','Bearing1_4','--window-samples','16','--components','4',
                        '--calibration-samples','16','--iterations','2','--exploration-steps','1',
                        '--max-clusters','2','--stability-samples','5','--verbose','silent'])
                    with patch('hdpgpc.experiments.xjtu_sy.estimate_autoencoder_observation',
                               side_effect=AssertionError('Basis must not train an AE')), \
                         patch('hdpgpc.experiments.xjtu_sy.estimate_pca_observation',
                               side_effect=AssertionError('Basis must not apply PCA')):
                        reports=run_xjtu_sy(args)
                    self.assertEqual([r['status'] for r in reports],['ok','ok'])
                    decoders=[]
                    for report in reports:
                        p=Path(report['output_directory'])
                        self.assertEqual(report['metadata']['representation'],representation)
                        self.assertEqual(report['metadata']['dynamics_clock'],'physical-clock')
                        self.assertTrue(report['metadata']['phase_preserved'])
                        self.assertFalse(report['metadata']['preliminary_projection'])
                        self.assertEqual(report['metrics'],{})
                        self.assertIn(f'{representation}_calibration.png',report['artifacts'])
                        basis=json.loads((p/'basis.json').read_text())
                        self.assertEqual(len(basis['coefficients']),4)
                        self.assertEqual(basis['retained_columns_per_channel'],[2,2])
                        with np.load(p/f'{representation}_calibration.npz') as saved:
                            decoders.append(saved['C'])
                            self.assertNotIn('losses',saved.files)
                        with np.load(p/'latent_trajectories.npz') as paths:
                            np.testing.assert_array_equal(paths['cluster_0_indices'],np.arange(10))
                    np.testing.assert_array_equal(*decoders)
                    args.resume=True
                    with patch('hdpgpc.experiments.xjtu_sy.calibrate_windows',
                               side_effect=AssertionError('Resume must skip basis calibration')), \
                         patch('hdpgpc.experiments.xjtu_sy.HDPGPC',
                               side_effect=AssertionError('Resume must skip inference')):
                        resumed=run_xjtu_sy(args)
                    self.assertEqual([r['run_id'] for r in reports],[r['run_id'] for r in resumed])


if __name__=='__main__':unittest.main()
