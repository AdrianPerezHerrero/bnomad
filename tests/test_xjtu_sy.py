"""Bearing acquisition clocks, bounded downloads, raw-window provenance and real inference."""
import csv
from contextlib import ExitStack
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from hdpgpc.datasets import xjtu_sy
from hdpgpc.experiments.xjtu_sy import build_parser, run_xjtu_sy


def fixture(root):
    records={'Bearing1_1':(10,'outer race'),'Bearing1_4':(10,'cage')}
    for bearing in records:
        folder=root/bearing;folder.mkdir()
        for acquisition in range(1,11):
            times=np.arange(64)/25600
            amplitude=1 if acquisition<7 else 4
            data=np.column_stack((amplitude*np.sin(2*np.pi*1200*times+acquisition),
                                  .5*amplitude*np.cos(2*np.pi*1400*times-acquisition)))
            np.savetxt(folder/f'{acquisition}.csv',data,delimiter=',',
                header='Horizontal_vibration_signals,Vertical_vibration_signals',comments='')
    return records


class XJTUDataTests(unittest.TestCase):
    def test_acquisition_clock_raw_amplitudes_and_missing_bursts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=fixture(root)
            with patch.object(xjtu_sy,'XJTU_RECORDS',records),patch.object(xjtu_sy,'XJTU_SAMPLES',64):
                data=xjtu_sy.load_xjtu_sy_windows(root,window_samples=16,window_offset=8,step=2)
                np.testing.assert_array_equal(data.acquisition_numbers,[1,3,5,7,9])
                np.testing.assert_allclose(np.diff(data.times),120.)
                original=np.loadtxt(root/'Bearing1_1/7.csv',delimiter=',',skiprows=1)
                np.testing.assert_array_equal(data.values[3].reshape(16,2),original[8:24])
                self.assertGreater(data.acquisition_rms[-1,0],3*data.acquisition_rms[0,0])
                (root/'Bearing1_1/3.csv').unlink()
                with self.assertRaisesRegex(FileNotFoundError,'gaps'):
                    xjtu_sy.load_xjtu_sy_windows(root,window_samples=16,step=2)
                with self.assertRaisesRegex(ValueError,'inside one acquisition'):
                    xjtu_sy.load_xjtu_sy_windows(root,window_samples=16,window_offset=60)

    def test_download_member_crc_cache_checksum_and_selected_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=fixture(root)
            buffer=io.BytesIO()
            with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
                for file in (root/'Bearing1_1').glob('*.csv'):
                    bundle.writestr('35Hz12kN/Bearing1_1/'+file.name,file.read_bytes())
            archive=buffer.getvalue()
            target=root/'cache'
            with patch.object(xjtu_sy,'XJTU_RECORDS',records),patch.object(xjtu_sy,'XJTU_SAMPLES',64), \
                 patch.object(xjtu_sy,'_RemoteZip',side_effect=lambda *args:io.BytesIO(archive)):
                folder=xjtu_sy.download_xjtu_sy(target,max_series=3)
                self.assertEqual(sorted(p.name for p in folder.glob('*.csv')),['1.csv','2.csv','3.csv'])
                manifest=json.loads((folder/'download_manifest.json').read_text())
                self.assertEqual(len(manifest['files']),3)
                with patch.object(xjtu_sy,'_RemoteZip',side_effect=AssertionError('Must use cache')):
                    xjtu_sy.download_xjtu_sy(target,max_series=3)
                (folder/'1.csv').write_text('corrupted')
                with self.assertRaisesRegex(ValueError,'checksum'):
                    xjtu_sy.download_xjtu_sy(target,max_series=3)

    def test_full_archive_response_is_rejected_before_read(self):
        class Response:
            status=200
            headers={}
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,*args): raise AssertionError('Must not read a full archive response')
        with patch.object(xjtu_sy,'urlopen',return_value=Response()):
            with self.assertRaisesRegex(ValueError,'refusing full archive'):
                xjtu_sy._RemoteZip('https://example.invalid/data.zip',100).read(10)


class XJTUExperimentTests(unittest.TestCase):
    def test_pca_shared_decoder_artifacts_and_resume_without_ae_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=fixture(root)
            with patch.object(xjtu_sy,'XJTU_RECORDS',records),patch.object(xjtu_sy,'XJTU_SAMPLES',64):
                args=build_parser(representation='pca').parse_args([
                    '--data-dir',str(root),'--output-dir',str(root/'pca'),
                    '--records','Bearing1_1','Bearing1_4','--window-samples','16','--components','2',
                    '--calibration-samples','16','--iterations','2','--exploration-steps','1',
                    '--max-clusters','2','--stability-samples','20',
                    '--configurations','independent','dynamic','--verbose','silent'])
                with patch('hdpgpc.experiments.xjtu_sy.estimate_autoencoder_observation',
                           side_effect=AssertionError('PCA must not train an AE')):
                    reports=run_xjtu_sy(args)
                self.assertEqual([r['status'] for r in reports],['ok']*4)
                decoders=[]
                for report in reports:
                    path=Path(report['output_directory'])
                    self.assertEqual(report['metadata']['representation'],'pca')
                    self.assertEqual(report['metrics'],{})
                    self.assertIn('pca_calibration.png',report['artifacts'])
                    self.assertIn('latent_trajectories.png',report['artifacts'])
                    self.assertNotIn('autoencoder_calibration.npz',report['artifacts'])
                    diagnostic=report['metadata']['diagnostics']
                    self.assertIn('pca_explained_variance',diagnostic)
                    self.assertNotIn('autoencoder_explained_variance',diagnostic)
                    with np.load(path/'pca_calibration.npz') as saved:
                        decoders.append(saved['C'])
                        np.testing.assert_allclose(saved['C'].T@saved['C'],np.eye(2),atol=1e-12)
                        self.assertGreater(saved['explained_variance_ratio'],0)
                        self.assertLessEqual(saved['explained_variance_ratio'],1)
                        self.assertNotIn('losses',saved.files)
                for C in decoders[1:]:np.testing.assert_array_equal(C,decoders[0])
                args.resume=True
                with patch('hdpgpc.experiments.xjtu_sy.calibrate_windows',
                           side_effect=AssertionError('Resume must skip PCA calibration')):
                    resumed=run_xjtu_sy(args)
                self.assertEqual([r['run_id'] for r in reports],[r['run_id'] for r in resumed])

    def test_shared_raw_calibration_unlabeled_artifacts_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=fixture(root)
            with ExitStack() as stack:
                stack.enter_context(patch.object(xjtu_sy,'XJTU_RECORDS',records))
                stack.enter_context(patch.object(xjtu_sy,'XJTU_SAMPLES',64))
                args=build_parser().parse_args(['--data-dir',str(root),'--output-dir',str(root/'runs'),
                    '--records','Bearing1_1','Bearing1_4','--window-samples','16','--components','2',
                    '--calibration-samples','16','--epochs','5','--iterations','2','--exploration-steps','1',
                    '--max-clusters','2','--stability-samples','20','--configurations','independent','dynamic',
                    '--verbose','silent'])
                reports=run_xjtu_sy(args)
                self.assertEqual([r['status'] for r in reports],['ok']*4)
                decoders=[]
                for report in reports:
                    path=Path(report['output_directory'])
                    self.assertFalse(report['reference_labels_available'])
                    self.assertEqual(report['metrics'],{})
                    self.assertFalse(report['metadata']['preliminary_projection'])
                    self.assertTrue(report['metadata']['complete_run_to_failure'])
                    for artifact in ['latent_trajectories.png','latent_embedding.png','prepared_data.npz','dynamics.json']:
                        self.assertIn(artifact,report['artifacts'])
                        self.assertTrue((path/artifact).is_file())
                    with np.load(path/'autoencoder_calibration.npz') as saved:
                        decoders.append(saved['C'])
                        self.assertLess(saved['losses'][-1],saved['losses'][0])
                    with (path/'assignments.csv').open(newline='') as stream:
                        rows=list(csv.DictReader(stream))
                    self.assertTrue(all(row['true_label']=='' for row in rows))
                    np.testing.assert_allclose(np.diff([float(row['second']) for row in rows]),60.)
                    self.assertIn('Results saved to:',(path/'clustering.log').read_text())
                for C in decoders[1:]:np.testing.assert_array_equal(C,decoders[0])
                self.assertTrue((root/'runs/comparison.png').is_file())
                args.resume=True
                with patch('hdpgpc.experiments.xjtu_sy.calibrate_windows',side_effect=AssertionError('Must skip training')), \
                     patch('hdpgpc.experiments.xjtu_sy.HDPGPC',side_effect=AssertionError('Must skip inference')):
                    resumed=run_xjtu_sy(args)
                self.assertEqual([r['run_id'] for r in reports],[r['run_id'] for r in resumed])

    def test_invalid_controls(self):
        from hdpgpc.experiments.xjtu_sy import _validate
        for flags in [['--normalize-samples'],['--step','0'],['--gamma-prior','nan'],
                      ['--window-samples','32768'],['--window-offset','32760']]:
            with self.subTest(flags=flags),self.assertRaises(ValueError):
                _validate(build_parser().parse_args(flags))


if __name__=='__main__':unittest.main()
