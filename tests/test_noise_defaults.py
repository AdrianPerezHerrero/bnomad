"""Uniform omitted noise priors, explicit overrides and calibrated scaling."""
from pathlib import Path
import runpy
import unittest

import numpy as np
from sklearn.gaussian_process.kernels import RBF
import torch

from hdpgpc.config import (HDPGPCConfig, GPEmissionConfig, DynamicsConfig,
                           HDPConfig, AlignmentConfig)
from hdpgpc.dynamics import LinearDynamics, StaticDynamics
from hdpgpc.emissions import AffineGaussianObservation
from hdpgpc.experiments import (records, ucr, cas_colon, zebrafish, xenopus,
                                flutter, xjtu_sy, synthetic_image_clustering)


class NoiseDefaultTests(unittest.TestCase):
    def test_core_and_low_level_defaults_and_explicit_covariances(self):
        self.assertEqual(GPEmissionConfig().ini_sigma, .1)
        self.assertEqual(DynamicsConfig().ini_gamma, .1)
        restored = HDPGPCConfig.from_dict({'schema_version': 1, 'emission': {}, 'dynamics': {}})
        self.assertEqual(restored.emission.ini_sigma, .1)
        self.assertEqual(restored.dynamics.ini_gamma, .1)
        grid = np.arange(3.)[:, None]
        config = HDPGPCConfig(x_basis=grid, hdp=HDPConfig(M=1, max_models=2),
            emission=GPEmissionConfig(kernels=[RBF(1.)], fit_gp=False),
            dynamics=DynamicsConfig(annealing=False), alignment=AlignmentConfig(policy='none'))
        state = config.build_model()
        emission = state.gpmodels[0][0]
        torch.testing.assert_close(emission.Sigma[0], .1*torch.eye(3))
        torch.testing.assert_close(emission.Gamma[0], .1*torch.eye(3))
        for policy in (LinearDynamics(), StaticDynamics()):
            with self.subTest(policy=type(policy).__name__):
                dynamic = type(policy) is LinearDynamics
                _, gamma, _, sigma = policy.initial_parameters(grid)
                torch.testing.assert_close(sigma, .1*torch.eye(3))
                torch.testing.assert_close(gamma, (.1 if dynamic else 0.)*torch.eye(3))
                kwargs = {'gamma': .3} if dynamic else {}
                _, gamma, _, sigma = policy.initial_parameters(grid, sigma=.7, **kwargs)
                torch.testing.assert_close(sigma, .7*torch.eye(3))
                torch.testing.assert_close(gamma, (.3 if kwargs else 0.)*torch.eye(3))

    def test_all_dataset_parsers_and_script_overrides(self):
        parsers = []
        for dataset in ('mitbih', 'daphnet'):
            for representation in ('waveform', 'pca', 'autoencoder'):
                parsers.append((dataset+'/'+representation, records.build_parser(dataset, representation)))
        for representation in ucr.REPRESENTATIONS:
            parsers.append(('ucr/'+representation, ucr.build_parser(representation)))
        for representation in ('pca', 'autoencoder', 'sfa', 'linear_prediction', 'mfa'):
            parsers.append(('cas/'+representation, cas_colon.build_parser(representation=representation)))
        for module in (zebrafish, xenopus):
            for representation in ('mfa', 'pca', 'autoencoder'):
                parsers.append((module.__name__+'/'+representation, module.build_parser(representation=representation)))
        for representation in ('pca', 'autoencoder', 'fourier', 'wavelet'):
            parsers.append(('xjtu/'+representation, xjtu_sy.build_parser(representation=representation)))
        parsers.extend([('flutter', flutter.build_parser()), ('synthetic', synthetic_image_clustering.build_parser())])
        root = Path(__file__).resolve().parents[1]
        for name in ('xenopus_autoencoder_clustering', 'mitbih_autoencoder_clustering'):
            script = runpy.run_path(str(root/'examples'/f'{name}.py'))
            parser = (xenopus.build_parser(defaults=script['DEFAULTS'], representation='autoencoder')
                      if name.startswith('xenopus') else records.build_parser('mitbih', 'autoencoder', defaults=script['DEFAULTS']))
            parsers.append((name, parser))
        for name, parser in parsers:
            with self.subTest(entry_point=name):
                args = parser.parse_args([])
                self.assertEqual((args.sigma_prior, args.gamma_prior), (.1, .1))
                args = parser.parse_args(['--sigma-prior', '.23', '--gamma-prior', '.37'])
                self.assertEqual((args.sigma_prior, args.gamma_prior), (.23, .37))

    def test_calibrated_observation_noise_is_scaled_once(self):
        from hdpgpc.experiments._affine import make_config
        grid = np.arange(4.)[:, None]
        decoder = AffineGaussianObservation(np.ones((4, 1)), np.eye(4)*2.,
            grid, np.zeros(4), np.eye(1))
        args = zebrafish.build_parser(representation='pca').parse_args([])
        config = make_config(args, decoder, dynamic=True)
        np.testing.assert_allclose(config.observation.Sigma_policy.value, np.eye(4)*.2)
        self.assertEqual(config.dynamics.ini_gamma, .1)
        self.assertEqual(config.hdp.hdp_hyp, 'balanced')
        self.assertIsNone(config.hdp.gamma)  # HDP concentration is a separate parameter.
        args.sigma_prior, args.gamma_prior = .3, .4
        config = make_config(args, decoder, dynamic=True)
        np.testing.assert_allclose(config.observation.Sigma_policy.value, np.eye(4)*.6)
        self.assertEqual(config.dynamics.ini_gamma, .4)


if __name__ == '__main__':
    unittest.main()
