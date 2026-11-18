"""Constant-input configuration is optional and reaches every shared runner."""
from dataclasses import replace
import unittest

import numpy as np

from hdpgpc.config import HDPGPCConfig, GPEmissionConfig, DynamicsConfig, LDS_PRESETS, with_lds_preset
from hdpgpc.config._models import _initialization_options
from hdpgpc.emissions import AffineGaussianObservation


class ConstantInputConfigurationTests(unittest.TestCase):
    def test_old_recipes_roundtrip_without_changing_disabled_defaults(self):
        config = HDPGPCConfig()
        serialized = config.to_dict()
        self.assertNotIn('constant_input', serialized['dynamics'])
        self.assertNotIn('constant_input_strength', serialized['dynamics'])
        restored = HDPGPCConfig.from_dict(serialized)
        self.assertFalse(restored.dynamics.constant_input)
        self.assertEqual(restored.dynamics.constant_input_strength, 1.)
        self.assertEqual(restored.to_dict(), serialized)
        config = replace(config, dynamics=DynamicsConfig(constant_input=True, constant_input_strength=3.))
        self.assertEqual(HDPGPCConfig.from_yaml(config.to_yaml()), config)

    def test_validation_and_initialization_options(self):
        config = HDPGPCConfig(x_basis=np.arange(3.)[:, None],
                             emission=GPEmissionConfig(ini_lengthscale=1., bound_lengthscale=(.1, 10.),
                                                       ini_outputscale=1.),
                             dynamics=DynamicsConfig(constant_input=True, constant_input_strength=.5))
        options = _initialization_options(config)
        self.assertTrue(options['constant_input'])
        self.assertEqual(options['constant_input_strength'], .5)
        for field, values in (('constant_input', (1, 'true', None)),
                              ('constant_input_strength', (0., -1., float('inf'), float('nan'), True))):
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(config, dynamics=replace(config.dynamics, **{field: value})).validate()

    def test_presets_preserve_explicit_constant_input_controls(self):
        base = HDPGPCConfig(dynamics=DynamicsConfig(constant_input=True, constant_input_strength=2.))
        for preset in LDS_PRESETS:
            with self.subTest(preset=preset):
                config = with_lds_preset(base, preset)
                self.assertTrue(config.dynamics.constant_input)
                self.assertEqual(config.dynamics.constant_input_strength, 2.)

    def test_aliases_and_overrides_in_all_runner_families(self):
        from hdpgpc.experiments.records import build_parser as records_parser
        from hdpgpc.experiments.ucr import build_parser as ucr_parser
        from hdpgpc.experiments._expression import build_parser as expression_parser
        from hdpgpc.experiments.flutter import build_parser as flutter_parser
        from hdpgpc.experiments.xjtu_sy import build_parser as xjtu_parser
        from hdpgpc.experiments.cas_colon import build_parser as colon_parser
        from hdpgpc.experiments.synthetic_image_clustering import build_parser as synthetic_parser
        parsers = [records_parser(representation=rep) for rep in ('waveform', 'pca', 'autoencoder')]
        parsers += [ucr_parser(rep) for rep in ('waveform', 'pca', 'autoencoder', 'fourier')]
        parsers += [expression_parser(representation=rep) for rep in ('mfa', 'pca', 'autoencoder')]
        parsers += [flutter_parser(), synthetic_parser()]
        parsers += [xjtu_parser(representation=rep) for rep in ('autoencoder', 'pca', 'fourier', 'wavelet')]
        parsers += [colon_parser(representation=rep)
                    for rep in ('pca', 'autoencoder', 'sfa', 'linear_prediction', 'mfa')]
        for parser in parsers:
            with self.subTest(parser=parser.prog):
                self.assertFalse(parser.parse_args([]).constant_input)
                self.assertTrue(parser.parse_args(['--constant-input']).constant_input)
                self.assertTrue(parser.parse_args(['--latent-equilibrium']).constant_input)
                self.assertFalse(parser.parse_args(['--latent-equilibrium', '--no-constant-input']).constant_input)
                args = parser.parse_args(['--constant-input', '--no-latent-equilibrium',
                                          '--latent-equilibrium-strength', '2.5'])
                self.assertFalse(args.constant_input)
                self.assertEqual(args.constant_input_strength, 2.5)

    def test_runner_builders_pass_constant_input_to_native_recipe(self):
        from hdpgpc.experiments import records, ucr, _affine, flutter, synthetic_image_clustering
        grid = np.arange(4.)[:, None]
        decoder = AffineGaussianObservation(np.eye(4, 2), np.eye(4), grid, np.zeros(4), np.eye(2))
        flags = ['--latent-equilibrium', '--constant-input-strength', '4.']
        configs = [records.make_config(grid, records.build_parser().parse_args(flags)),
                   records.make_config(grid, records.build_parser(representation='autoencoder').parse_args(flags),
                                       decoder=decoder),
                   ucr.make_config(grid, ucr.build_parser('waveform').parse_args(flags)),
                   _affine.make_config(flutter.build_parser().parse_args(flags), decoder, dynamic=True),
                   synthetic_image_clustering.make_config(
                       synthetic_image_clustering.build_parser().parse_args(flags), decoder, .1)]
        for config in configs:
            self.assertTrue(config.dynamics.constant_input)
            self.assertEqual(config.dynamics.constant_input_strength, 4.)


if __name__ == '__main__':
    unittest.main()
