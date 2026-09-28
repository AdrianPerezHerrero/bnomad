"""Installed metadata, resources and public-only tutorial integration checks."""
from importlib import metadata
from importlib.resources import files
import os
from pathlib import Path
import unittest

from hdpgpc.examples import MODES, configuration
from hdpgpc.examples.reference import check_reference, run_experiment


class ReleaseTests(unittest.TestCase):
    def test_packaged_recipes_round_trip(self):
        from hdpgpc import HDPGPCConfig
        for mode in MODES:
            config = configuration(mode)
            self.assertEqual(config.to_dict(), HDPGPCConfig.from_dict(config.to_dict()).to_dict())
        self.assertTrue(files('hdpgpc.examples').joinpath('reference_offline.json').is_file())

    def test_public_examples_and_repeatability(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                first = run_experiment(mode)
                second = run_experiment(mode)
                self.assertEqual(first['scientific_result'], second['scientific_result'])
                result = first['scientific_result']
                # This dynamic tutorial retains one cluster online; the static
                # growth case is separately protected by the original baselines.
                self.assertEqual(result['labels'], [0, 0, 0, 0] if mode == 'online' else [0, 0, 1, 1, 0, 1])
                self.assertEqual(result['n_clusters'], 1 if mode == 'online' else 2)
                if mode == 'offline':
                    check_reference(first)
                if mode == 'warping':
                    self.assertGreater(max(abs(x[0]) for x in result['alignment_demo']['offsets']), 1e-3)

    @unittest.skipUnless(os.environ.get('HDPGPC_TEST_INSTALLED'), 'installed-artifact gate')
    def test_installed_distribution(self):
        import hdpgpc
        location = Path(hdpgpc.__file__).resolve()
        self.assertIn('site-packages', location.parts)
        dist = metadata.distribution('hdpgpc')
        self.assertEqual(dist.version, '0.3.0')
        self.assertEqual(dist.metadata['License-Expression'], 'GPL-3.0-only')
        names = [str(p).replace('\\', '/') for p in dist.files]
        self.assertTrue(any(p.endswith('/licenses/LICENSE') for p in names))
        self.assertTrue(any(p.endswith('/licenses/licenses/bnpy.txt') for p in names))
        self.assertTrue(any(e.name == 'hdpgpc-reference' for e in dist.entry_points))
        self.assertFalse(any('/data/' in p or '/results/' in p for p in names))
