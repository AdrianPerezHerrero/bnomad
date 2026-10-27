"""Temporal baseline semantics and CAS reporting/header integration."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hdpgpc.experiments._baselines import equal_duration_baseline, add_baseline_header
from hdpgpc.experiments._runs import save_results


class EqualDurationBaselineTests(unittest.TestCase):
    def test_balanced_contiguous_bins_and_perfect_majorities(self):
        labels = np.array(['A']*4+['B']*3+['C']*3)
        b = equal_duration_baseline(labels, 3)
        np.testing.assert_array_equal(b['assignments'], [0]*4+[1]*3+[2]*3)
        np.testing.assert_array_equal(b['majority_labels'], labels)
        self.assertEqual(b['clusters'], 3)
        self.assertEqual([s['frames'] for s in b['segments']], [4, 3, 3])
        for name in ('purity','adjusted_rand_index','normalized_mutual_information'):
            self.assertEqual(b['metrics'][name], 1.)
        for n in range(1, 15):
            for k in range(1, n+1):
                r = equal_duration_baseline(np.repeat('A', n), k)
                counts = np.bincount(r['assignments'])
                self.assertEqual(len(counts), k)
                self.assertLessEqual(counts.max()-counts.min(), 1)
                self.assertTrue((np.diff(r['assignments']) >= 0).all())

    def test_ari_uses_bins_not_majority_merged_labels(self):
        b = equal_duration_baseline(np.array(['A','A','B','B']*2), 2)
        self.assertEqual(b['cluster_majority_labels'], {0:'A', 1:'A'})
        self.assertEqual(b['metrics']['purity'], .5)
        self.assertAlmostEqual(b['metrics']['adjusted_rand_index'], -1/6)
        self.assertAlmostEqual(b['metrics']['normalized_mutual_information'], 0.)

    def test_excluded_labels_occupy_time_but_do_not_fit_majorities(self):
        b = equal_duration_baseline(['ambiguous']*4+['B']*4, 2,
                                   evaluation_mask=[False]*4+[True]*4)
        np.testing.assert_array_equal(b['assignments'], [0]*4+[1]*4)
        self.assertEqual(b['cluster_majority_labels'], {0:None,1:'B'})
        self.assertEqual(b['metrics']['purity'], 1.)
        self.assertEqual(b['metrics']['evaluation_frames'], 4)
        self.assertEqual(b['metrics']['excluded_evaluation_frames'], 4)
        self.assertTrue(all(v is None for v in b['majority_labels'][:4]))
        for labels, k, mask in (([],1,None), (['A'],0,None), (['A'],2,None),
                                (['A'],True,None), (['A'],1,[False]), (['A'],1,[True,False])):
            with self.assertRaises(ValueError):
                equal_duration_baseline(labels,k,evaluation_mask=mask)

    def test_existing_metric_header_retains_position_and_adds_one_baseline(self):
        b = equal_duration_baseline(['A']*4, 2)
        fig = plt.figure()
        try:
            fig.suptitle('Record | Purity 1.000 | ARI 1.000 | NMI 1.000\nDetails', y=.93)
            top_margin = .07 * fig.get_size_inches()[1]
            add_baseline_header(fig,b['metrics'],b)
            lines = fig._suptitle.get_text().splitlines()
            self.assertIn('Equal-duration baseline (K=2)',lines[1])
            self.assertEqual(lines[-1], 'Details')
            self.assertAlmostEqual((1-fig._suptitle.get_position()[1])*fig.get_size_inches()[1], top_margin)
        finally:
            plt.close(fig)

    def test_reporting_saves_baseline_and_headers_without_changing_model_metrics(self):
        labels = np.array(['A']*3+['B']*5)
        assignments = np.array([0]*3+[1]*5)
        gp = SimpleNamespace(A=[np.eye(1)],C=[np.ones((4,1))],
                             Gamma=[np.eye(1)],Sigma=[np.eye(4)])
        model = SimpleNamespace(labels_=assignments, n_clusters_=2, cluster_ids_=np.array([0,1]),
            _backend=SimpleNamespace(gpmodels=[[gp,gp]]),
            config=SimpleNamespace(inference=SimpleNamespace(warp=False), to_yaml=lambda:'{}'))
        def figure(title):
            f=plt.figure();f.suptitle(title);return f
        def summary(*args, **kwargs):
            return [figure('Model gallery | Purity 1.000 | ARI 1.000 | NMI 1.000')]
        args=SimpleNamespace(plots=True, projection_plots=False, lds_plots=True,
            lds_dynamics_plots=False,warp_plots=False,birth_plots=0,max_clusters=5,min_cluster_samples=1)
        for enabled, plots in ((True,True),(True,False),(False,True)):
            with self.subTest(enabled=enabled,plots=plots), tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp);headers={}
                run=SimpleNamespace(dataset='video_test',run_id='test',experiment='cas_colon_mfa',
                                    path=path,started=time.perf_counter())
                def savefig(f,file,**kwargs):
                    headers[Path(file).name]=f._suptitle.get_text()
                    Path(file).touch()
                args.plots=plots
                with patch('matplotlib.figure.Figure.savefig',new=savefig):
                    report=save_results(run,model,np.ones((8,4,1)),labels,np.arange(4.)[:,None],args,
                        ordered=True,summary_plotter=summary, extra_figures=lambda:{'latent_trajectories':figure('States')},
                        row_metadata={'second':np.arange(8)},equal_duration_baseline=enabled)
                self.assertEqual(report['metrics']['purity'],1.)
                if enabled:
                    baseline=report['baselines']['equal_duration']
                    self.assertEqual(baseline['clusters'],2)
                    self.assertEqual(baseline['metrics']['purity'],7/8)
                    saved=json.loads((path/'equal_duration_baseline.json').read_text())
                    self.assertEqual(saved['assignments'],[0]*4+[1]*4)
                    with (path/'equal_duration_assignments.csv').open() as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))),8)
                    if plots:
                        self.assertIn('equal_duration_baseline.png',headers)
                        self.assertIn('latent_trajectories.png',headers)
                        self.assertIn('lds_parameters_output_0_01.png',headers)
                        self.assertTrue(all('Equal-duration baseline (K=2)' in h for h in headers.values()))
                    else:
                        self.assertFalse(headers)
                    self.assertIn('equal_duration_baseline.json',report['artifacts'])
                else:
                    self.assertNotIn('baselines',report)
                    self.assertFalse((path/'equal_duration_baseline.json').exists())
                    self.assertTrue(all('Equal-duration baseline' not in h for h in headers.values()))


if __name__=='__main__':
    unittest.main()
