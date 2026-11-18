"""Independent state/pair invariants across hard inference and overrides."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.gaussian_process.kernels import RBF

from hdpgpc import HDPGPC
from hdpgpc.config import HDPConfig
from hdpgpc.inference import _assignments, _messages, _online, _ordered_messages
from test_dynamics_clock import recipe


class AssignmentConsistencyTests(unittest.TestCase):
    def assert_chain(self, states, pairs):
        self.assertEqual(pairs.dtype, states.dtype)
        self.assertEqual(pairs.device, states.device)
        self.assertEqual(pairs.shape, (len(states), states.shape[1], states.shape[1]))
        torch.testing.assert_close(states.sum(1), states.new_ones(len(states)), rtol=0, atol=0)
        torch.testing.assert_close(pairs[0], torch.zeros_like(pairs[0]), rtol=0, atol=0)
        torch.testing.assert_close(pairs[1:].sum(2), states[:-1], rtol=0, atol=0)
        torch.testing.assert_close(pairs[1:].sum(1), states[1:], rtol=0, atol=0)
        self.assertEqual(float(pairs.sum()), max(0, len(states)-1))

    def owner(self, hard=True):
        owner = SimpleNamespace(hard_assignments=hard)
        owner._safe_exp = lambda values, hard=None: _messages._safe_exp(owner, values, hard)
        return owner

    def test_ambiguous_joint_uses_the_committed_state_sequence(self):
        joint = torch.tensor([[.30, .31], [.29, .10]], dtype=torch.float64)
        state_log = torch.stack((joint.sum(1), joint.sum(0))).log()
        pair_log = torch.stack((torch.full_like(joint, -torch.inf), joint.log()))
        states, pairs = _assignments.from_log_marginals(self.owner(), state_log, pair_log)
        self.assertEqual(states.argmax(1).tolist(), [0, 0])
        self.assertEqual(pairs[1].reshape(-1).argmax().item(), 0)
        self.assert_chain(states, pairs)
        # The independent pair maximum demonstrates why separate hardening fails.
        self.assertEqual(joint.reshape(-1).argmax().item(), 1)

    def test_native_local_terms_commit_matching_pairs_for_an_ambiguous_hmm(self):
        from hdpgpc.hdp import HDPVariational
        cfg = recipe('cluster-clock', fixed=False, hard=True)
        cfg = replace(cfg, hdp=HDPConfig(M=2, max_models=2),
                      emission=replace(cfg.emission, kernels=[RBF(), RBF()]))
        state = cfg.build_model()
        state.snr_norm = torch.ones(2, 1)
        joint = torch.tensor([[.30, .31], [.29, .10]], dtype=torch.float64)
        start = joint.sum(1)
        transition = joint/start[:, None]
        with patch.object(HDPVariational, 'expected_log_start', return_value=start.log()), \
             patch.object(HDPVariational, 'expected_log_transitions', return_value=transition.log()):
            resp, _, pair, log_pair = state.variational_local_terms(torch.zeros(2, 2, 1))
        self.assertEqual(resp.argmax(1).tolist(), [0, 0])
        self.assertEqual(log_pair[1].reshape(-1).argmax().item(), 1)
        self.assertEqual(pair[1].reshape(-1).argmax().item(), 0)
        self.assert_chain(resp, pair)

    def test_soft_marginals_and_explicit_overrides_keep_existing_values(self):
        joint = torch.tensor([[.30, .31], [.29, .10]], dtype=torch.float64)
        states_log = torch.stack((joint.sum(1), joint.sum(0))).log()
        pairs_log = torch.stack((torch.full_like(joint, -torch.inf), joint.log()))
        owner = self.owner()
        states, pairs = _assignments.from_log_marginals(owner, states_log, pairs_log, hard=False)
        torch.testing.assert_close(states, owner._safe_exp(states_log, hard=False), rtol=0, atol=0)
        torch.testing.assert_close(pairs, owner._safe_exp(pairs_log, hard=False), rtol=0, atol=0)
        torch.testing.assert_close(pairs[1].sum(1), states[0])
        torch.testing.assert_close(pairs[1].sum(0), states[1])
        states, pairs = _assignments.from_log_marginals(self.owner(False), states_log, pairs_log, hard=True)
        self.assert_chain(states, pairs)

    def test_forcing_appending_and_permuting_preserve_the_chain(self):
        devices = ['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    ref = torch.empty(0, device=device, dtype=dtype)
                    states, pairs = _assignments.from_labels([0, 1, 0], 3, ref)
                    before = states.clone(), pairs.clone()
                    extended, incoming = _assignments.append_hard(states, pairs, 2, 4)
                    self.assert_chain(extended, incoming)
                    self.assertEqual(extended.argmax(1).tolist(), [0, 1, 0, 2])
                    torch.testing.assert_close(states, before[0], rtol=0, atol=0)
                    torch.testing.assert_close(pairs, before[1], rtol=0, atol=0)
                    _assignments.force_last(extended, incoming, 1)
                    self.assert_chain(extended, incoming)
                    permutation = torch.tensor([3, 1, 0, 2], device=device)
                    self.assert_chain(extended[:, permutation], incoming[:, permutation][:, :, permutation])
                    states, pairs = _assignments.from_labels([0], 2, ref)
                    _assignments.force_last(states, pairs, 1)
                    self.assert_chain(states, pairs)
                    states, pairs = _assignments.append_hard(states, None, 0, 2)
                    self.assert_chain(states, pairs)

    def test_loaded_hard_cache_is_repaired_once(self):
        states, pairs = _assignments.from_labels([0, 1, 0], 2, torch.empty(0, dtype=torch.float64))
        owner = SimpleNamespace(device='cpu', hard_assignments=True, M=2, T=3,
            resp_last=states, respPair_last=torch.zeros_like(pairs),
            q_last=torch.zeros(3, 2, 1), q_lat_last=torch.zeros(3, 2, 1))
        _online._ensure_fast_cache_online(owner)
        self.assert_chain(owner.resp_last, owner.respPair_last)
        with patch.object(_online, 'hard_pairs', side_effect=AssertionError('Repeated cache reconstruction')):
            _online._ensure_fast_cache_online(owner)

    def test_ordered_decoding_preserves_a_feasible_joint_path(self):
        initial = torch.tensor([0., -torch.inf], dtype=torch.float64)
        transitions = torch.tensor([[.6, .4], [0., 1.]], dtype=torch.float64).log()
        unary = torch.tensor([[.8, .2], [.5, .5], [.1, .9]], dtype=torch.float64).log()
        states, pairs, path = _ordered_messages.marginals(initial, transitions, unary, hard=True)
        self.assert_chain(states, pairs)
        self.assertEqual(path[0].item(), 0)
        self.assertTrue(bool(((path[1:]-path[:-1] >= 0) & (path[1:]-path[:-1] <= 1)).all()))

    def test_fitting_strategies_clocks_and_emissions_commit_coherent_pairs(self):
        values = np.array([[1., .3], [.9, .2], [-1., -.3], [-.8, -.2]]*2)
        for clock in ('cluster-clock', 'physical-clock'):
            for strategy in ('legacy', 'split_merge'):
                for affine in (False, True):
                    with self.subTest(clock=clock, strategy=strategy, affine=affine), redirect_stdout(io.StringIO()):
                        cfg = recipe(clock, fixed=False, hard=True)
                        cfg = replace(cfg, inference=replace(cfg.inference, offline_strategy=strategy))
                        if not affine:
                            from hdpgpc.config import ObservationConfig, DynamicsConfig
                            cfg = replace(cfg, x_basis=np.arange(2.)[:, None], observation=ObservationConfig(),
                                dynamics=DynamicsConfig(clock=clock, annealing=False, lds_update='conjugate_batch'))
                        model = HDPGPC(cfg).fit(values, grid=np.arange(2.))
                        state = model._backend
                        self.assert_chain(state.resp_last, state.respPair_last)
                        torch.testing.assert_close(state.transStateCount_last,
                                                   state.respPair_last[1:].sum(0))

    def test_online_forced_births_change_pairs_before_learning(self):
        for clock in ('cluster-clock', 'physical-clock'):
            for schedule in ('online', 'online_fast'):
                for hmm in (False, True):
                    with self.subTest(clock=clock, schedule=schedule, hmm=hmm), redirect_stdout(io.StringIO()):
                        cfg = recipe(clock, schedule=schedule, update='conjugate_sequential', fixed=False, hard=True)
                        cfg = replace(cfg, hdp=HDPConfig(max_models=2, hmm_switch=hmm))
                        state = cfg.build_model()
                        op = state.online.update_fast if schedule == 'online_fast' else state.online.update
                        seen = []
                        original = type(state.hdp).transition_counts
                        def counts(owner, pair):
                            if owner._state is state:
                                seen.append(pair.detach().clone())
                            return original(owner, pair)
                        with patch.object(type(state.hdp), 'transition_counts', new=counts):
                            for chosen in (0, 1, 0, 1):
                                op(np.arange(2.)[:, None], np.array([[1.], [.3]]), force_model=chosen)
                                self.assertEqual(state.actual_state, chosen)
                                self.assertEqual(state.resp_assigned[-1][-1].item(), chosen)
                                if schedule == 'online_fast':
                                    self.assert_chain(state.resp_last, state.respPair_last)
                        if hmm:
                            labels = state.resp_assigned[-1]
                            expected, pair = _assignments.from_labels(labels, seen[-1].shape[1], seen[-1])
                            torch.testing.assert_close(seen[-1], pair, rtol=0, atol=0)
                        if schedule == 'online_fast':
                            self.assertEqual(state.resp_assigned[-1].tolist(), [0, 1, 0, 1])


if __name__ == '__main__':
    unittest.main()
