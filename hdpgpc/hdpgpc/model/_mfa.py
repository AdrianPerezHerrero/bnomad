"""Public DP/HDP-HMM-MFA facade, without GP trajectory compatibility state."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from numbers import Integral
from pathlib import Path
from os import PathLike
from typing import Any, Iterable, Literal, Mapping

from numpy.typing import ArrayLike, NDArray

import numpy as np
import torch

from ..config._mfa import HDPMFAConfig
from ..emissions import FactorAnalysisEmission
from ..hdp import HDPVariational
from ..inference._independent_assignments import _iid_messages
from ..inference._mfa import MFAOfflineInference
from ..inference._mfa_assignments import markov_assignments
from ..hdp._markov import markov_counts
from ._mfa_state import MFAState
from ._mfa_results import MFAFactors


def _snapshot(value):
    result = np.array(value.detach().cpu().numpy() if torch.is_tensor(value) else value, copy=True)
    result.setflags(write=False)
    return result


class HDPMFA:
    """Offline variational DP/HDP-HMM-MFA with normalized soft assignments.

    hdp.hmm_switch selects independent or Markov assignments. Explicit sequence
    lengths restart the HMM. Optional birth/pruning is fully objective-gated.
    `fit` replaces state transactionally; this is not streaming/partial fitting.
    """

    def __init__(self, config: HDPMFAConfig | None = None) -> None:
        if config is None:
            config = HDPMFAConfig()
        if type(config) is not HDPMFAConfig:
            raise TypeError('config must be HDPMFAConfig')
        self._config = deepcopy(config.validate())
        self._state = None

    @property
    def config(self) -> HDPMFAConfig:
        return deepcopy(self._config)

    def _require_fitted(self):
        if self._state is None:
            raise RuntimeError('Fit the DP-MFA model before reading results or predicting')
        return self._state

    def _values(self, observations, *, fitting=False):
        device = torch.device('cuda' if self._config.numerics.cuda else 'cpu')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA requested but unavailable')
        complex_values = observations.is_complex() if torch.is_tensor(observations) else np.iscomplexobj(observations)
        if complex_values:
            raise ValueError('observations must be real-valued')
        try:
            values = torch.as_tensor(observations, dtype=torch.float64, device=device).detach()
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError('observations must be a real numeric matrix (N,d)') from error
        if (values.layout != torch.strided or values.ndim != 2 or values.shape[1] == 0
                or (fitting and len(values) == 0) or not bool(torch.isfinite(values).all())):
            raise ValueError('observations must be a finite matrix (N,d), nonempty for fitting')
        if fitting:
            if self._config.n_factors > values.shape[1]:
                raise ValueError('n_factors cannot exceed observation dimension')
            if self._config.hdp.M > len(values):
                raise ValueError('hdp.M cannot exceed the number of fitting observations')
        elif values.shape[1] != self._require_fitted().observation_dim:
            raise ValueError('observation dimension differs from the fitted model')
        return values

    @staticmethod
    def _lengths(lengths, count):
        if lengths is None:
            return (count,)
        try:
            lengths = tuple(lengths)
        except TypeError as error:
            raise ValueError('sequence_lengths must be positive integers summing to N') from error
        if (not lengths or any(isinstance(value, bool) or not isinstance(value, Integral) or value < 1
                               for value in lengths) or sum(lengths) != count):
            raise ValueError('sequence_lengths must be positive integers summing to N')
        return tuple(int(value) for value in lengths)

    @torch.no_grad()
    def fit(self, observations: ArrayLike | torch.Tensor,
            sequence_lengths: Iterable[int] | None = None) -> HDPMFA:
        values = self._values(observations, fitting=True)
        lengths = self._lengths(sequence_lengths, len(values))
        self._config.numerics.apply()
        candidate = MFAState(self._config, values, lengths)
        MFAOfflineInference(candidate).fit(values)
        self._state = candidate
        return self

    def fit_predict(self, observations: ArrayLike | torch.Tensor,
                    sequence_lengths: Iterable[int] | None = None) -> NDArray[np.int64]:
        """Fit a fresh model and return its stored training MAP assignments."""
        return self.fit(observations, sequence_lengths).labels_

    @property
    def n_features_in_(self) -> int:
        return self._require_fitted().observation_dim

    @property
    def n_factors_(self) -> int:
        return self._require_fitted().config.n_factors

    @property
    def responsibilities_(self) -> NDArray[np.float64]:
        """Training q(s), from the final coordinate sweep (not hard assignments)."""
        return _snapshot(self._require_fitted().resp)

    @property
    def labels_(self) -> NDArray[np.int64]:
        return _snapshot(self._require_fitted().resp.argmax(1))

    @property
    def occupancies_(self) -> NDArray[np.float64]:
        return _snapshot(self._require_fitted().resp.sum(0))

    @property
    def component_ids_(self) -> NDArray[np.int64]:
        """Current support slots; pruning compacts IDs, so they are not persistent."""
        return _snapshot(np.arange(self._require_fitted().M, dtype=np.int64))

    @property
    def cluster_ids_(self) -> NDArray[np.int64]:
        """MAP-occupied current slots (fixed strategy retains unused slots)."""
        return _snapshot(np.unique(self.labels_))

    @property
    def n_components_(self) -> int:
        return self._require_fitted().M

    @property
    def n_clusters_(self) -> int:
        return len(self.cluster_ids_)

    @property
    def means_(self) -> NDArray[np.float64]:
        return _snapshot(torch.stack([component.mean for component in self._require_fitted().components]))

    @property
    def loadings_(self) -> NDArray[np.float64]:
        return _snapshot(torch.stack([component.loadings for component in self._require_fitted().components]))

    @property
    def parameter_covariances_(self) -> NDArray[np.float64]:
        return _snapshot(torch.stack([component.parameter_covariance
                                     for component in self._require_fitted().components]))

    @property
    def noise_(self) -> NDArray[np.float64]:
        """One learned/fixed diagonal variance vector shared by every component."""
        return _snapshot(self._require_fitted().noise)

    @property
    def components_(self) -> tuple[FactorAnalysisEmission, ...]:
        state = self._require_fitted()
        return tuple(component.copy().set_noise(state.noise) for component in state.components)

    @property
    def stick_parameters_(self) -> dict[str, NDArray[np.float64]]:
        state = self._require_fitted()
        return {'a': _snapshot(state.rho * state.omega),
                'b': _snapshot((1 - state.rho) * state.omega)}

    def _stick_weights(self):
        state = self._require_fitted()
        return state.hdp.rho2beta(state.rho.detach().cpu().numpy(), returnSize='K+1')

    @property
    def tail_mass_(self) -> float:
        """Posterior mean mass beyond the K represented components."""
        return float(self._stick_weights()[-1])

    @property
    def weights_(self) -> NDArray[np.float64]:
        """Posterior mean stick weights, CONDITIONED on represented components."""
        weights = self._stick_weights()[:-1]
        return _snapshot(weights / weights.sum())

    @property
    def elbo_(self) -> float:
        return self._require_fitted().elbo_parts['total']

    @property
    def elbo_components_(self) -> dict[str, float]:
        return self._require_fitted().elbo_parts.copy()

    @property
    def elbo_history_(self) -> NDArray[np.float64]:
        """Initial ELBO followed by each complete sweep's ELBO."""
        return _snapshot(self._require_fitted().elbo_history)

    @property
    def history_(self) -> list[dict[str, Any]]:
        return deepcopy(self._require_fitted().history)

    @property
    def n_iter_(self) -> int:
        return self._require_fitted().n_iter

    @property
    def converged_(self) -> bool:
        return self._require_fitted().converged

    @property
    def sequence_lengths_(self) -> tuple[int, ...]:
        return self._require_fitted().sequence_lengths

    def _require_markov(self):
        state = self._require_fitted()
        if not state.hmm_switch:
            raise ValueError('Markov results require hdp.hmm_switch=True')
        return state

    @property
    def pair_responsibilities_(self) -> NDArray[np.float64]:
        return _snapshot(self._require_markov().respPair)

    @property
    def start_counts_(self) -> NDArray[np.float64]:
        return _snapshot(markov_counts(self._require_markov())[0])

    @property
    def transition_counts_(self) -> NDArray[np.float64]:
        return _snapshot(markov_counts(self._require_markov())[1])

    @property
    def transition_parameters_(self) -> NDArray[np.float64]:
        """Dirichlet factors: K active source rows, K+1 destinations with tail."""
        return _snapshot(self._require_markov().transTheta)

    @property
    def start_parameters_(self) -> NDArray[np.float64]:
        return _snapshot(self._require_markov().startTheta)

    @property
    def transition_matrix_(self) -> NDArray[np.float64]:
        """Posterior mean transitions conditioned on represented destinations."""
        theta = self._require_markov().transTheta[:, :-1]
        return _snapshot(theta / theta.sum(1, keepdim=True))

    @property
    def start_probabilities_(self) -> NDArray[np.float64]:
        theta = self._require_markov().startTheta[:-1]
        return _snapshot(theta / theta.sum())

    @property
    def structure_history_(self) -> list[dict[str, Any]]:
        return deepcopy(self._require_fitted().structure_history)

    @torch.no_grad()
    def _evidence(self, values, *, marginal=False):
        state = self._require_fitted()
        return torch.stack([(component.marginal_log_prob if marginal else component.local_evidence)(
            values, noise=state.noise, chunk_size=self._config.chunk_size)
            for component in state.components], 1)

    def predict_proba(self, observations: ArrayLike | torch.Tensor,
                      sequence_lengths: Iterable[int] | None = None) -> NDArray[np.float64]:
        """Soft variational assignments under fitted emission/HDP factors.

        A fresh assignment step can differ from the stored training q(s) when
        fitting stopped before convergence. No weights/parameters are updated.
        In HMM mode these are smoothed marginals; lengths default to one chain.
        """
        state, values = self._require_fitted(), self._values(observations)
        lengths = self._lengths(sequence_lengths, len(values))
        evidence = self._evidence(values)
        if state.hmm_switch:
            return _snapshot(markov_assignments(state, evidence, lengths)[0])
        return _snapshot(_iid_messages(state, evidence)[0])

    def predict(self, observations: ArrayLike | torch.Tensor,
                      sequence_lengths: Iterable[int] | None = None) -> NDArray[np.int64]:
        return _snapshot(self.predict_proba(observations, sequence_lengths).argmax(1))

    def predict_pair_proba(self, observations: ArrayLike | torch.Tensor,
                      sequence_lengths: Iterable[int] | None = None) -> NDArray[np.float64]:
        state, values = self._require_markov(), self._values(observations)
        lengths = self._lengths(sequence_lengths, len(values))
        return _snapshot(markov_assignments(state, self._evidence(values), lengths)[1])

    def transform_components(self, observations: ArrayLike | torch.Tensor) -> MFAFactors:
        values, state = self._values(observations), self._require_fitted()
        posteriors = [component.posterior_factors(values, noise=state.noise,
            chunk_size=self._config.chunk_size) for component in state.components]
        return MFAFactors(_snapshot(torch.stack([posterior.mean for posterior in posteriors], 1)),
                          _snapshot(torch.stack([posterior.covariance for posterior in posteriors])))

    def transform(self, observations: ArrayLike | torch.Tensor, *, assignment: Literal['map'] = 'map',
                  sequence_lengths: Iterable[int] | None = None) -> NDArray[np.float64]:
        """Select each row's MAP component coordinates; never average local bases."""
        if assignment != 'map':
            raise ValueError("Only assignment='map' is supported; component bases are not aligned")
        factors = self.transform_components(observations)
        labels = self.predict(observations, sequence_lengths)
        return _snapshot(factors.mean[np.arange(len(labels)), labels])

    def score_samples(self, observations: ArrayLike | torch.Tensor,
                      sequence_lengths: Iterable[int] | None = None) -> NDArray[np.float64]:
        """Normalized PLUG-IN density, conditional on represented support.

        Uses posterior-mean B and posterior-mean stick weights normalized over
        K components. Not the ELBO or exact Bayesian posterior predictive.
        HMM mode returns filtered predictive log-density increments, restarting
        at sequence boundaries. Their sum is the sequence log likelihood.
        """
        values = self._values(observations)
        lengths = self._lengths(sequence_lengths, len(values))
        state = self._require_fitted()
        if state.hmm_switch:
            # Normalized plug-in HMM density, with explicit support conditioning.
            start = state.startTheta[:-1]
            transition = state.transTheta[:, :-1]
            return _snapshot(markov_assignments(state, self._evidence(values, marginal=True), lengths,
                start=(start / start.sum()).log(),
                transition=(transition / transition.sum(1, keepdim=True)).log())[2])
        weights = torch.as_tensor(self.weights_.copy(), dtype=values.dtype, device=values.device)
        return _snapshot(torch.logsumexp(self._evidence(values, marginal=True) + weights.log(), 1))

    def variational_score_samples(self, observations: ArrayLike | torch.Tensor,
                      sequence_lengths: Iterable[int] | None = None) -> NDArray[np.float64]:
        """Local log-partition increments; excludes global parameter/HDP KLs.

        IID: logsumexp(ell+Elog beta). HMM: forward log-normalizer increments
        from expected-log start/transition potentials, without renormalizing.
        """
        state, values = self._require_fitted(), self._values(observations)
        lengths = self._lengths(sequence_lengths, len(values))
        if state.hmm_switch:
            return _snapshot(markov_assignments(state, self._evidence(values), lengths)[2])
        return _snapshot(torch.logsumexp(self._evidence(values)
                         + state.hdp.expected_log_iid_weights(state.M), 1))

    def score(self, observations: ArrayLike | torch.Tensor,
              sequence_lengths: Iterable[int] | None = None) -> float:
        """Mean normalized plug-in log density (per-observation units).

        For an HMM this is the sequence log likelihood divided by total rows.
        Empty arrays have no mean score and are rejected.
        """
        scores = self.score_samples(observations, sequence_lengths)
        if not len(scores):
            raise ValueError('score requires at least one observation')
        return float(scores.mean())

    def state_dict(self) -> dict[str, Any]:
        state = self._require_fitted()
        components = [component.state_dict() for component in state.components]
        for component in components:
            component.pop('noise')  # one shared learned noise representation
        result = {'type': 'hdpmfa', 'version': 2, 'config': self._config.to_dict(),
                'observation_dim': state.observation_dim, 'noise': state.noise.cpu().tolist(),
                'components': components, 'responsibilities': state.resp.cpu().tolist(),
                'rho': state.rho.cpu().tolist(), 'omega': state.omega.cpu().tolist(),
                'sequence_lengths': list(state.sequence_lengths),
                'history': deepcopy(state.history), 'elbo_history': list(state.elbo_history),
                'elbo_parts': state.elbo_parts.copy(), 'n_iter': state.n_iter,
                'converged': state.converged,
                'structure_history': deepcopy(state.structure_history),
                'seen_partitions': sorted(state.seen_partitions)}
        if state.hmm_switch:
            result.update(pair_responsibilities=state.respPair.cpu().tolist(),
                transition_parameters=state.transTheta.cpu().tolist(),
                start_parameters=state.startTheta.cpu().tolist())
        return result

    @classmethod
    def from_state_dict(cls, data: Mapping[str, Any]) -> HDPMFA:
        """Restore validated data only, without refitting or executing user code."""
        from ._mfa_serialization import validate_payload, validate_restored
        try:
            validate_payload(data)
            model = cls._from_state_dict_impl(data)
            validate_restored(model._state)
            return model
        except (KeyError, TypeError, ValueError, RuntimeError, IndexError, OverflowError) as error:
            raise ValueError(f'Invalid HDPMFA state: {error}') from error

    @classmethod
    def _from_state_dict_impl(cls, data):
        if data.get('type') != 'hdpmfa' or data.get('version') not in (1, 2):
            raise ValueError('Unsupported DP-MFA state type or version')
        model = cls(HDPMFAConfig.from_dict(data['config']))
        state = object.__new__(MFAState)  # restore without PCA/k-means or data
        state.config = model._config
        state.dtype = torch.float64
        state.device = torch.device('cuda' if state.config.numerics.cuda else 'cpu')
        if state.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('Saved configuration requests CUDA, but CUDA is unavailable')
        state.M, state.hmm_switch = len(data['components']), state.config.hdp.hmm_switch
        if (state.M < 1 or (state.config.component_strategy == 'fixed' and state.M != state.config.hdp.M)
                or (state.config.hdp.max_models is not None and state.M > state.config.hdp.max_models)):
            raise ValueError('Saved component support violates the configuration')
        state.observation_dim = data['observation_dim']
        state.hdp = HDPVariational(state)
        state.hdp.set_hyperparameters(state.config.hdp.hdp_hyp)
        state.components = [FactorAnalysisEmission.from_state_dict(dict(component, noise=data['noise']),
                            device=state.device, dtype=state.dtype) for component in data['components']]
        if len(state.components) != state.M or any(component.observation_dim != state.observation_dim
                or component.n_factors != state.config.n_factors for component in state.components):
            raise ValueError('Saved component dimensions do not match the configuration')
        state.noise = state.components[0].noise
        state._saved_noise = state.cond_to_torch(data['noise'])
        state._saved_markov_fields = any(name in data for name in
            ('pair_responsibilities', 'transition_parameters', 'start_parameters'))
        state.resp = state.cond_to_torch(data['responsibilities'])
        if (state.resp.ndim != 2 or state.resp.shape[1] != state.M or not len(state.resp)
                or not bool(torch.isfinite(state.resp).all()) or bool((state.resp < 0).any())
                or not torch.allclose(state.resp.sum(1), state.resp.new_ones(len(state.resp)), atol=1e-10, rtol=1e-10)):
            raise ValueError('Saved responsibilities must be normalized finite probabilities')
        state.sequence_lengths = cls._lengths(data['sequence_lengths'], len(state.resp))
        state.rho, state.omega = state.cond_to_torch(data['rho']), state.cond_to_torch(data['omega'])
        if (state.rho.shape != (state.M,) or state.omega.shape != (state.M,)
                or not bool(torch.isfinite(state.rho).all() and torch.isfinite(state.omega).all())
                or not bool(((state.rho > 0) & (state.rho < 1) & (state.omega > 0)).all())):
            raise ValueError('Saved stick parameters must be positive proper Beta factors')
        if state.hmm_switch:
            cls._restore_markov(state, data)
        else:
            saved_rho, saved_omega = state.rho.clone(), state.omega.clone()
            state.hdp.update_iid_weights(state.resp)
            if not (torch.allclose(state.rho, saved_rho, atol=1e-12, rtol=1e-12)
                    and torch.allclose(state.omega, saved_omega, atol=1e-10, rtol=1e-12)):
                raise ValueError('Saved sticks disagree with responsibility sufficient statistics')
            state.rho, state.omega = saved_rho, saved_omega
        state.structure_history = deepcopy(data.get('structure_history', []))
        state.seen_partitions = set(data.get('seen_partitions', []))
        state.history, state.elbo_history = deepcopy(data['history']), list(data['elbo_history'])
        state.elbo_parts = dict(data['elbo_parts'])
        state.n_iter, state.converged = data['n_iter'], data['converged']
        if (not isinstance(state.n_iter, Integral) or isinstance(state.n_iter, bool)
                or not 1 <= state.n_iter <= state.config.max_iter
                or type(state.converged) is not bool
                or len(state.history) != state.n_iter or len(state.elbo_history) != state.n_iter + 1
                or set(state.elbo_parts) != {'local_evidence', 'allocation', 'assignment_entropy',
                                             'parameter_kl', 'total'}
                or not all(isinstance(value, (float, int)) and math.isfinite(value)
                           for value in [*state.elbo_parts.values(), *state.elbo_history])):
            raise ValueError('Saved DP-MFA objective history is invalid')
        parts = state.elbo_parts
        total = parts['local_evidence'] + parts['allocation'] + parts['assignment_entropy'] - parts['parameter_kl']
        if not (math.isclose(parts['total'], total, rel_tol=1e-12, abs_tol=1e-9)
                and math.isclose(parts['total'], state.elbo_history[-1], rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError('Saved objective components do not sum to the final ELBO')
        state.initial_elbo = state.elbo_history[0]
        model._state = state
        return model

    @staticmethod
    def _restore_markov(state, data):
        state.respPair = state.cond_to_torch(data['pair_responsibilities'])
        state.transTheta = state.cond_to_torch(data['transition_parameters'])
        state.startTheta = state.cond_to_torch(data['start_parameters'])
        n, k = state.resp.shape
        pair = state.respPair
        if (pair.shape != (n, k, k) or not bool(torch.isfinite(pair).all())
                or bool((pair < 0).any())):
            raise ValueError('Saved pair responsibilities have invalid dimensions/probabilities')
        offset = 0
        for length in state.sequence_lengths:
            end = offset + length
            if (bool((pair[offset] != 0).any())
                    or not torch.allclose(pair[offset + 1:end].sum(1), state.resp[offset + 1:end], atol=1e-8, rtol=1e-8)
                    or not torch.allclose(pair[offset + 1:end].sum(2), state.resp[offset:end - 1], atol=1e-8, rtol=1e-8)):
                raise ValueError('Saved pair marginals violate sequence boundaries or singleton marginals')
            offset = end
        if (state.transTheta.shape != (k, k + 1) or state.startTheta.shape != (k + 1,)
                or not bool(torch.isfinite(state.transTheta).all() and torch.isfinite(state.startTheta).all())
                or not bool((state.transTheta > 0).all() and (state.startTheta > 0).all())):
            raise ValueError('Saved HMM Dirichlet factors are invalid')
        starts, counts = markov_counts(state)
        beta = state.hdp.rho2beta(state.rho.cpu().numpy(), returnSize='K+1')
        expected = state.transAlpha * beta.repeat(k, 1)
        expected[:, :k] += counts + state.kappa * torch.eye(k, dtype=state.dtype, device=state.device)
        expected_start = state.startAlpha * beta
        expected_start[:k] += starts
        if (not torch.allclose(state.transTheta, expected, atol=1e-9, rtol=1e-10)
                or not torch.allclose(state.startTheta, expected_start, atol=1e-9, rtol=1e-10)):
            raise ValueError('Saved HMM factors disagree with sufficient statistics')

    def save(self, path: str | PathLike[str]) -> None:
        """Atomically replace a data-only JSON snapshot; never pickle callbacks."""
        from ._mfa_serialization import atomic_save
        atomic_save(path, self.state_dict())

    @classmethod
    def load(cls, path: str | PathLike[str]) -> HDPMFA:
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding='utf-8')))
