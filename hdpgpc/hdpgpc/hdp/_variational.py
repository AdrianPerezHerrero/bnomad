"""Existing HDP variational calculations, with a narrow live state view.

Relocated from GPI_HDP.py (Adrian Perez Herrero). The bnpy-adapted optimizer
and all its attribution/comments are retained in _optimizer.py.
No method here constructs or updates GP, LDS or alignment components.
"""

import numpy as np
import torch
from scipy.special._ufuncs import psi as digamma
from scipy.special._ufuncs import gammaln
from ._optimizer import find_optimum_multiple_tries, kvec
from ._entropy import state_entropy_terms, pair_entropy_terms
from ._iid import (update_iid_weights, expected_log_iid_weights,
                   iid_linear_elbo)
from . import _topology


class HDPVariational:
    """A live view of existing variational state, not a second state store.

    state provides rho/omega, transition/start theta, gamma/transAlpha/
    startAlpha/kappa, OptimizerInfo, M, device and the four existing cond_*
    conversion operations. Only those named fields can cross this boundary.
    Arrays are neither copied nor coerced by the view. Callers retain ownership
    of initialization timing, global/local iteration counts and cluster lifecycle.
    """

    __slots__ = ("_state",)
    _FIELDS = frozenset(("rho", "omega", "transTheta", "startTheta", "gamma",
                         "transAlpha", "startAlpha", "kappa", "OptimizerInfo"))
    _CONTEXT = frozenset(("M", "device", "cond_cpu", "cond_cuda", "cond_to_numpy", "cond_to_torch"))

    def __init__(self, state):
        object.__setattr__(self, "_state", state)

    def __getattr__(self, name):
        if name in self._FIELDS or name in self._CONTEXT:
            return getattr(self._state, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self._FIELDS:
            setattr(self._state, name, value)
        else:
            raise AttributeError(name)

    def set_hyperparameters(self, hdp_hyp):
        """Resolve the existing preset for Markov or independent assignments.

        HMM profiles retain their calibrated joint concentrations. Without an
        HMM, gamma alone controls the mixture stick prior; its preset ordering
        increases from less to max. Transition/start concentrations and sticky
        mass are inactive in the independent-label objective.
        """

        # HDP Hyperparams
        if hdp_hyp in ('less', 'less_extra_sticky'):
            # Less clusters scheme
            self.gamma = 1e+03
            self.transAlpha = 1e+03
            self.startAlpha = 1e+00
            self.kappa = 100.0 if hdp_hyp == 'less_extra_sticky' else 0.0
        elif hdp_hyp in ('balanced', 'balanced_sticky'):
            #Balanced scheme
            self.gamma = 1e+03
            self.transAlpha = 1e+02
            self.startAlpha = 1e+00
            self.kappa = 10.0 if hdp_hyp == 'balanced_sticky' else 0.0
        elif hdp_hyp in ('more', 'more_sticky', 'more_extra_sticky'):
            #More clusters scheme
            self.gamma = 1e+01
            self.transAlpha = 1e+02
            self.startAlpha = 1e+00
            self.kappa = {'more': 0.0, 'more_sticky': 10.0, 'more_extra_sticky': 100.0}[hdp_hyp]
        elif hdp_hyp == 'max':
            self.gamma = 1e+00
            self.transAlpha = 1e-02
            self.startAlpha = 1e+00
            self.kappa = 0.0

        if not self._state.hmm_switch:
            self.gamma = {'less': 0.001, 'less_extra_sticky': 0.001,
                          'balanced': 0.1, 'balanced_sticky': 0.1,
                          'more': 1.0, 'more_sticky': 1.0, 'more_extra_sticky': 1.0,
                          'max': 100.0}[hdp_hyp]


    def init_global_params(self, d_dim, M):
        self.rho = self.create_initrho(M)
        self.omega = (1.0 + self.gamma) * self.cond_cuda(torch.ones(M))
        transStateCount = self.cond_cuda(torch.ones((M, M)))  # Cause we are on initial moment
        if _topology.enabled(self._state):
            transStateCount.zero_()
        startStateCount = self.cond_cuda(torch.ones(M))  # Cause we are on initial moment
        self.transTheta, self.startTheta = self._calcThetaFull(transStateCount, startStateCount, M + 1)
        if not self._state.hmm_switch:
            self.update_iid_weights(torch.zeros((0, M), dtype=torch.float64))

    update_iid_weights = update_iid_weights
    expected_log_iid_weights = expected_log_iid_weights
    iid_linear_elbo = iid_linear_elbo

    def iid_log_transitions(self, n_states):
        log_weights = self.expected_log_iid_weights(n_states)
        return log_weights.repeat(n_states, 1)


    def reinit_global_params(self, M, transStateCount_, startStateCount_):
        self.rho = self.create_initrho(M)
        self.omega = (1.0 + self.gamma) * torch.ones(M)
        self.transTheta, self.startTheta = self._calcThetaFull(transStateCount_, startStateCount_, M=M)


    def temp_reinit_global_params(self, M, transStateCount_, startStateCount_, rho=None, omega=None):
        if rho is None:
            rho = self.rho
        if omega is None:
            omega = self.omega
        rho_ = self.create_initrho(M)
        rho_[:rho.shape[0]] = rho
        omega_ = (1.0 + self.gamma) * torch.ones(M)
        omega_[:omega.shape[0]] = omega
        transTheta_, startTheta_ = self._calcThetaFull(transStateCount_, startStateCount_, M + 1, rho_)
        return rho_, omega_, transTheta_, startTheta_


    def create_initrho(self, M):
        if _topology.enabled(self._state):
            return torch.full((M,), 1./(1.+self.gamma), dtype=torch.float64, device=self.device)
        remMass = np.minimum(0.1, 1.0 / (M * M))
        delta = (-1 + remMass) * np.arange(0, M, 1, dtype=np.float64)
        rho = (1 - remMass) / (M + delta)
        return self.cond_cuda(self.cond_to_torch(rho))


    def _calcThetaPost(self, transStateCount, startStateCount, M, rho):
        if _topology.enabled(self._state):
            return _topology.posterior_matrices(self._state, transStateCount, rho)
        transStateCount_ = transStateCount
        Ebeta = self.rho2beta(self.cond_cpu(rho), returnSize='K+1')
        alphaEbeta = self.transAlpha * Ebeta

        transTheta = self.cond_cuda(torch.zeros((M, M)))
        transTheta += alphaEbeta[np.newaxis, :]
        # coef = self.M / M
        transTheta[:M - 1, :M - 1] += self.transTheta * 0.8
        transTheta[:M, :M] += transStateCount_[:M, :M] * 0.2 + self.kappa * self.cond_cuda(torch.eye(M))

        startTheta = self.startAlpha * Ebeta
        startTheta[:M - 1] += self.startTheta
        startTheta[:M] += startStateCount[:M]
        # return transTheta / torch.sum(transTheta), startTheta / torch.sum(startTheta)
        return transTheta, startTheta


    def _calcThetaFull(self, transStateCount, startStateCount, M=None, rho=None, kappa=None):
        if _topology.enabled(self._state):
            if kappa is not None and kappa != self.kappa:
                raise ValueError('Ordered transition updates must use the configured sticky prior')
            return _topology.posterior_matrices(self._state, transStateCount, rho)
        transStateCount_ = transStateCount
        if M is None:
            M = self.M + 1
        if rho is None:
            rho = self.rho
        if kappa is None:
            kappa = self.kappa
        if M == rho.shape[0]:
            Ebeta = self.rho2beta(self.cond_cpu(rho), returnSize='K')
        else:
            Ebeta = self.rho2beta(self.cond_cpu(rho), returnSize='K+1')
        alphaEbeta = self.transAlpha * Ebeta

        transTheta = self.cond_cuda(torch.zeros((M, M)))
        transTheta += alphaEbeta[np.newaxis, :]
        transTheta[:M - 1, :M - 1] += transStateCount_[:M - 1, :M - 1] + kappa * self.cond_cuda(torch.eye(M - 1))

        startTheta = self.startAlpha * Ebeta
        startTheta[:M - 1] += startStateCount[:M - 1]

        # return transTheta / torch.sum(transTheta), startTheta / torch.sum(startTheta)
        return transTheta, startTheta


    def compute_Pi(self):
        """Return the posterior mean transition matrix E[Pi]."""
        theta = self.cond_cpu(self.transTheta)
        return theta / torch.sum(theta, dim=1, keepdim=True)


    def rho2beta(self, rho, returnSize='K+1'):
        rho = np.asarray(rho, dtype=np.float64)
        if returnSize == 'K':
            beta = rho.copy()
            beta[1:] *= np.cumprod(1 - rho[:-1])
        else:
            beta = np.append(rho, 1.0)
            beta[1:] *= np.cumprod(1.0 - rho)
        return self.cond_cuda(self.cond_to_torch(beta))


    def beta2rho(self, beta, K):
        ''' Get rho vector that can deterministically produce provided beta.

        Returns
        ------
        rho : 1D array, size K
            Each entry rho[k] >= 0.
        '''
        beta = np.asarray(beta, dtype=np.float64)
        rho = beta.copy()
        beta_gteq = 1 - np.cumsum(beta[:-1])
        rho[1:] /= np.maximum(1e-100, beta_gteq)
        if beta.size == K + 1:
            return self.cond_cuda(self.cond_to_torch(rho[:-1]))
        elif beta.size == K:
            return self.cond_cuda(self.cond_to_torch(rho))
        else:
            raise ValueError('Provided beta needs to be of length K or K+1')


    def elbo_Linears(self, resp, respPair, post=False, one_sample=False):
        """ Compute ELBO for linear terms. HDP.
        """
        if _topology.enabled(self._state):
            if float(resp[0, 0]) < 1.-1e-8:
                return -np.inf
            return _topology.linear_terms(self._state, self.transition_counts(respPair))
        startStateCount = resp[0]
        transStateCount = self.transition_counts(respPair)
        M = resp.shape[1]

        # Augment suff stats to be sure have 0 in final column,
        # which represents inactive states.
        if startStateCount.shape[0] == M:
            startStateCount = torch.hstack([startStateCount, torch.zeros(1)])
        if transStateCount.shape[-1] == M:
            transStateCount = torch.hstack([transStateCount, torch.zeros((M, 1))])
            transStateCount = torch.vstack([transStateCount, torch.zeros((1, M + 1))])

        if self.rho.shape[0] == M:
            rho_ = torch.clone(self.rho)
            omega_ = torch.clone(self.omega)
        else:
            rho_, omega_, transTheta_, startTheta_ = self.temp_reinit_global_params(M, torch.clone(transStateCount),
                                                                                    torch.clone(startStateCount))
        # nIters = 4
        # for giter in range(nIters):
        if post:
            nIters = 1
            for giter in range(nIters):
                transTheta_, startTheta_ = self._calcThetaFull(self.cond_cuda(torch.clone(transStateCount)),
                                                               self.cond_cuda(torch.clone(startStateCount)), M + 1,
                                                               rho=rho_)

        else:
            transTheta_, startTheta_ = self._calcThetaFull(self.cond_cuda(torch.clone(transStateCount)),
                                                           self.cond_cuda(torch.clone(startStateCount)), M + 1,
                                                           rho=rho_)

        return self.calcELBO_LinearTerms(rho=self.cond_to_numpy(self.cond_cpu(rho_)),
                                         omega=self.cond_to_numpy(self.cond_cpu(omega_)),
                                         alpha=self.transAlpha,
                                         startAlpha=self.startAlpha,
                                         kappa=self.kappa, gamma=self.gamma,
                                         transTheta=self.cond_to_numpy(self.cond_cpu(transTheta_)),
                                         startTheta=self.cond_to_numpy(self.cond_cpu(startTheta_)),
                                         startStateCount=self.cond_to_numpy(self.cond_cpu(startStateCount)),
                                         transStateCount=self.cond_to_numpy(
                                             torch.clone(self.cond_cpu(transStateCount))))


    def refill_resp(self, resp, respPair=None, occupied=None):
        """ Refill HDP parameters if a model is reasignated.
        """
        if _topology.enabled(self._state):
            raise ValueError('Ordered stages must be removed by adjacent merges, not slot swapping')
        resp_per_group = torch.sum(resp, axis=0)
        if occupied is None:
            occupied = resp_per_group >= 1.0
        if torch.any(~occupied[:-1]):
            empty_group_ind = torch.where(~occupied)[0]
            if empty_group_ind.shape[0] > 1:
                empty_group_ind = empty_group_ind[0]
            resp_last_group = torch.clone(resp[:, -1])
            resp[:, -1] = torch.clone(resp[:, empty_group_ind.item()])
            resp[:, empty_group_ind.item()] = resp_last_group
            if respPair is not None:
                resp_pair_last_group_rows = torch.clone(respPair[:, -1, :])
                resp_pair_last_group_rows_ = torch.clone(resp_pair_last_group_rows[:, -1])
                resp_pair_last_group_rows[:, -1] = torch.clone(resp_pair_last_group_rows[:, empty_group_ind.item()])
                resp_pair_last_group_rows[:, empty_group_ind.item()] = torch.clone(resp_pair_last_group_rows_)
                resp_pair_last_group_cols = torch.clone(respPair[:, :-1, -1])
                resp_pair_last_group_cols_ = torch.clone(resp_pair_last_group_cols[:, empty_group_ind.item()])
                respPair[:, -1, :] = torch.clone(respPair[:, empty_group_ind.item(), :])
                respPair[:, :-1, -1] = torch.clone(respPair[:, :-1, empty_group_ind.item()])
                respPair[:, empty_group_ind.item(), :] = resp_pair_last_group_rows
                access_indexes = tuple(np.delete(np.arange(respPair.shape[1]), empty_group_ind.item()))
                respPair[:, access_indexes, empty_group_ind.item()] = resp_pair_last_group_cols
                respPair[:, -1, empty_group_ind.item()] = resp_pair_last_group_cols_
        if respPair is not None:
            return resp, respPair
        else:
            return resp


    def calcELBO_LinearTerms(self, rho, omega, alpha, startAlpha, kappa, gamma, transTheta, startTheta, startStateCount,
                             transStateCount):
        """ Method to compute HPD linear terms.
        """
        if _topology.enabled(self._state):
            if (alpha, kappa, gamma) != (self.transAlpha, self.kappa, self.gamma):
                raise ValueError('Ordered ELBO must use the configured transition prior')
            return _topology.linear_terms(self._state, transStateCount, rho, omega, transTheta)
        transStateCount_ = transStateCount
        Ltop = self.L_top(rho=rho, omega=omega, alpha=alpha, startAlpha=startAlpha, kappa=kappa, gamma=gamma)
        LdiffcDir = - self.c_Dir(transTheta) - self.c_Dir(startTheta)
        K = transStateCount_.shape[0]
        if startTheta.shape[0] == rho.size:
            Ebeta = self.cond_to_numpy(self.cond_cpu(self.rho2beta(rho, returnSize='K')))
        else:
            Ebeta = self.cond_to_numpy(self.cond_cpu(self.rho2beta(rho, returnSize='K + 1')))
        LstartSlack = np.inner(
            startStateCount + startAlpha * Ebeta - startTheta,
            digamma(startTheta) - digamma(np.sum(startTheta))
        )
        alphaEbetaPlusKappa = alpha * np.tile(Ebeta, (K, 1))
        alphaEbetaPlusKappa[:, :K] += kappa * np.eye(K)
        # digammaSum = np.log(np.sum(np.exp(digamma(transTheta)), axis=1))
        digammaSum = digamma(np.sum(transTheta, axis=1))
        transStateCount_[:K, :] = transStateCount_[:K, :] + alphaEbetaPlusKappa
        # LtransSlack = np.sum(
        #     (transStateCount_ - transTheta) *
        #     (digamma(transTheta) - digammaSum[:, np.newaxis])
        # )
        LtransSlack = np.sum(
            (transStateCount_ - transTheta) *
            (digamma(transTheta) - digammaSum[:, None])
        )
        return Ltop + LdiffcDir + LstartSlack + LtransSlack


    def calcELBO_NonlinearTerms(self, resp, respPair):
        """H[q(z_0)] + sum_{t>0} H[q(z_t | z_{t-1})]."""
        if not self._state.hmm_switch:
            return state_entropy_terms(resp).sum()
        Htable = self.calc_Htable(respPair[1:])
        Hstart = self.calc_Hstart(resp[:1])
        Lentropy = Htable.sum() + Hstart.sum()
        return Lentropy

    def transition_counts(self, respPair):
        """Count actual transitions; row zero has no predecessor state."""
        return torch.sum(respPair[1:], dim=0)


    def calc_Hstart(self, resp, eps=1e-30):
        return state_entropy_terms(resp, eps)


    def calc_Htable(self, respPair, eps=1e-30):
        return pair_entropy_terms(respPair, eps)


    def L_top(self, rho, omega, alpha, startAlpha, kappa, gamma):
        if _topology.enabled(self._state):
            a, b = _topology.shapes(rho, omega)
            return -_topology._beta_kl(a, b, gamma)
        K = rho.size
        eta1 = rho * omega
        eta0 = (1 - rho) * omega
        digamma_omega = digamma(omega)
        ElogU = digamma(eta1) - digamma_omega
        Elog1mU = digamma(eta0) - digamma_omega

        def c_Beta(a1, a0):
            return np.sum(gammaln(a1 + a0)) - np.sum(gammaln(a1)) - np.sum(gammaln(a0))

        diff_cBeta = K * c_Beta(1.0, gamma) - c_Beta(eta1, eta0)

        tAlpha = K * K * np.log(alpha) + K * np.log(startAlpha)
        if kappa > 0:
            coefU = K + 1.0 - eta1
            coef1mU = K * kvec(K) + 1.0 + gamma - eta0
            sumEbeta = np.sum(self.cond_to_numpy(self.cond_cpu(self.rho2beta(rho, returnSize='K'))))
            tBeta = sumEbeta * (np.log(alpha + kappa) - np.log(kappa))
            tKappa = K * (np.log(kappa) - np.log(alpha + kappa))
        else:
            coefU = (K + 1) + 1.0 - eta1
            coef1mU = (K + 1) * kvec(K) + gamma - eta0
            tBeta = 0.0
            tKappa = 0.0

        diff_logU = np.inner(coefU, ElogU)\
                    + np.inner(coef1mU, Elog1mU)
        return tAlpha + tKappa + tBeta + diff_cBeta + diff_logU


    def c_Dir(self, AMat, arem=None):
        ''' Evaluate cumulant function of the Dir distribution
        When input is vectorized, we compute sum over all entries.
        Returns
        -------
        c : scalar real
        '''
        AMat = np.asarray(AMat)
        if _topology.enabled(getattr(self, '_state', None)) and arem is None:
            rows = AMat[None] if AMat.ndim == 1 else AMat
            return sum(gammaln(row[row > 0].sum())-gammaln(row[row > 0]).sum()
                       for row in rows)
        D = AMat.shape[0]
        if arem is None:
            if AMat.ndim == 1:
                return gammaln(np.sum(AMat)) - np.sum(gammaln(AMat))
            else:
                return np.sum(gammaln(np.sum(AMat, axis=1)))\
                    - np.sum(gammaln(AMat))

        return np.sum(gammaln(np.sum(AMat, axis=1) + arem))\
            - np.sum(gammaln(AMat))\
            - D * np.sum(gammaln(arem))


    def find_optimum_rhoOmega(self, startTheta=None, transTheta=None, rho=None, omega=None, M=None):
        ''' Performs numerical optimization of rho and omega for M-step update.

                Note that the optimizer forces rho to be in [EPS, 1-EPS] for
                the sake of numerical stability

                Returns
                -------
                rho : 1D array, size K
                omega : 1D array, size K
                Info : dict of information about optimization.
                '''

        if _topology.enabled(self._state):
            rho, omega, info = _topology.optimize_sticks(
                self._state, self.transTheta if transTheta is None else transTheta)
            self.OptimizerInfo = info
            return rho, omega
        # Calculate expected log transition probability
        # using theta vectors for all K states plus initial state
        if startTheta is None:
            startTheta = self.startTheta
        if transTheta is None:
            transTheta = self.transTheta
        jit = 1e-5
        ELogPi = digamma(self.cond_to_numpy(self.cond_cpu(transTheta)))\
                 - np.log(np.sum(np.exp(digamma(self.cond_to_numpy(self.cond_cpu(transTheta)))), axis=1) + jit)[:,
                   np.newaxis]
        sumELogPi = np.sum(ELogPi, axis=0)
        startELogPi = digamma(self.cond_to_numpy(self.cond_cpu(startTheta)))\
                      - np.log(np.sum(np.exp(digamma(self.cond_to_numpy(self.cond_cpu(startTheta))))) + jit)

        # Select initial rho, omega values for gradient descent
        if not rho is None:
            initRho = self.cond_to_numpy(self.cond_cpu(rho))
        else:
            if hasattr(self, 'rho'):
                initRho = self.cond_to_numpy(self.cond_cpu(self.rho))
            else:
                initRho = None
        if not omega is None:
            initOmega = self.cond_to_numpy(self.cond_cpu(omega))
        else:
            if hasattr(self, 'omega'):
                initOmega = self.cond_to_numpy(self.cond_cpu(self.omega))
            else:
                initOmega = None
        if not M is None:
            M_ = M
        else:
            M_ = self.M + 1

        # Do the optimization
        try:
            rho, omega, fofu, Info =\
                find_optimum_multiple_tries(
                    sumLogPi=sumELogPi,
                    sumLogPiActiveVec=None,
                    sumLogPiRemVec=None,
                    startAlphaLogPi=self.startAlpha * startELogPi,
                    nDoc=M_,
                    gamma=self.gamma,
                    alpha=self.transAlpha,
                    kappa=self.kappa,
                    initrho=initRho,
                    initomega=initOmega)
            self.OptimizerInfo = Info
            self.OptimizerInfo['fval'] = fofu

        except ValueError as error:
            if hasattr(self, 'rho') and self.rho.size()[0] == self.M:
                if self.verbose:
                    print('***** Optim failed. Remain at cur val. ' + str(error))
                rho = self.rho
                omega = self.omega
            else:
                if self.verbose:
                    print('***** Optim failed. Set to prior. ' + str(error))
                omega = (self.gamma + 1) * self.cond_cuda(torch.ones(self.M))
                rho = 1 / float(1 + self.gamma) * self.cond_cuda(torch.ones(self.M))

        return self.cond_cuda(self.cond_to_torch(rho)), self.cond_cuda(self.cond_to_torch(omega))


    def compute_trans_A(self, M):
        if _topology.enabled(self._state):
            return _topology.expected_logs(self.transTheta, M)
        digammaSumTransTheta = digamma(torch.sum(self.cond_cpu(self.transTheta[:M, :M + 1]), axis=1))
        transPi = digamma(self.cond_cpu(self.transTheta[:M, :M])) - digammaSumTransTheta[:, None]
        if transPi.shape[0] == M:
            return transPi
        else:
            transPi_ = torch.zeros(M, M) - np.inf
            transPi_[:M - 1, :M - 1] = transPi
            return transPi_


    def compute_trans_pi(self, M, pi):
        if pi.shape[0] == M:
            return pi
        else:
            pi_ = torch.zeros(M) - np.inf
            pi_[:M - 1] = pi
            return pi_


    def expected_log_transitions(self, M, transTheta):
        if _topology.enabled(self._state):
            return _topology.expected_logs(transTheta, M)
        digammaSumTransTheta = digamma(torch.sum(self.cond_cpu(transTheta[:M, :M + 1]), axis=1))
        transPi = digamma(self.cond_cpu(transTheta[:M, :M])) - digammaSumTransTheta[:, None]
        return transPi

    def expected_log_start(self, M, startTheta):
        if _topology.enabled(self._state):
            return _topology.start_logs(startTheta, M)
        startPi = digamma(self.cond_cpu(startTheta[:M])) - digamma(torch.sum(self.cond_cpu(startTheta[:M + 1])))
        return startPi

    def batch_log_transitions(self, M, transTheta):
        return self.expected_log_transitions(M, transTheta)

    def batch_log_start(self, M, startTheta):
        return self.expected_log_start(M, startTheta)

    def stored_log_transitions(self, M, transTheta):
        return self.expected_log_transitions(M, transTheta)

    def classification_log_transitions(self, M, transTheta):
        return self.expected_log_transitions(M, transTheta)

    def classification_log_start(self, M, startTheta):
        return self.expected_log_start(M, startTheta)

