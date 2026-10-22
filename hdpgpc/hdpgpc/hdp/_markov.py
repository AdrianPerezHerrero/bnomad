"""Active-state adapter for the native HDP-HMM variational bound.

K source rows and K+1 destinations (including the unoccupied tail). The
legacy trajectory adapter pads a source row too; MFA must not score that row.
"""
import math

import numpy as np
import torch

from ._optimizer import find_optimum_multiple_tries


def markov_counts(state):
    starts = np.cumsum((0,) + state.sequence_lengths[:-1]).tolist()
    return state.resp[starts].sum(0), state.respPair.sum(0)


def markov_linear_elbo(self, start_counts, transition_counts):
    k = len(self.rho)
    counts = torch.cat((transition_counts, transition_counts.new_zeros((k, 1))), 1)
    starts = torch.cat((start_counts, start_counts.new_zeros(1)))
    array = lambda x: x.detach().cpu().numpy().copy()
    value = self.calcELBO_LinearTerms(
        rho=array(self.rho), omega=array(self.omega), alpha=self.transAlpha,
        startAlpha=self.startAlpha, kappa=self.kappa, gamma=self.gamma,
        transTheta=array(self.transTheta), startTheta=array(self.startTheta),
        startStateCount=array(starts), transStateCount=array(counts))
    return start_counts.new_tensor(value)


def update_markov_weights(self, start_counts, transition_counts, *, initialize=False):
    """Coordinate updates of Dirichlet factors and native optimized Beta sticks.

Use exact E[log Pi], K active transition rows, and the existing optimizer.
Keep the previous sticks if numerical optimization cannot improve the bound.
"""
    k = len(start_counts)
    if initialize:
        self.rho = self.create_initrho(k)
        self.omega = self.rho.new_full((k,), self.gamma + 1.)

    def theta_update():
        beta = self.rho2beta(self.rho.detach().cpu().numpy(), returnSize='K+1')
        self.transTheta = self.transAlpha * beta.repeat(k, 1)
        self.transTheta[:, :k] += transition_counts + self.kappa * torch.eye(
            k, dtype=beta.dtype, device=beta.device)
        self.startTheta = self.startAlpha * beta
        self.startTheta[:k] += start_counts

    theta_update()
    for _ in range(2):
        before = float(markov_linear_elbo(self, start_counts, transition_counts))
        log_pi = torch.digamma(self.transTheta) - torch.digamma(self.transTheta.sum(1))[:, None]
        log_start = torch.digamma(self.startTheta) - torch.digamma(self.startTheta.sum())
        old_rho, old_omega = self.rho, self.omega
        try:
            rho, omega, _, _ = find_optimum_multiple_tries(
                sumLogPi=log_pi.sum(0).cpu().numpy(),
                startAlphaLogPi=self.startAlpha * log_start.cpu().numpy(),
                nDoc=k + 1, gamma=self.gamma, alpha=self.transAlpha, kappa=self.kappa,
                initrho=old_rho.cpu().numpy(), initomega=old_omega.cpu().numpy())
            self.rho, self.omega = self.cond_to_torch(rho), self.cond_to_torch(omega)
            after = float(markov_linear_elbo(self, start_counts, transition_counts))
            if not math.isfinite(after) or after < before:
                self.rho, self.omega = old_rho, old_omega
        except ValueError:
            self.rho, self.omega = old_rho, old_omega
        theta_update()

