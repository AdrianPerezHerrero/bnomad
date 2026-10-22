"""Conjugate stick weights and ELBO terms for independent mixture labels."""

import torch


def _posterior_from_counts(counts, gamma):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    tail = torch.flip(torch.cumsum(torch.flip(counts, dims=(0,)), dim=0), dims=(0,)) - counts
    a = 1.0 + counts
    b = torch.as_tensor(gamma, dtype=counts.dtype, device=counts.device) + tail
    return a, b


def update_iid_weights(state, responsibilities):
    counts = torch.as_tensor(responsibilities, dtype=torch.float64).sum(dim=0)
    a, b = _posterior_from_counts(counts, state.gamma)
    state.rho = a / (a + b)
    state.omega = a + b


def expected_log_iid_weights(state, n_states):
    rho = torch.as_tensor(state.rho, dtype=torch.float64)
    omega = torch.as_tensor(state.omega, dtype=rho.dtype, device=rho.device)
    a, b = rho * omega, (1.0 - rho) * omega
    log_stick = torch.digamma(a) - torch.digamma(a + b)
    log_remainder = torch.digamma(b) - torch.digamma(a + b)
    result = []
    previous_remainder = a.new_zeros(())
    for index in range(n_states):
        if index < len(a):
            result.append(previous_remainder + log_stick[index])
            previous_remainder = previous_remainder + log_remainder[index]
        elif index == n_states - 1:
            result.append(previous_remainder)
        else:
            prior_a = a.new_tensor(1.0)
            prior_b = a.new_tensor(float(state.gamma))
            result.append(previous_remainder + torch.digamma(prior_a)
                          - torch.digamma(prior_a + prior_b))
            previous_remainder = (previous_remainder + torch.digamma(prior_b)
                                  - torch.digamma(prior_a + prior_b))
    return torch.stack(result)


def iid_linear_elbo(state, responsibilities):
    """Optimized stick ELBO: E log p(v)/q(v) + sum_n E log beta[z_n].

    The independent-label entropy is added by the inference engine exactly
    once, separately from this linear term.
    """
    counts = torch.as_tensor(responsibilities, dtype=torch.float64).sum(dim=0)
    a, b = _posterior_from_counts(counts, state.gamma)
    prior_a = torch.ones_like(a)
    prior_b = torch.full_like(b, float(state.gamma))
    digamma_sum = torch.digamma(a + b)
    log_v = torch.digamma(a) - digamma_sum
    log_1mv = torch.digamma(b) - digamma_sum
    previous = torch.cat((a.new_zeros(1), torch.cumsum(log_1mv[:-1], dim=0)))
    expected_log_beta = previous + log_v
    log_b_q = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    log_b_p = (torch.lgamma(prior_a) + torch.lgamma(prior_b)
               - torch.lgamma(prior_a + prior_b))
    kl = (log_b_p - log_b_q + (a - prior_a) * log_v
          + (b - prior_b) * log_1mv).sum()
    return (counts * expected_log_beta).sum() - kl
