"""Native independent-assignment core, without GP/model imports.

Extracted unchanged from _messages for reuse by non-GP emission schedules.
"""
import torch


def _shifted_log_evidence(q):
    """Keep emission ratios without exponentiating a large score gap."""
    maxima = torch.max(q, dim=1).values
    bad = torch.where(~torch.isfinite(maxima) | torch.isnan(q).any(dim=1))[0]
    if bad.numel():
        raise FloatingPointError(
            f'No finite emission maximum at observation {int(bad[0].item())}'
        )
    return q - maxima[:, None]


def _iid_messages(self, q):
    """Independent-label marginals without any forward/backward recurrence."""
    evidence = _shifted_log_evidence(q)
    weights = self.hdp.expected_log_iid_weights(q.shape[1]).to(q)
    joint = evidence + weights - weights.max()
    normalizers = torch.logsumexp(joint, dim=1)
    if not bool(torch.isfinite(normalizers).all()):
        raise FloatingPointError('Independent assignments have no finite component')
    return torch.softmax(joint, dim=1), torch.ones_like(q), normalizers.exp()
