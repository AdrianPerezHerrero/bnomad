"""Independent HMM recurrence values and autograd/subclass dispatch contracts."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.characterization._support import ModelState, torch


def reference_messages(q, transition, pi):
    """Independent log-space HMM recurrence with a common transition scale."""
    evidence = q - q.amax(dim=1, keepdim=True)
    log_trans = transition - transition.amax()
    log_start = pi - pi.amax()
    alpha, scales = [], []
    for t in range(len(q)):
        joint = log_start + evidence[t] if t == 0 else evidence[t] + torch.logsumexp(
            alpha[-1][:, None] + log_trans, dim=0)
        normalizer = torch.logsumexp(joint, dim=0)
        alpha.append(joint-normalizer)
        scales.append(normalizer.exp())
    beta = [torch.zeros_like(pi)]
    for t in range(len(q)-2, -1, -1):
        row = torch.logsumexp(log_trans + evidence[t+1][None, :] + beta[-1][None, :], dim=1)
        beta.append(row-torch.logsumexp(row, dim=0))
    return torch.stack(alpha).exp(), torch.stack(scales), torch.stack(beta[::-1]).exp()


def owner(q,transition,pi):
    return SimpleNamespace(hmm_switch=True, q=[q],trans_A=transition,pi=[pi],M=q.shape[1],
                           device=q.device,fmsg=None,verbose=False,
                           compute_trans_pi=lambda k,p:p,
                           compute_trans_A=lambda k:transition,
                           cond_cuda=lambda value:value, cond_to_torch=lambda value:torch.as_tensor(value))


class MessageRowTests(unittest.TestCase):
    def assert_values(self,q,transition,pi):
        state=owner(q,transition,pi)
        expected=reference_messages(q,transition,pi)
        with redirect_stdout(io.StringIO()):
            actual=(*ModelState.forward(state),ModelState.backward(state))
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=32*torch.finfo(q.dtype).eps,atol=32*torch.finfo(q.dtype).eps)
            self.assertEqual(a.dtype,e.dtype)
            self.assertEqual(a.device,e.device)
        return actual,expected

    def test_exact_float32_float64_lengths_and_cluster_counts(self):
        saved=torch.get_default_dtype()
        try:
            for dtype in (torch.float32,torch.float64):
                torch.set_default_dtype(dtype)
                for T in (1,2,17):
                    for K in (1,2,4):
                        with self.subTest(dtype=dtype,T=T,K=K):
                            q=torch.linspace(-8.,1.,T*K).reshape(T,K)
                            transition=torch.linspace(-40.,2.,K*K).reshape(K,K)
                            pi=torch.linspace(-35.,0.,K)
                            self.assert_values(q,transition,pi)
        finally:
            torch.set_default_dtype(saved)

    def test_noncontiguous_and_nonfinite_scores(self):
        q=torch.arange(24.).reshape(3,8).T
        q[2,1]=-torch.inf
        self.assertFalse(q.is_contiguous())
        self.assert_values(q,torch.tensor([[0.,-20.,-50.],[-1.,0.,-2.],[-4.,-2.,0.]]),torch.zeros(3))

    def test_nan_scores_raise_an_actionable_error(self):
        q = torch.zeros(3, 2)
        q[1, 0] = torch.nan
        state = owner(q, torch.zeros(2, 2), torch.zeros(2))
        for method in (ModelState.forward, ModelState.backward):
            with self.subTest(method=method.__name__), self.assertRaisesRegex(FloatingPointError, 'observation 1'):
                method(state)

    def test_gradient_inputs_use_original_indexed_path(self):
        for field in ('q','transition','pi'):
            values={'q':torch.linspace(-3.,0.,12).reshape(4,3),
                    'transition':torch.linspace(-4.,0.,9).reshape(3,3),
                    'pi':torch.zeros(3)}
            values[field].requires_grad_(True)
            # Python copy_ only appears on the candidate path. Original indexed
            # writes happen internally in PyTorch, so this detects dispatch.
            state=owner(**values)
            with patch.object(torch.Tensor,'copy_',side_effect=AssertionError('gradient fast path')):
                ModelState.forward(state)
                if field != 'pi':
                    ModelState.backward(state)

    def test_tensor_subclass_uses_original_indexed_path(self):
        class CustomTensor(torch.Tensor):
            pass
        q=torch.zeros((4,3)).as_subclass(CustomTensor)
        state=owner(q,torch.zeros((3,3)),torch.zeros(3))
        with patch.object(torch.Tensor,'copy_',side_effect=AssertionError('subclass fast path')):
            ModelState.forward(state)
            ModelState.backward(state)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_exact_cuda_values(self):
        self.assert_values(torch.linspace(-4.,0.,24,device='cuda').reshape(8,3),
                           torch.eye(3,device='cuda'),torch.zeros(3,device='cuda'))
