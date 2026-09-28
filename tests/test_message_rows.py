"""P02: exact recurrence values plus untouched autograd/subclass fallback."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.characterization._support import ModelState, torch


def safe_exp(x):
    return torch.nan_to_num(torch.exp(torch.sub(x, torch.atleast_2d(torch.max(x, axis=1)[0]).T)), 1e-8)


def original(q, transition, pi):
    """Literal pre-P02 cold forward/backward recurrence and normalization."""
    T,K=q.shape
    fmsg=torch.zeros((T,K),device=q.device)
    marg=torch.zeros(T,device=q.device)
    pi_=torch.exp(pi)
    PiTMat=safe_exp(transition.T)
    q_=safe_exp(q)
    PiTMat[PiTMat < 1e-6] += 1e-4
    pi_[pi_ < 1e-10] += 1e-4
    for t in range(T):
        if t == 0:
            fmsg[t]=pi_*q_[0]
        else:
            fmsg[t]=torch.matmul(PiTMat,fmsg[t-1])*q_[t]
        marg[t]=torch.sum(fmsg[t])
        fmsg[t] /= marg[t]
    PiMat=safe_exp(transition)
    q_=safe_exp(q)
    bmsg=torch.ones((T,K),device=q.device)
    PiMat[PiMat < 1e-5] += 1e-4
    for t in range(T-2,-1,-1):
        bmsg[t]=torch.matmul(PiMat,bmsg[t+1]*q_[t+1])
        bmsg[t] /= torch.sum(bmsg[t][:-1])
    return fmsg,marg,bmsg


def owner(q,transition,pi):
    return SimpleNamespace(q=[q],trans_A=transition,pi=[pi],M=q.shape[1],
                           device=q.device,fmsg=None,verbose=False,
                           compute_trans_pi=lambda k,p:p,
                           compute_trans_A=lambda k:transition,
                           cond_cuda=lambda value:value)


class MessageRowTests(unittest.TestCase):
    def assert_values(self,q,transition,pi):
        state=owner(q,transition,pi)
        expected=original(q,transition,pi)
        with redirect_stdout(io.StringIO()):
            actual=(*ModelState.forward(state),ModelState.backward(state))
        for a,e in zip(actual,expected):
            torch.testing.assert_close(a,e,rtol=0,atol=0,equal_nan=True)
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
        q[3,2]=torch.nan
        self.assertFalse(q.is_contiguous())
        self.assert_values(q,torch.tensor([[0.,-20.,-50.],[-1.,0.,-2.],[-4.,-2.,0.]]),torch.zeros(3))

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
