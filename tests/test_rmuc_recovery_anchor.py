import importlib.util
from pathlib import Path
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'scripts/rsl_rl/rmuc_recovery_anchor.py'
spec=importlib.util.spec_from_file_location('recovery_anchor',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_gradient_matches_explicit_auxiliary_loss_and_preserves_unprotected_rows():
    torch.manual_seed(5)
    weight=torch.randn(3,6,requires_grad=True)
    duplicate=weight.detach().clone().requires_grad_(True)
    obs=torch.randn(8,3);reference=torch.randn(8,6)
    mask=torch.tensor([True,False,True,False,False,True,True,False])[:,None]
    mean=obs@weight;expected=obs@duplicate
    unchanged=mean.detach().clone()
    recorded=m.regularize_masked_mean(mean,reference,mask,2.)
    mean.square().mean().backward()
    penalty=2.*((expected-reference).square()*mask).sum()/(mask.sum()*6)
    (expected.square().mean()+penalty).backward()
    torch.testing.assert_close(weight.grad,duplicate.grad)
    assert recorded==pytest.approx(float(penalty.detach()))
    assert torch.equal(mean.detach(),unchanged)


@pytest.mark.parametrize('coefficient,selected',[(0.,True),(2.,False)])
def test_disabled_or_empty_mask_is_gradient_noop(coefficient,selected):
    mean=torch.ones(4,6,requires_grad=True)
    mask=torch.full((4,1),selected,dtype=torch.bool)
    m.regularize_masked_mean(mean,torch.zeros_like(mean),mask,coefficient)
    mean.sum().backward()
    assert torch.equal(mean.grad,torch.ones_like(mean))


def test_mask_survives_storage_style_shuffle():
    from tensordict import TensorDict
    data=TensorDict({'policy':torch.arange(12.).reshape(2,3,2),
                    m.MASK_KEY:torch.tensor([True,False,True])[None,:,None].expand(2,-1,-1)},batch_size=[2,3])
    flat=data.flatten(0,1);indices=torch.tensor([5,1,4,0,3,2])
    shuffled=flat[indices]
    expected=(shuffled['policy'][:,0].long()//2)%3!=1
    assert torch.equal(shuffled[m.MASK_KEY].flatten(),expected)
