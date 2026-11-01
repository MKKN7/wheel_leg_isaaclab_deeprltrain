import copy
import torch
import pytest
from rmuc_physics_adaptation import PhysicsAdaptation

def test_identity_directional_room_and_frozen_anchor():
    torch.manual_seed(30)
    h=PhysicsAdaptation();h.install_stop_network([.08,.08,.2,.08,.08,.2])
    with torch.no_grad():h.stop_network[-1].bias.fill_(1.)
    x=torch.randn(200,25);x[:,7]=0;x[:,2]=0;x[:,8]=0
    old=h.stop_residual(x).detach();budget=h.stop_budget.clone()
    h.stop_budget[[0,3]]=.12;h.install_stop_budget_anchor(budget)
    torch.testing.assert_close(h.stop_residual(x),old,atol=1e-8,rtol=0)
    assert all(not p.requires_grad for p in h.stop_budget_anchor.parameters())
    before={k:v.clone() for k,v in h.stop_budget_anchor.state_dict().items()}
    with torch.no_grad():h.stop_network[-1].bias[[0,3]]+=.5
    new=h.stop_residual(x)
    assert (new[:,[0,3]]>old[:,[0,3]]).all()
    assert (new.abs()<=h.stop_budget+1e-7).all()
    assert all(torch.equal(before[k],v) for k,v in h.stop_budget_anchor.state_dict().items())
    new.sum().backward();assert h.stop_network[-1].weight.grad is not None
    x[:,7]=.5;assert torch.equal(h.stop_residual(x),torch.zeros_like(old))
    other=PhysicsAdaptation();other.install_stop_network(h.stop_budget);other.install_stop_budget_anchor(budget);other.load_state_dict(h.state_dict())
    assert torch.equal(h.stop_residual(x),other.stop_residual(x))

def test_invalid_anchor_rejected():
    h=PhysicsAdaptation()
    with pytest.raises(ValueError):h.install_stop_budget_anchor([0]*6)
    h.install_stop_network([.08]*6)
    with pytest.raises(ValueError):h.install_stop_budget_anchor([.1]*6)
