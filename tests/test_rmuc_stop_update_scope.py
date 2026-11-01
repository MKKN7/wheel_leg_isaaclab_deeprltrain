import sys
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation


def head():
    torch.manual_seed(71);h=PhysicsAdaptation();h.install_stop_network([.12,.08,.2,.12,.08,.2])
    with torch.no_grad():h.stop_network[-1].weight.normal_(0,.1)
    h.install_stop_budget_anchor([.08,.08,.2,.08,.08,.2]);return h


def test_scope_preserves_initial_outputs_and_freezes_low_speed():
    h=head();o=torch.randn(500,25)*.1;o[:,7:9]=0;o[:,2]=0;o[:,6]=torch.linspace(-2,1,500)
    reference=h.stop_residual(o).detach();h.install_stop_update_scope([.58,.68])
    assert torch.equal(reference,h.stop_residual(o))
    with torch.no_grad():h.stop_network[-1].bias.add_(.3)
    low=(-o[:,6]/2.1<=.58);v=h.stop_residual(o)
    assert torch.equal(v[low],reference[low]);assert not torch.equal(v[~low],reference[~low])
    assert (v.abs()<=h.stop_budget+1e-7).all()
    v[low].sum().backward();assert all(p.grad is None or p.grad.count_nonzero()==0 for p in h.stop_network.parameters())
    assert all(not p.requires_grad for p in h.stop_update_anchor.parameters())


def test_scope_roundtrip_and_valid_bounds():
    h=head();h.install_stop_update_scope([.58,.68]);other=head();other.install_stop_update_scope([.58,.68]);other.load_state_dict(h.state_dict())
    x=torch.randn(30,25);assert torch.equal(h.stop_residual(x),other.stop_residual(x))
    for d in [[.7,.6],[0,.6],[.6,1.2],[float('nan'),.7]]:
        with pytest.raises(ValueError):head().install_stop_update_scope(d)
