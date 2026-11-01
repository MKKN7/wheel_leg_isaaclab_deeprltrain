import sys,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation


def test_rebound_zero_init_and_shared_budget():
    torch.manual_seed(31);h=PhysicsAdaptation();h.install_stop_network([.12,.08,.2,.12,.08,.2])
    with torch.no_grad():h.stop_network[-1].weight.normal_(0,.2)
    x=torch.randn(300,25)*.1;x[:,2]=0;x[:,7:9]=0;x[:,5]=-1
    x[100:200,3]=.3;x[200:,6]=.6
    old=h.stop_residual(x).detach();h.install_stop_rebound([math.radians(11.5),math.radians(13.5),.15,.2]);assert torch.equal(old,h.stop_residual(x))
    gate=h.stop_rebound_gate(x)
    with torch.no_grad():h.stop_rebound_network[-1].bias.fill_(100)
    out=h.stop_residual(x);assert torch.equal(out[gate==0],old[gate==0]);assert (out.abs()<=h.stop_budget+1e-6).all()
    x[:,7]=.5;assert torch.equal(h.stop_residual(x),torch.zeros_like(out))


def test_rebound_gradient_available_in_failed_states():
    h=PhysicsAdaptation();h.install_stop_network([.12,.08,.2,.12,.08,.2]);h.requires_grad_(False);h.install_stop_rebound([.2,.24,.15,.2]);h.stop_rebound_network.requires_grad_(True)
    x=torch.zeros(2,25);x[:,5]=-1;x[0,3]=.3;x[1,6]=.63
    h.stop_residual(x).sum().backward();assert h.stop_rebound_network[-1].bias.grad.abs().sum()>0
