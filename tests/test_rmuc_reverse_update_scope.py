import sys,torch,pytest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation

def test_scope_preserves_old_commands_and_bounds():
 h=PhysicsAdaptation();h.install_reverse_network([.2]*6,[-1.2,-1,-.02,0]);x=torch.randn(100,25);x[:,7]=torch.linspace(-.6,.4,100)
 with torch.no_grad():h.reverse_network[-1].bias.fill_(.2)
 before=h.reverse_output(x).detach();h.install_reverse_update_scope([-.6,-.45]);assert torch.equal(before,h.reverse_output(x))
 with torch.no_grad():h.reverse_network[-1].bias.fill_(.5)
 out=h.reverse_output(x);mask=x[:,7]/.5>=-.45
 assert torch.equal(out[mask],before[mask]);assert (out.abs()<=1).all()
 assert not torch.equal(out[~mask],before[~mask]);out[mask].sum().backward()
 assert all(p.grad is None or not p.grad.count_nonzero() for p in h.reverse_network.parameters())
 with pytest.raises(ValueError):h.install_reverse_update_scope([-.6,-.45])


def test_extension_preserves_old_low_and_fast_reverse_after_update():
 h=PhysicsAdaptation();h.install_reverse_network([.2]*6,[-1.2,-1,-.02,0]);h.install_reverse_update_scope([-.6,-.45])
 with torch.no_grad():h.reverse_network[-1].bias.fill_(.2)
 x=torch.randn(100,25);x[:,7]=torch.linspace(-.5,.1,100)
 before=h.reverse_output(x).detach();h.install_reverse_extension_scope([-.8,-.65]);assert torch.equal(before,h.reverse_output(x))
 with torch.no_grad():h.reverse_network[-1].bias.fill_(.7)
 after=h.reverse_output(x);old=x[:,7]/.5>=-.65
 assert torch.equal(before[old],after[old]);assert not torch.equal(before[~old],after[~old])
 after[old].sum().backward();assert all(p.grad is None or not p.grad.count_nonzero() for p in h.reverse_network.parameters())
