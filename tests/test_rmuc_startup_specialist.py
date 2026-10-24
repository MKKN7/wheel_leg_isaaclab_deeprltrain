import sys
from pathlib import Path
import torch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import state_network_gate

def test_startup_scope_zero_migration_and_gradient():
 h=PhysicsAdaptation();x=torch.zeros(8,25);x[:,7]=torch.tensor([-.4,0,.1,.25,.5,.6,2.5,.25])*.5;x[-1,8]=.2
 before=h.correction(x).detach();h.install_startup_network([.04,.04,.1,.04,.04,.1],[0,.05,.5,.6]);assert torch.equal(before,h.correction(x))
 for n,p in h.named_parameters():p.requires_grad_(n.startswith('startup_network.'))
 with torch.no_grad():h.startup_network[-1].bias.fill_(1)
 delta=h.correction(x)-before;assert torch.equal(delta[[0,1,5,6,7]],torch.zeros(5,6));assert delta[2:5].abs().min()>0
 assert (delta.abs()<=h.startup_budget+1e-7).all();h.correction(x).sum().backward();assert h.startup_network[-1].bias.grad.abs().sum()>0
 st={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};assert torch.equal(state_network_gate(st,'rmuc_physics_adaptation.startup_network',x),h.startup_gate(x))
 h2=PhysicsAdaptation();h2.install_startup_network(h.startup_budget,h.startup_domain);h2.load_state_dict(h.state_dict());assert torch.equal(h.correction(x),h2.correction(x))

def test_invalid_rejected():
 h=PhysicsAdaptation()
 with pytest.raises(ValueError):h.install_startup_network([.3]*6,[0,.1,.5,.6])
 with pytest.raises(ValueError):h.install_startup_network([.1]*6,[0,.1,2,2.2])
