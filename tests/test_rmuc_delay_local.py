import sys
from pathlib import Path
import copy
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import state_network_gate

def head():
 h=PhysicsAdaptation();h.install_alignment_network([.01]*6,[0,.1,1.5,1.7]);h.install_timing_network([0,0,.03,0,0,.03]);h.install_timing_update_scope([2.1,2.4,2.6,2.8]);return h

def test_local_zero_preserves_old_and_updated_output_is_scoped():
 h=head();x=torch.zeros(10,25);x[:,7]=torch.tensor([-.4,0,.25,1.5,1.55,1.675,1.8,2.5,1.675,1.675])*.5;x[8,8]=.1;x[9,8]=-.1
 before=h.correction(x).detach().clone();h.install_delay_local_network([.01,.01,.02,.01,.01,.02],[1.55,1.65,1.7,1.8]);assert torch.equal(before,h.correction(x))
 for n,v in h.named_parameters():v.requires_grad_(n.startswith('delay_local_network.'))
 with torch.no_grad():h.delay_local_network[-1].bias.fill_(1)
 after=h.correction(x);outside=torch.tensor([0,1,2,3,4,6,7,8,9]);assert torch.equal(before[outside],after[outside]);assert (after[5]-before[5]).abs().max()>0
 assert ((after-before).abs()<=h.delay_local_budget+1e-7).all()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.delay_local_network',x),h.delay_local_gate(x))
 h2=head();h2.install_delay_local_network(h.delay_local_budget,h.delay_local_domain);h2.load_state_dict(h.state_dict());assert torch.equal(h2.correction(x),after)

def test_invalid_and_duplicate_rejected():
 h=head()
 with pytest.raises(ValueError):h.install_delay_local_network([.04]*6,[1.5,1.6,1.7,1.8])
 with pytest.raises(ValueError):h.install_delay_local_network([.01]*6,[1.5,1.6,1.6,1.8])
 h.install_delay_local_network([.01]*6,[1.5,1.6,1.7,1.8])
 with pytest.raises(ValueError):h.install_delay_local_network([.01]*6,[1.5,1.6,1.7,1.8])
