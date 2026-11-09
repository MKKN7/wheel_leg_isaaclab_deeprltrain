import copy
import torch
import pytest
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import network_gate,state_network_gate

def test_nested_scope_exact_endpoints_gradient_and_budget():
 h=PhysicsAdaptation();h.install_joint_delay_network([.08,.05,.1,.08,.05,.1]);h.install_joint_delay_scope([-1.5,-1.4,-.1,0,.01,.05]);h.install_joint_delay_reference();h.install_joint_turn_reference([.9,1,1.5,1.6,-1.6,-1.4,-.6,-.45]);source=copy.deepcopy(h)
 h.install_joint_turn_update_scope([1.3,1.45,1.5,1.6,-1.6,-1.45,-1.35,-1.2])
 with torch.no_grad():h.joint_delay_reference_network[-1].bias.add_(.3)
 learned=copy.deepcopy(source);learned.joint_delay_reference_network.load_state_dict(h.joint_delay_reference_network.state_dict())
 o=torch.zeros(6,25);o[:,7]=torch.tensor([1,1.5,1.5,1.5,1.4,-.6])*.5;o[:,8]=torch.tensor([-.8,-.6,-1.4,1.4,-1.4,0.])
 a=h.correction(o);old=source.correction(o);new=learned.correction(o)
 torch.testing.assert_close(a[[0,1,3,5]],old[[0,1,3,5]],atol=0,rtol=0);torch.testing.assert_close(a[2],new[2],atol=0,rtol=0)
 g=network_gate(h,'joint_delay_reference_network',o);assert 0<g[4]<1
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};torch.testing.assert_close(g,state_network_gate(state,'rmuc_physics_adaptation.joint_delay_reference_network',o))
 h.zero_grad();a[[0,1,3,5]].sum().backward();assert all(x.grad is None or not x.grad.any() for x in h.joint_delay_reference_network.parameters())
 assert (a.abs()<=h.correction_bound()+1e-6).all()
 with pytest.raises(ValueError):h.install_joint_turn_update_scope([0]*8)
