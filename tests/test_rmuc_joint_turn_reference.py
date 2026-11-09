import copy
import pytest
import torch
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import network_gate

def make():
 h=PhysicsAdaptation();h.install_joint_delay_network([.08,.05,.1,.08,.05,.1]);h.install_joint_delay_scope([-1.5,-1.4,-.1,0,.01,.05]);h.install_joint_delay_reference();return h

def test_turn_update_preserves_straight_positive_reverse_and_far_negative():
 h=make();source=copy.deepcopy(h);h.install_joint_turn_reference([.9,1.,1.5,1.6,-1.6,-1.4,-.6,-.45])
 with torch.no_grad():h.joint_delay_reference_network[-1].bias.add_(.5)
 learned=copy.deepcopy(source);learned.joint_delay_reference_network.load_state_dict(h.joint_delay_reference_network.state_dict())
 obs=torch.zeros(7,25);obs[:,7]=torch.tensor([.4,1.,1.5,-.6,1.5,1.,1.5])*.5;obs[:,8]=torch.tensor([0,1.4,-1.8,0,-.6,-1.4,-.5])
 a=h.correction(obs);torch.testing.assert_close(a[:4],source.correction(obs)[:4],atol=0,rtol=0);torch.testing.assert_close(a[4:6],learned.correction(obs)[4:6],atol=0,rtol=0)
 assert (a.abs()<=h.correction_bound()+1e-7).all()
 torch.testing.assert_close(network_gate(h,'joint_delay_reference_network',obs)[:6],torch.tensor([0.,0.,0.,0.,1.,1.]))
 with pytest.raises(ValueError):h.install_joint_turn_reference([.9,1.,1.5,1.6,-1.6,-1.4,-.6,-.45])

def test_invalid_box_rejected():
 for box in [[0]*8,[.9,1.,1.5,1.6,-2.1,-1.4,-.6,-.45]]:
  with pytest.raises(ValueError):make().install_joint_turn_reference(box)
