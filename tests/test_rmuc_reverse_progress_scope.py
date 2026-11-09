"""Run only AFTER applying pending_progress_scope.patch following MC review."""
import torch
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import network_gate,state_network_gate
from rmuc_command_adapter import install_rmuc_physics_adaptation

def test_progress_freezes_all_previous_scope_transitions_and_matches_gradient():
 torch.manual_seed(11856)
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_update_scope([-.6,-.45]);h.install_reverse_extension_scope([-.8,-.65]);h.install_reverse_fast_update_scope([-.975,-.95]);h.install_reverse_coordination_scope([-1.025,-1.]);h.install_reverse_frontier_preservation([-1.1,-1.075]);h.install_reverse_frontier_update_scope([-1.35,-1.3]);h.install_reverse_frontier_retention([-1.5,-1.45]);h.reverse_frontier_anchor[-1].bias.data.add_(.3)
 x=torch.randn(2048,25)*.3;x[:,7]=torch.linspace(-1.8,2.,len(x))*.5;x[:,8]=-x[:,2]/.25
 before=h.correction(x).detach().clone();h.install_reverse_frontier_progress([-1.6,-1.5]);assert torch.equal(h.correction(x),before)
 h.reverse_frontier_anchor[-1].bias.data.add_(.2);after=h.correction(x);keep=x[:,7]>=-.75;assert torch.equal(before[keep],after[keep]);assert not torch.equal(before[~keep],after[~keep])
 h.reverse_frontier_anchor.requires_grad_(True);h.reverse_frontier_anchor[-1].weight.data.zero_();h.reverse_frontier_anchor[-1].bias.data.zero_()
 o=torch.zeros(7,25);o[:,7]=torch.tensor([-1.8,-1.6,-1.55,-1.5,-1.475,-1.4,0])*.5
 actual=torch.stack([torch.autograd.grad(h.correction(o[i:i+1])[0,2],h.reverse_frontier_anchor[-1].bias)[0][2]/.1 for i in range(len(o))]);state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 torch.testing.assert_close(actual,network_gate(h,'reverse_frontier_anchor',o),atol=1e-6,rtol=1e-6);torch.testing.assert_close(actual,state_network_gate(state,'rmuc_physics_adaptation.reverse_frontier_anchor',o),atol=1e-6,rtol=1e-6);assert not actual[3:].any()
 a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_position_authority=torch.nn.Identity();r=install_rmuc_physics_adaptation(a,state);r.load_state_dict(h.state_dict(),strict=True);assert torch.equal(r.correction(x),h.correction(x));assert not any(p.requires_grad for p in r.reverse_frontier_progress_anchor.parameters())
