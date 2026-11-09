import sys,copy
from pathlib import Path
import pytest,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_command_adapter import install_rmuc_physics_adaptation
from rmuc_physics_selection import network_gate,state_network_gate

def test_nested_negative_extension_keeps_inherited_bound_and_old_domain():
 torch.manual_seed(83);inner=PhysicsAdaptation();middle=PhysicsAdaptation(inner,update_domain=[1.5,1.7,.45,.6]);outer=PhysicsAdaptation(middle,update_domain=[.25,.45,.35,.5]);original=copy.deepcopy(outer)
 for h in [inner,middle]:
  h.network[-1].bias.data.fill_(2.)
 original.load_state_dict(outer.state_dict())
 x=torch.randn(2048,25)*.2;x[:,7]=torch.linspace(-2.,2.,len(x))*.5;x[:,8]=-x[:,2]/.25;before=outer.correction(x).detach()
 inner.install_legacy_reverse_gate_extension([1.8,2.]);middle.install_legacy_reverse_gate_extension([1.8,2.]);after=outer.correction(x)
 old=x[:,7]>=-1.5*.5;assert torch.equal(before[old],after[old]);assert (after.abs()<=outer.budget+1e-6).all()
 fast=(x[:,7]<-1.7*.5)&(x[:,7]>=-1.8*.5);assert after[fast,2].min()>.3;assert before[fast].abs().max()<1e-6
 turn=x.clone();turn[:,8]+=.3;assert torch.equal(original.correction(turn),outer.correction(turn))
 z=torch.zeros(5,25);z[:,7]=torch.tensor([-1.5,-1.6,-1.8,-2.,1.6])*.5
 torch.testing.assert_close(inner.update_gate(z),torch.tensor([1.,1.,1.,0.,.5]),atol=1e-6,rtol=0.)
 assert not outer.update_gate(z).any() # narrower old stage is not broadened
 state={'rmuc_physics_adaptation.'+k:v for k,v in outer.state_dict().items()};path='rmuc_physics_adaptation.frozen_source.frozen_source.network';assert torch.equal(state_network_gate(state,path,z),network_gate(inner,'network',z))
 a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_position_authority=torch.nn.Identity();restored=install_rmuc_physics_adaptation(a,state);restored.load_state_dict(outer.state_dict(),strict=True);assert torch.equal(restored.correction(x),after)
 for boundary in [-1.5,-1.8,-2.]:
  o=torch.zeros(1,25,requires_grad=True)
  with torch.no_grad():o[:,7]=boundary*.5
  assert torch.autograd.grad(inner.update_gate(o).sum(),o)[0][:,7].abs().max()<1e-5

@pytest.mark.parametrize('domain',[[1.4,2.],[1.8,1.8],[1.8,2.3],[float('nan'),2.],[1.8]])
def test_invalid_extension_rejected(domain):
 with pytest.raises(ValueError):PhysicsAdaptation().install_legacy_reverse_gate_extension(domain)
