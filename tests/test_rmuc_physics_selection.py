import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from rmuc_physics_selection import validate_paths, state_network_gate, network_gate
from rmuc_training_contract import validate_trainable_target_gate


def test_nested_head_reachable_when_outer_gate_zero():
 inner=PhysicsAdaptation()
 outer=PhysicsAdaptation(frozen_source=inner,update_domain=[.3,.4,.3,.4])
 state={'rmuc_physics_adaptation.'+k:v for k,v in outer.state_dict().items()}
 x=torch.zeros(2,25);x[:,7]=.5;x[:,8]=torch.tensor([.1,-.1])
 assert not outer.update_gate(x).any()
 path='rmuc_physics_adaptation.frozen_source.network'
 assert torch.equal(state_network_gate(state,path,x),inner.update_gate(x))
 assert validate_trainable_target_gate(state,1,.1,physics_only=True,physics_modules=[path])['passed']
 with pytest.raises(ValueError):validate_trainable_target_gate(state,1,.1,physics_only=True)
 for bad in ['mlp','rmuc_physics_adaptation','rmuc_physics_adaptation.not_present']:
  with pytest.raises(ValueError):validate_paths([bad],state)
 with pytest.raises(ValueError):validate_paths([path,path],state)


def test_nested_optimizer_freezes_other_networks_and_inactive_adam(tmp_path):
 from rmuc_position_authority import PositionAuthorityHead
 from rmuc_local_authority import LocalAuthorityCorrection
 from rmuc_position_authority_training import prepare_authority_optimizer,configure_authority_audit
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_turn_adapter=torch.nn.Linear(25,6)
 actor.rmuc_position_authority=PositionAuthorityHead();actor.rmuc_position_authority.local_correction=LocalAuthorityCorrection()
 actor.rmuc_physics_adaptation=PhysicsAdaptation(frozen_source=PhysicsAdaptation(),update_domain=[.3,.4,.3,.4]);actor.is_recurrent=False
 critic=torch.nn.Linear(25,1);x=torch.zeros(2,25);x[:,7]=.5;x[:,8]=.1
 alg=SimpleNamespace(get_policy=lambda:actor,schedule='fixed',critic=critic,storage=SimpleNamespace(observations={'policy':x[None]},actions=torch.zeros(1,2,6),distribution_params=(torch.zeros(1,2,6),torch.ones(1,2,6)),advantages=torch.ones(1,2,1)))
 runner=SimpleNamespace(alg=alg,logger=SimpleNamespace(log_dir=tmp_path),current_learning_iteration=0)
 path='rmuc_physics_adaptation.frozen_source.network'
 prepare_authority_optimizer(runner,actor.state_dict(),physics_only=True,physics_modules=[path])
 assert {id(p) for p in alg.optimizer.param_groups[0]['params']}=={id(p) for p in actor.get_submodule(path).parameters()}
 def step():
  alg.optimizer.zero_grad(set_to_none=False)
  y=actor.rmuc_physics_adaptation(x,actor.rmuc_position_authority(x,actor.mlp(x)))
  ((y-.5).square().mean()+critic(x).square().mean()).backward();alg.optimizer.step();return {}
 alg.update=step;configure_authority_audit(runner)
 before={k:v.clone() for k,v in actor.state_dict().items()};alg.update()
 changed=[k for k,v in actor.state_dict().items() if not torch.equal(v,before[k])]
 assert changed and all(k.startswith(path+'.') for k in changed)
 before={k:v.clone() for k,v in actor.state_dict().items()};x[:,7]=1.
 alg.update()
 assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())


def test_network_signed_extension_freezes_outside_and_preserves_budget():
 import copy
 torch.manual_seed(8)
 original=PhysicsAdaptation()
 with torch.no_grad():original.network[-1].weight.normal_(0,.02)
 head=PhysicsAdaptation(network_update_domain=[1.,1.1,1.5,1.6,.4,.55,1.,1.1])
 head.network.load_state_dict(original.network.state_dict())
 head.network_anchor.load_state_dict(original.network.state_dict())
 x=torch.randn(6,25)*.1;x[:,2]=0
 x[:,7]=torch.tensor([1.,1.25,1.5,1.25,.5,1.65])*.5
 x[:,8]=torch.tensor([.8,.6,.8,-.6,.6,.6])
 before=original.correction(x)
 assert torch.equal(head.correction(x),before)
 with torch.no_grad():head.network[-1].bias.add_(.1)
 y=head.correction(x);outside=head.network_update_gate(x)==0
 assert torch.equal(y[outside],before[outside])
 assert not torch.equal(y[~outside],before[~outside])
 assert (y.abs()<=head.budget+1e-7).all()
 assert all(not p.requires_grad for p in head.network_anchor.parameters())
 state={'rmuc_physics_adaptation.'+k:v for k,v in head.state_dict().items()}
 assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.network',x),network_gate(head,'network',x))
 clone=PhysicsAdaptation(network_update_domain=head.network_update_domain)
 clone.load_state_dict(head.state_dict(),strict=True)
 assert torch.equal(clone.correction(x),y)


def test_reverse_anchor_selection_and_frozen_idle_sequence():
 from rmuc_training_contract import validate_trainable_sequence_gates
 h=PhysicsAdaptation();h.install_reverse_network([.2]*6,[-1.2,-1,-.02,0]);h.install_reverse_update_scope([-.6,-.45])
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};x=torch.zeros(2,25);x[:,7]=torch.tensor([-.125,-.4])
 low='rmuc_physics_adaptation.reverse_update_anchor';high='rmuc_physics_adaptation.reverse_network'
 assert torch.equal(state_network_gate(state,low,x),torch.tensor([1.,0.]))
 assert torch.equal(state_network_gate(state,high,x),torch.tensor([0.,1.]))
 rows=validate_trainable_sequence_gates(state,[[0,0,0],[2,-.25,0],[5,-.8,0]],physics_only=True,physics_modules=[low,high])
 assert all(r['passed'] for r in rows)
 assert rows[0]['phase']=='frozen_teacher_idle'


def test_reverse_extended_gate_preserves_old_domain_and_covers_target():
 from rmuc_training_contract import validate_trainable_sequence_gates
 old=PhysicsAdaptation();old.install_reverse_network([.1]*6,[-1.2,-1.,-.02,0.])
 new=PhysicsAdaptation();new.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.])
 x=torch.zeros(101,25);x[:,7]=torch.linspace(-1,0,101)*.5
 assert torch.equal(old.reverse_gate(x),new.reverse_gate(x))
 x=torch.zeros(3,25);x[:,7]=torch.tensor([-1.,-1.2,-1.5])*.5
 assert new.reverse_gate(x).tolist()==[1.,1.,1.]
 assert old.reverse_gate(x).tolist()==[1.,0.,0.]
 state={'rmuc_physics_adaptation.'+k:v for k,v in new.state_dict().items()}
 result=validate_trainable_sequence_gates(state,[[0,0,0],[2,-1.5,0]],physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_network'])
 assert all(r['passed'] for r in result)
 # Expanded course limits permit a partially active -1.6 command, but this
 # historical checkpoint still has no reverse support at the new -1.8 target.
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,[[0,0,0],[2,-1.8,0]],physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_network'])


def test_explicit_reverse180_domain_retains_old_outputs_and_does_not_route_by_measured_speed():
 from rmuc_training_contract import validate_trainable_sequence_gates
 old=PhysicsAdaptation();old.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.])
 with torch.no_grad():old.reverse_network[-1].bias.fill_(.2)
 new=PhysicsAdaptation();new.install_reverse_network([.1]*6,[-2.,-1.8,-.02,0.])
 new.load_state_dict(old.state_dict())
 with torch.no_grad():new.reverse_domain.copy_(torch.tensor([-2.,-1.8,-.02,0.]))
 x=torch.randn(1024,25)*.2;x[:,7]=torch.linspace(-1.5,3.2,len(x))*.5;x[:,8]=-x[:,2]/.25
 assert torch.equal(old.correction(x),new.correction(x))
 edge=torch.zeros(5,25);edge[:,7]=torch.tensor([-1.5,-1.7,-1.8,-2.,0.])*.5
 assert new.reverse_gate(edge).tolist()==[1.,1.,1.,0.,0.]
 # obs6 represents velocity error; actual backward motion under a zero command
 # must not be confused with a negative command activating this expert.
 edge[-1,6]=2.;assert new.reverse_gate(edge)[-1]==0
 state={'rmuc_physics_adaptation.'+k:v for k,v in new.state_dict().items()}
 result=validate_trainable_sequence_gates(state,[[0,0,0],[2,-1.7,0],[5,-1.8,0]],physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_network'])
 assert all(r['passed'] for r in result)
 assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.reverse_network',edge),new.reverse_gate(edge))
 with pytest.raises(ValueError):
  validate_trainable_sequence_gates(state,[[0,0,0],[2,-1.9,0]],physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_network'])


def test_reverse_reserve_zero_init_isolation_bound_and_selection():
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.])
 x=torch.randn(100,25);x[:,7]=torch.linspace(-1.5,1,100)*.5;x[:,2]=0;x[:,8]=0
 original=h.correction(x).detach().clone();h.install_reverse_reserve([0,0,.1,0,0,.1],[-.8,-.65])
 assert torch.equal(original,h.correction(x))
 with torch.no_grad():h.reverse_reserve_network[-1].bias.fill_(-100)
 delta=h.correction(x)-original
 assert delta[:,[0,1,3,4]].abs().max()==0
 assert torch.equal(delta[:,2],delta[:,5])
 assert delta.abs().max()<=.100001
 assert delta[x[:,7]>=-.65*.5].abs().max()==0
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 path='rmuc_physics_adaptation.reverse_reserve_network'
 assert validate_paths([path],state)==[path]
 assert torch.equal(state_network_gate(state,path,x),h.reverse_reserve_gate(x))
 assert validate_trainable_target_gate(state,-.8,0,physics_only=True,physics_modules=[path])['passed']
 with pytest.raises(ValueError):validate_trainable_target_gate(state,-.6,0,physics_only=True,physics_modules=[path])


def test_reserve_sequence_explicitly_freezes_old_reverse_approach():
 from rmuc_training_contract import validate_trainable_sequence_gates
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.]);h.install_reverse_reserve([0,0,.1,0,0,.1],[-.8,-.65])
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 kwargs=dict(physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_reserve_network'])
 rows=validate_trainable_sequence_gates(state,[[0,0,0],[2,-.4,0],[5,-.6,0],[8,-.8,0],[12,-1.,0]],**kwargs)
 assert rows[2]['gates']==[0.] and rows[3]['gates']==[1.]
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,[[0,0,0],[2,-.6,0]],**kwargs)
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,[[0,0,0],[2,-.8,.1]],**kwargs)


def test_reserve_transition_preserves_fast_and_matches_slow_control():
 old=PhysicsAdaptation();old.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.]);old.install_reverse_reserve([0,0,.1,0,0,.1],[-.8,-.65])
 changed=PhysicsAdaptation();changed.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.]);changed.install_reverse_reserve([0,0,.1,0,0,.1],[-.8125,-.625])
 scoped=PhysicsAdaptation();scoped.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.]);scoped.install_reverse_reserve([0,0,.1,0,0,.1],[-.8125,-.625],[-.8,-.65,-.75,-.725])
 x=torch.zeros(1001,25);x[:,7]=torch.linspace(-1.5,.2,len(x))*.5
 g=scoped.reverse_reserve_gate(x);fast=x[:,7]<=-.75*.5;slow=x[:,7]>=-.725*.5
 assert torch.equal(g[fast],old.reverse_reserve_gate(x)[fast])
 assert torch.equal(g[slow],changed.reverse_reserve_gate(x)[slow])
 assert ((g>=0)&(g<=1)).all()
 state={'rmuc_physics_adaptation.'+k:v for k,v in scoped.state_dict().items()}
 assert torch.equal(state_network_gate(state,'rmuc_physics_adaptation.reverse_reserve_network',x),g)
 # Smooth blend meets both endpoint derivatives; no new command jump.
 for boundary,reference in [(-.75,old),(-.725,changed)]:
  probe=torch.zeros(1,25,requires_grad=True)
  with torch.no_grad():probe[:,7]=boundary*.5
  actual_grad=torch.autograd.grad(scoped.reverse_reserve_gate(probe).sum(),probe,retain_graph=True)[0]
  ref_grad=torch.autograd.grad(reference.reverse_reserve_gate(probe).sum(),probe)[0]
  torch.testing.assert_close(actual_grad,ref_grad,rtol=1e-5,atol=1e-5)
 x[:,8]=.2;assert not scoped.reverse_reserve_gate(x).any()


@pytest.mark.parametrize('invalid',[[-.8,-.65,-.7,-.75],[-.8,-.65,float('nan'),-.7],[-.8,-.65,-.75],[-.8,-.65,-.75,0.]])
def test_reserve_transition_rejects_bad_checkpoint_bounds(invalid):
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.])
 with pytest.raises(ValueError):h.install_reverse_reserve([0,0,.1,0,0,.1],[-.8125,-.625],invalid)


def test_joint_reverse_reserve_requires_matching_extension_isolation():
 from rmuc_training_contract import validate_trainable_sequence_gates
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.]);h.install_reverse_update_scope([-.6,-.45]);h.install_reverse_reserve([0,0,.1,0,0,.1],[-.8,-.65])
 kwargs=dict(physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_network','rmuc_physics_adaptation.reverse_reserve_network']);seq=[[0,0,0],[2,-.6,0],[4,-.8,0]]
 state=lambda:{'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state(),seq,**kwargs)
 h.install_reverse_extension_scope([-.8,-.65])
 result=validate_trainable_sequence_gates(state(),seq,**kwargs)
 assert result[1]['gates']==[0.] and result[2]['gates']==[1.]


def test_joint_delay_head_covers_all_commands_zero_init_and_bound():
 h=PhysicsAdaptation();x=torch.randn(128,25);x[:,7]=torch.linspace(-1.5,3.2,128)*.5;x[:,8]=torch.linspace(-8,8,128);before=h.correction(x).detach().clone()
 h.install_joint_delay_network([.08,.05,.1,.08,.05,.1])
 assert torch.equal(before,h.correction(x))
 with torch.no_grad():h.joint_delay_network[-1].bias.fill_(100.)
 delta=h.correction(x)-before
 torch.testing.assert_close(delta,h.joint_delay_budget.expand_as(delta))
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};path='rmuc_physics_adaptation.joint_delay_network'
 assert validate_paths([path],state)==[path]
 assert state_network_gate(state,path,x).min()==1


def test_joint_delay_scope_isolates_old_commands_and_clipped_yaw():
 h=PhysicsAdaptation();h.install_joint_delay_network([.08,.05,.1,.08,.05,.1]);h.install_joint_delay_scope([1.55,1.6,1.9,1.95,.01,.05])
 x=torch.zeros(10,25);x[:,7]=torch.tensor([-.8,0,1.5,1.55,1.575,1.6,1.9,1.95,1.7,1.7])*.5
 x[8,8]=.3;x[9,8]=2.;x[9,2]=-.5
 with torch.no_grad():h.joint_delay_network[-1].bias.fill_(1.)
 g=h.joint_delay_gate(x);torch.testing.assert_close(g,torch.tensor([0,0,0,0,.5,1,1,0,0,0.]),atol=1e-5,rtol=1e-5)
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 torch.testing.assert_close(state_network_gate(state,'rmuc_physics_adaptation.joint_delay_network',x),g)
 h.zero_grad();h.correction(x[g==0]).sum().backward()
 assert all(p.grad is None or p.grad.abs().max()==0 for p in h.joint_delay_network.parameters())
 h.zero_grad();h.correction(x[g==1]).sum().backward()
 assert h.joint_delay_network[-1].bias.grad.abs().sum()>0


def test_signed_joint_delay_scope_supports_idle_reverse_without_highspeed_leak():
 h=PhysicsAdaptation();h.install_joint_delay_network([.08,.05,.1,.08,.05,.1]);h.install_joint_delay_scope([-.5,-.4,.1,.2,.01,.05])
 x=torch.zeros(8,25);x[:,7]=torch.tensor([-.8,-.5,-.4,-.25,0.,.1,.2,1.6])*.5
 torch.testing.assert_close(h.joint_delay_gate(x),torch.tensor([0.,0.,1.,1.,1.,1.,0.,0.]))
 x[:,8]=.1;assert not h.joint_delay_gate(x).any()


@pytest.mark.parametrize('kind', ['turn', 'straight', 'reverse'])
def test_selectable_anchor_gates_match_actual_forward_derivatives(kind):
 """Check routing against the real forward graph, including fade regions."""
 h=PhysicsAdaptation()
 if kind=='turn':
  h.install_joint_delay_network([.1]*6)
  h.install_joint_delay_scope([-.9,-.8,-.3,-.2,.01,.05])
  h.install_joint_delay_reference()
  h.install_joint_turn_reference([.8,1.,1.5,1.7,-1.6,-1.4,-1.2,-1.])
  h.install_joint_turn_update_scope([1.2,1.25,1.5,1.6,-1.5,-1.4,-1.2,-1.1])
  commands=[[.5,.3],[1.,-1.4],[1.225,-1.4],[1.25,-1.4],[1.6,-1.3],[-.4,0]]
  names=['joint_delay_network','joint_delay_reference_network','joint_turn_source_network','joint_turn_update_anchor']
 elif kind=='straight':
  h.install_alignment_network([.1]*6,[-.01,0,.2,.3])
  h.install_straight_extension([.1]*6,[.3,.4,2.,2.2])
  h.install_straight_update_scope([1.3,1.4,1.6,1.7])
  commands=[[.2,0],[.35,0],[1.,0],[1.35,0],[1.5,0],[1.9,0],[2.1,0],[1.9,.3]]
  names=['straight_extension_network','straight_update_anchor']
 else:
  h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0])
  h.install_reverse_update_scope([-.6,-.45])
  h.install_reverse_extension_scope([-.8,-.65])
  commands=[[-.25,0],[-.5,0],[-.6,0],[-.7,0],[-.8,0],[-1.5,0],[0,0],[-.8,.3]]
  names=['reverse_network','reverse_update_anchor','reverse_extension_anchor']
 x=torch.zeros(len(commands),25)
 x[:,7]=torch.tensor([c[0]*.5 for c in commands]);x[:,8]=torch.tensor([c[1] for c in commands])
 for name in names:
  module=getattr(h,name);module.requires_grad_(True)
  with torch.no_grad():module[-1].weight.zero_();module[-1].bias.zero_()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 for name in names:
  bias=getattr(h,name)[-1].bias
  actual=torch.stack([torch.autograd.grad(h.correction(x[i:i+1])[0,2],bias)[0][2]/.1
                      for i in range(len(x))])
  path='rmuc_physics_adaptation.'+name
  torch.testing.assert_close(actual,network_gate(h,name,x),atol=2e-6,rtol=2e-6)
  torch.testing.assert_close(actual,state_network_gate(state,path,x),atol=2e-6,rtol=2e-6)
 if kind=='turn':
  assert network_gate(h,'joint_delay_reference_network',x)[0]==0
  assert network_gate(h,'joint_turn_source_network',x)[0]==1


def test_turn_source_update_scope_preserves_straight_and_checkpoint_restore():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 torch.manual_seed(61106)
 h=PhysicsAdaptation();h.install_joint_delay_network([.1]*6)
 h.install_joint_delay_scope([-.9,-.8,-.3,-.2,.01,.05]);h.install_joint_delay_reference()
 h.install_joint_turn_reference([.8,1.,1.5,1.7,-1.6,-1.4,-1.2,-1.])
 with torch.no_grad():h.joint_turn_source_network[-1].bias.normal_(0,.1)
 x=torch.randn(200,25)*.1;x[:,7]=torch.linspace(-1.5,3.2,200)*.5;x[:,2]=0;x[:,8]=0
 x[:8,7]=torch.tensor([.35,.4,.425,.45,.5,.55,.6,.65])*.5;x[:8,8]=.3
 before=h.correction(x).detach().clone()
 h.install_joint_turn_source_update_scope([.4,.45,.55,.6,.15,.2,.4,.45])
 assert torch.equal(before,h.correction(x))
 with torch.no_grad():h.joint_turn_source_network[-1].bias.add_(.4)
 after=h.correction(x);gate=h.joint_turn_source_update_gate(x)
 assert torch.equal(before[gate==0],after[gate==0])
 assert not torch.equal(before[gate==1],after[gate==1])
 assert not any(p.requires_grad for p in h.joint_turn_source_anchor.parameters())
 h.joint_turn_source_network.requires_grad_(True)
 h.zero_grad();h.correction(x[gate==0]).sum().backward()
 assert all(p.grad is None or p.grad.abs().max()==0 for p in h.joint_turn_source_network.parameters())
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 expected=network_gate(h,'joint_turn_source_network',x)
 torch.testing.assert_close(state_network_gate(state,'rmuc_physics_adaptation.joint_turn_source_network',x),expected)
 # Restore through the production installer, not a hand-built test copy.
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6))
 actor.rmuc_position_authority=torch.nn.Identity()
 installed=install_rmuc_physics_adaptation(actor,state)
 installed.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(after,installed.correction(x))
 with pytest.raises(ValueError):h.install_joint_turn_source_update_scope([.4,.45,.55,.6,.15,.2,.4,.45])
 from rmuc_training_contract import validate_trainable_sequence_gates
 selection=dict(physics_only=True,physics_modules=['rmuc_physics_adaptation.joint_turn_source_network'])
 sequence=[[0,1.7,0],[2,.5,.3]]
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,sequence,**selection)
 rows=validate_trainable_sequence_gates(state,sequence,target_only_commands=[[.5,.3]],**selection)
 assert rows[0]['phase']=='frozen_target_mask_rehearsal'
 assert rows[0]['actor_update_enabled'] is False and rows[1]['gates'][0]>0
 # The positive mirror cannot justify learning at an inactive negative target.
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,sequence,target_only_commands=[[.5,-.3]],**selection)
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,sequence,target_only_commands=[],**selection)


def test_broad_turn_anchor_preserves_straight_spin_and_reverse_and_restores():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_joint_delay_network([.1]*6)
 h.install_joint_delay_scope([-1.5,-1.4,-.1,0.,.01,.05]);h.install_joint_delay_reference()
 h.install_joint_turn_reference([.9,1.,1.5,1.6,-1.6,-1.4,-.6,-.45])
 h.install_joint_turn_source_update_scope([.4,.45,.55,.6,.15,.2,.4,.45])
 x=torch.zeros(10,25);x[:,7]=torch.tensor([-.6,0.,.5,1.,1.5,2.,.5,1.,1.5,2.])*.5
 x[:,8]=torch.tensor([0.,4.,0.,0.,0.,0.,-.3,.3,1.4,-.8])
 before=h.correction(x).detach().clone();h.install_joint_turn_anchor_scope([.1,.2,2.,2.1,.02,.05])
 assert torch.equal(before,h.correction(x))
 with torch.no_grad():h.joint_turn_source_anchor[-1].bias.add_(.2)
 after=h.correction(x).detach();assert torch.equal(before[:6],after[:6]);assert (after[6:]-before[6:]).abs().max()>0
 h.joint_turn_source_anchor.requires_grad_(True);h.zero_grad();h.correction(x[:6]).sum().backward()
 assert all(p.grad is None or p.grad.abs().max()==0 for p in h.joint_turn_source_anchor.parameters())
 path='rmuc_physics_adaptation.joint_turn_source_anchor';state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 torch.testing.assert_close(network_gate(h,'joint_turn_source_anchor',x),state_network_gate(state,path,x))
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(after,restored.correction(x))
 with pytest.raises(ValueError):h.install_joint_turn_anchor_scope([.1,.2,2.,2.1,.02,.05])


@pytest.mark.parametrize('wheel_budget',[.09,.11])
def test_reverse_reserve_frontier_preserves_old_domain_and_restores(wheel_budget):
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.])
 h.install_reverse_reserve([0,0,.1,0,0,.1],[-.85,-.6],[-.8,-.65,-.75,-.725])
 with torch.no_grad():h.reverse_reserve_network[-1].bias.fill_(-2.)
 x=torch.zeros(10,25);x[:,7]=torch.tensor([.5,0,-.65,-.75,-.85,-.875,-.9,-.925,-1.5,-.95])*.5;x[-1,8]=.3
 before=h.correction(x).detach();bound=h.correction_bound().clone()
 h.install_reverse_reserve_frontier([0,0,wheel_budget,0,0,wheel_budget],[-.925,-.875])
 after=h.correction(x).detach();assert torch.equal(before[:6],after[:6]);assert torch.equal(before[-1],after[-1])
 assert torch.equal(before[:,[0,1,3,4]],after[:,[0,1,3,4]])
 assert (after[6:9]-before[6:9]).abs().max()>0
 expected_bound=bound.clone();expected_bound[[2,5]]+=max(wheel_budget-.1,0.)
 torch.testing.assert_close(h.correction_bound(),expected_bound)
 assert (after.abs()<=h.correction_bound()+1e-6).all()
 # Full frontier output remains within the existing .15 common-wheel limit.
 assert h.reverse_reserve_output(x).abs().max()<=max(.1,wheel_budget)
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(after,restored.correction(x))
 for vx in [-.925,-.875]:
  probe=torch.zeros(1,25);probe[:,7]=vx*.5;probe.requires_grad_(True)
  actual=torch.autograd.grad(h.reverse_reserve_output(probe).sum(),probe)[0]
  # Both the original gate and frontier mix are flat at these endpoints.
  assert actual[0,7].abs()<1e-5
 bad=dict(state);del bad['rmuc_physics_adaptation.reverse_reserve_frontier_domain']
 other=torch.nn.Module();other.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));other.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(other,bad)


@pytest.mark.parametrize('budget,domain',[
 ([0,0,.16,0,0,.16],[-.925,-.875]),([.01,0,.1,0,0,.1],[-.925,-.875]),
 ([0,0,.1,0,0,.11],[-.925,-.875]),([0,0,.1,0,0,.1],[-.875,-.925]),
 ([0,0,.1,0,0,.1],[-.8,-.7]),([0,0,float('nan'),0,0,.1],[-.925,-.875])])
def test_reverse_reserve_frontier_rejects_invalid_contract(budget,domain):
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0.]);h.install_reverse_reserve([0,0,.1,0,0,.1],[-.8,-.65])
 with pytest.raises(ValueError):h.install_reverse_reserve_frontier(budget,domain)


@pytest.mark.parametrize('gain',[.8,1.,1.2,1.7,2.])
def test_reverse_base_leg_response_scope_bound_center_and_restore(gain):
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 torch.manual_seed(9)
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0])
 x=torch.randn(2048,25)*.2;x[:,7]=torch.linspace(-1.5,2.,len(x))*.5;x[:,8]=-x[:,2]/.25
 action=torch.randn(len(x),6);assert torch.equal(action,h.reverse_base_leg_response(x,action))
 center=torch.tensor([.1,.2,.3,-.1,-.2,-.3]);h.install_reverse_base_leg_response(gain,center,[-.925,-.875],.06)
 out=h.reverse_base_leg_response(x,action);protected=x[:,7]>=-.875*.5
 assert torch.equal(out[protected],action[protected]);assert torch.equal(out[:,[2,5]],action[:,[2,5]])
 assert (out-action).abs().max()<=.060001
 assert torch.equal(h.reverse_base_leg_response(x,center.expand_as(action)),center.expand_as(action))
 if gain==1:assert torch.equal(out,action)
 else:assert not torch.equal(out,action)
 x[:,8]+=.3;assert torch.equal(h.reverse_base_leg_response(x,action),action)
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 x[:,8]-=.3;assert torch.equal(restored.reverse_base_leg_response(x,action),h.reverse_base_leg_response(x,action))
 bad=dict(state);del bad['rmuc_physics_adaptation.reverse_base_leg_limit']
 other=torch.nn.Module();other.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));other.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(other,bad)


@pytest.mark.parametrize('gain,center,domain,limit',[(2.1,[0]*6,[-.925,-.875],.06),(.8,[0]*5,[-.925,-.875],.06),(.8,[0]*6,[-.8,-.7],.06),(.8,[0]*6,[-.925,-.875],.11),(.8,[0]*6,[-.925,-.875],float('nan'))])
def test_reverse_base_leg_response_invalid(gain,center,domain,limit):
 h=PhysicsAdaptation()
 with pytest.raises(ValueError):h.install_reverse_base_leg_response(gain,center,domain,limit)


def test_fast_leg_gain_preserves_slow_response_and_never_expands_action_bound():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_base_leg_response(1.5,torch.zeros(6),[-.925,-.875],.06)
 x=torch.zeros(1001,25);x[:,7]=torch.linspace(-1.8,0,len(x))*.5
 action=torch.linspace(-.15,.15,6006).reshape(1001,6);old=h.reverse_base_leg_response(x,action)
 h.install_reverse_base_leg_fast_gain(1.8,[-1.,-.975]);new=h.reverse_base_leg_response(x,action)
 assert torch.equal(old[x[:,7]>=-.975*.5],new[x[:,7]>=-.975*.5])
 assert torch.equal(old[:,[2,5]],new[:,[2,5]])
 assert (new-action).abs().max()<=.060001
 assert not torch.equal(old[x[:,7]<=-.5],new[x[:,7]<=-.5])
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict())
 assert torch.equal(new,restored.reverse_base_leg_response(x,action))
 incomplete=dict(state);del incomplete['rmuc_physics_adaptation.reverse_base_leg_fast_gain_domain']
 fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,incomplete)


def test_reverse_wheel_extension_preserves_verified_domain_and_budget_restore():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_reserve([0,0,.1,0,0,.1],[-.8,-.65]);h.install_reverse_reserve_frontier([0,0,.112,0,0,.112],[-.925,-.875]);h.install_reverse_reserve_fast_preservation([0,0,.12,0,0,.12],[-.95,-.925])
 with torch.no_grad():h.reverse_reserve_network[-1].bias.fill_(-2.)
 x=torch.zeros(2049,25);x[:,7]=torch.linspace(-1.8,0,len(x))*.5
 old=h.reverse_reserve_output(x);h.install_reverse_reserve_extension([0,0,.14,0,0,.14],[-1.05,-1.]);new=h.reverse_reserve_output(x)
 assert torch.equal(old[x[:,7]>=-.5],new[x[:,7]>=-.5])
 assert torch.equal(old[:,[0,1,3,4]],new[:,[0,1,3,4]])
 assert (new[:,[2,5]].abs()<=.140001).all()
 assert (new[x[:,7]<=-.525,2]<old[x[:,7]<=-.525,2]).all()
 assert h.correction_bound()[2]>=.24
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict())
 assert torch.equal(new,restored.reverse_reserve_output(x))
 incomplete=dict(state);del incomplete['rmuc_physics_adaptation.reverse_reserve_extension_domain']
 fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,incomplete)


def test_reverse_fast_preservation_exact_both_regions_and_restore():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 import copy
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0]);h.install_reverse_reserve([0,0,.1,0,0,.1],[-.85,-.6],[-.8,-.65,-.75,-.725]);h.install_reverse_reserve_frontier([0,0,.12,0,0,.12],[-.925,-.875])
 with torch.no_grad():h.reverse_reserve_network[-1].bias.fill_(-2)
 new=copy.deepcopy(h);new.reverse_reserve_frontier_budget[[2,5]]=.112
 scoped=copy.deepcopy(new);scoped.install_reverse_reserve_fast_preservation([0,0,.12,0,0,.12],[-.975,-.95])
 x=torch.zeros(2001,25);x[:,7]=torch.linspace(-1.5,0.,len(x))*.5
 old=h.reverse_reserve_output(x);changed=new.reverse_reserve_output(x);result=scoped.reverse_reserve_output(x)
 assert torch.equal(result[x[:,7]<=-.975*.5],old[x[:,7]<=-.975*.5]);assert torch.equal(result[x[:,7]>=-.95*.5],changed[x[:,7]>=-.95*.5])
 assert torch.equal(result[:,[0,1,3,4]],old[:,[0,1,3,4]])
 assert torch.all(result>=torch.minimum(old,changed)-1e-7) and torch.all(result<=torch.maximum(old,changed)+1e-7)
 assert torch.equal(scoped.correction_bound(),h.correction_bound())
 for vx,reference in [(-.975,h),(-.95,new)]:
  probe=torch.zeros(1,25);probe[:,7]=vx*.5;probe.requires_grad_(True)
  actual=torch.autograd.grad(scoped.reverse_reserve_output(probe).sum(),probe)[0];expected=torch.autograd.grad(reference.reverse_reserve_output(probe).sum(),probe)[0]
  torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
 state={'rmuc_physics_adaptation.'+k:v for k,v in scoped.state_dict().items()}
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity();restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(scoped.state_dict(),strict=True)
 assert torch.equal(result,restored.reverse_reserve_output(x))
 with pytest.raises(ValueError):new.install_reverse_reserve_fast_preservation([0,0,.12,0,0,.12],[-.95,-.9])


def test_reverse_fast_leg_limit_preserves_earlier_domain_wheels_and_restores():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0]);h.install_reverse_base_leg_response(1.4,torch.zeros(6),[-.925,-.875],.06)
 x=torch.zeros(1001,25);x[:,7]=torch.linspace(-1.5,0,len(x))*.5;actions=torch.ones(len(x),6)
 old=h.reverse_base_leg_response(x,actions);h.install_reverse_base_leg_fast_limit(.08,[-.95,-.925]);new=h.reverse_base_leg_response(x,actions)
 assert torch.equal(old[x[:,7]>=-.925*.5],new[x[:,7]>=-.925*.5]);assert torch.equal(old[:,[2,5]],new[:,[2,5]])
 assert (new-actions).abs().max()<=.080001
 torch.testing.assert_close((new-actions)[x[:,7]<=-.95*.5,0],torch.full_like(new[x[:,7]<=-.95*.5,0],.08))
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity();restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(new,restored.reverse_base_leg_response(x,actions))
 with pytest.raises(ValueError):PhysicsAdaptation().install_reverse_base_leg_fast_limit(.08,[-.95,-.925])
 bad=dict(state);bad['rmuc_physics_adaptation.reverse_base_leg_fast_limit']=torch.tensor(.11)
 other=torch.nn.Module();other.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));other.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(other,bad)


def test_reverse_fast_update_preserves_old_function_and_matches_gradient_gate():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-1.7,-1.5,-.02,0]);h.install_reverse_update_scope([-.6,-.45]);h.install_reverse_extension_scope([-.8,-.65])
 with torch.no_grad():h.reverse_network[-1].bias.fill_(.2)
 x=torch.zeros(10,25);x[:,7]=torch.tensor([-.6,-.8,-.9,-.925,-.95,-.975,-1.,-1.5,0.,1.5])*.5
 before=h.correction(x).detach();h.install_reverse_fast_update_scope([-.975,-.925]);assert torch.equal(before,h.correction(x))
 with torch.no_grad():h.reverse_network[-1].bias.add_(.1)
 after=h.correction(x).detach();protected=x[:,7]>=-.925*.5;assert torch.equal(before[protected],after[protected]);assert (after[~protected]-before[~protected]).abs().max()>0
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 for leaf in ['reverse_network','reverse_fast_update_anchor','reverse_extension_anchor','reverse_update_anchor']:
  module=getattr(h,leaf);module.requires_grad_(True)
  with torch.no_grad():module[-1].weight.zero_();module[-1].bias.zero_()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 for leaf in ['reverse_network','reverse_fast_update_anchor','reverse_extension_anchor','reverse_update_anchor']:
  bias=getattr(h,leaf)[-1].bias;actual=torch.stack([torch.autograd.grad(h.correction(x[i:i+1])[0,2],bias)[0][2]/.1 for i in range(len(x))]);expected=network_gate(h,leaf,x)
  torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6);torch.testing.assert_close(actual,state_network_gate(state,'rmuc_physics_adaptation.'+leaf,x),atol=2e-6,rtol=2e-6)
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity();restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(h.correction(x),restored.correction(x))
 assert not restored.reverse_fast_update_anchor[-1].bias.requires_grad
 with pytest.raises(ValueError):h.install_reverse_fast_update_scope([-.975,-.925])


def test_leg_extension_gain_preserves_minus_one_bounds_and_checkpoint_restore():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_base_leg_response(1.5,[0]*6,[-.9,-.85],.06);h.install_reverse_base_leg_fast_gain(1.8,[-1,-.975])
 x=torch.zeros(2049,25);x[:,7]=torch.linspace(-1.8,1,len(x))*.5
 actions=(.2*torch.sin(torch.arange(len(x)*6)*.03)).reshape(-1,6);old=h.reverse_base_leg_response(x,actions)
 h.install_reverse_base_leg_extension_gain(2.,[-1.025,-1.]);new=h.reverse_base_leg_response(x,actions)
 assert torch.equal(old[x[:,7]>=-.5],new[x[:,7]>=-.5])
 assert torch.equal(old[:,[2,5]],new[:,[2,5]])
 assert (new-actions).abs().max()<=.060001
 assert not torch.equal(old,new)
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 a=torch.nn.Module();a.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));a.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(a,state);restored.load_state_dict(h.state_dict())
 assert torch.equal(new,restored.reverse_base_leg_response(x,actions))
 bad=dict(state);del bad['rmuc_physics_adaptation.reverse_base_leg_extension_gain_domain']
 fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,bad)
 h2=PhysicsAdaptation();h2.install_reverse_base_leg_response(1.5,[0]*6,[-.9,-.85],.06);h2.install_reverse_base_leg_fast_gain(1.8,[-1,-.975])
 with pytest.raises(ValueError):h2.install_reverse_base_leg_extension_gain(2.01,[-1.025,-1.])
 with pytest.raises(ValueError):h2.install_reverse_base_leg_extension_gain(2.,[-1.025,-.99])


def test_reverse_coordination_protects_minus_one_under_real_updates_and_gate_audit():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_update_scope([-.6,-.45]);h.install_reverse_extension_scope([-.8,-.65]);h.install_reverse_fast_update_scope([-.975,-.95])
 with torch.no_grad():h.reverse_network[-1].bias.fill_(.2)
 x=torch.zeros(13,25);x[:,7]=torch.tensor([-1.8,-1.5,-1.1,-1.075,-1.05,-1.025,-1.0125,-1.,-.975,-.95,-.6,0.,2.])*.5
 before=h.correction(x).detach();h.install_reverse_coordination_scope([-1.025,-1.]);assert torch.equal(before,h.correction(x))
 opt=torch.optim.SGD(h.reverse_network.parameters(),lr=.2);opt.zero_grad();(-h.correction(x[:7])[:,2].mean()).backward();opt.step()
 after=h.correction(x).detach();protected=x[:,7]>=-.5;assert torch.equal(before[protected],after[protected]);assert not torch.equal(before[~protected],after[~protected])
 for name in ['reverse_network','reverse_update_anchor','reverse_extension_anchor','reverse_fast_update_anchor','reverse_coordination_anchor']:
  m=getattr(h,name);m.requires_grad_(True)
  with torch.no_grad():m[-1].weight.zero_();m[-1].bias.zero_()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 for leaf in ['reverse_network','reverse_update_anchor','reverse_extension_anchor','reverse_fast_update_anchor','reverse_coordination_anchor']:
  bias=getattr(h,leaf)[-1].bias;actual=torch.stack([torch.autograd.grad(h.correction(x[i:i+1])[0,2],bias)[0][2]/.1 for i in range(len(x))]);expected=network_gate(h,leaf,x)
  torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6);torch.testing.assert_close(actual,state_network_gate(state,'rmuc_physics_adaptation.'+leaf,x),atol=2e-6,rtol=2e-6)
 assert torch.equal(network_gate(h,'reverse_network',x)[protected],torch.zeros(protected.sum()))
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity();restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(h.correction(x),restored.correction(x));assert not restored.reverse_coordination_anchor[-1].bias.requires_grad
 incomplete=dict(state);del incomplete['rmuc_physics_adaptation.reverse_coordination_domain'];fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,incomplete)


def test_reverse_frontier_preservation_keeps_near_gains_fast_reference_and_gradients():
 import copy
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 old=PhysicsAdaptation();old.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);old.install_reverse_update_scope([-.6,-.45]);old.install_reverse_extension_scope([-.8,-.65]);old.install_reverse_fast_update_scope([-.975,-.95]);old.install_reverse_coordination_scope([-1.025,-1.])
 near=copy.deepcopy(old)
 with torch.no_grad():near.reverse_network[-1].bias.add_(.2)
 h=copy.deepcopy(near);h.install_reverse_frontier_preservation([-1.1,-1.075],old.reverse_network.state_dict())
 x=torch.zeros(15,25);x[:,7]=torch.tensor([-1.8,-1.5,-1.15,-1.1,-1.09,-1.08,-1.075,-1.05,-1.025,-1.01,-1.,-.975,-.95,0.,2.])*.5
 y=h.correction(x);slow=x[:,7]>=-1.075*.5;fast=x[:,7]<=-1.1*.5
 assert torch.equal(y[slow],near.correction(x)[slow]);assert torch.equal(y[fast],old.correction(x)[fast])
 assert (y>=torch.minimum(near.correction(x),old.correction(x))-1e-7).all();assert (y<=torch.maximum(near.correction(x),old.correction(x))+1e-7).all()
 leaves=['reverse_network','reverse_update_anchor','reverse_extension_anchor','reverse_fast_update_anchor','reverse_coordination_anchor','reverse_frontier_anchor']
 for leaf in leaves:
  module=getattr(h,leaf);module.requires_grad_(True)
  with torch.no_grad():module[-1].weight.zero_();module[-1].bias.zero_()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 for leaf in leaves:
  bias=getattr(h,leaf)[-1].bias;actual=torch.stack([torch.autograd.grad(h.correction(x[i:i+1])[0,2],bias)[0][2]/.1 for i in range(len(x))]);expected=network_gate(h,leaf,x)
  torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6);torch.testing.assert_close(actual,state_network_gate(state,'rmuc_physics_adaptation.'+leaf,x),atol=2e-6,rtol=2e-6)
 assert (network_gate(h,'reverse_frontier_anchor',x)[slow]==0).all();assert (network_gate(h,'reverse_network',x)[fast]==0).all()
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity();restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True);assert torch.equal(h.correction(x),restored.correction(x));assert not restored.reverse_frontier_anchor[-1].bias.requires_grad
 bad=dict(state);del bad['rmuc_physics_adaptation.reverse_frontier_domain'];fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,bad)


def test_frontier_update_preserves_passed125_and_actual_gradient_routing():
 import copy
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_update_scope([-.6,-.45]);h.install_reverse_extension_scope([-.8,-.65]);h.install_reverse_fast_update_scope([-.975,-.95]);h.install_reverse_coordination_scope([-1.025,-1.]);h.install_reverse_frontier_preservation([-1.1,-1.075])
 x=torch.randn(25,25);x[:,7]=torch.linspace(-1.8,2,25)*.5;x[:,2]=0;x[:,8]=0
 before=h.correction(x).detach();h.install_reverse_frontier_update_scope([-1.3,-1.25]);assert torch.equal(before,h.correction(x))
 with torch.no_grad():h.reverse_frontier_anchor[-1].bias.add_(.2)
 protected=x[:,7]>=-1.25*.5
 assert torch.equal(before[protected],h.correction(x)[protected]);assert not torch.equal(before[~protected],h.correction(x)[~protected])
 x=torch.zeros(9,25);x[:,7]=torch.tensor([-1.8,-1.4,-1.3,-1.275,-1.25,-1.1,-1.075,-.8,0])*.5
 leaves=['reverse_network','reverse_update_anchor','reverse_extension_anchor','reverse_fast_update_anchor','reverse_coordination_anchor','reverse_frontier_anchor','reverse_frontier_update_anchor']
 for leaf in leaves:
  module=getattr(h,leaf);module.requires_grad_(True)
  with torch.no_grad():module[-1].weight.zero_();module[-1].bias.zero_()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 for leaf in leaves:
  bias=getattr(h,leaf)[-1].bias
  actual=torch.stack([torch.autograd.grad(h.correction(x[i:i+1])[0,2],bias)[0][2]/.1 for i in range(len(x))])
  torch.testing.assert_close(actual,network_gate(h,leaf,x),atol=2e-6,rtol=2e-6)
  torch.testing.assert_close(actual,state_network_gate(state,'rmuc_physics_adaptation.'+leaf,x),atol=2e-6,rtol=2e-6)
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(h.correction(x),restored.correction(x));assert not restored.reverse_frontier_update_anchor[-1].bias.requires_grad
 bad=dict(state);del bad['rmuc_physics_adaptation.reverse_frontier_update_domain']
 fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
 with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,bad)


def test_frontier_retention_preserves_entire_old_transition_and_roundtrip():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_update_scope([-.6,-.45]);h.install_reverse_extension_scope([-.8,-.65]);h.install_reverse_fast_update_scope([-.975,-.95]);h.install_reverse_coordination_scope([-1.025,-1.]);h.install_reverse_frontier_preservation([-1.1,-1.075]);h.install_reverse_frontier_update_scope([-1.35,-1.3])
 with torch.no_grad():h.reverse_frontier_anchor[-1].bias.fill_(.3)
 x=torch.randn(51,25)*.1;x[:,7]=torch.linspace(-1.8,.2,51)*.5;x[:,2]=0;x[:,8]=0
 before=h.correction(x).detach();h.install_reverse_frontier_retention([-1.5,-1.45]);assert torch.equal(before,h.correction(x))
 with torch.no_grad():h.reverse_frontier_anchor[-1].bias.add_(.2)
 after=h.correction(x);keep=x[:,7]/.5>=-1.45
 assert torch.equal(before[keep],after[keep]);assert not torch.equal(before[~keep],after[~keep])
 # Test the exact old and new boundaries, including the older blend interval.
 o=torch.zeros(8,25);o[:,7]=torch.tensor([-1.8,-1.5,-1.475,-1.45,-1.4,-1.35,-1.325,-1.3])*.5
 module=h.reverse_frontier_anchor;module.requires_grad_(True)
 with torch.no_grad():module[-1].weight.zero_();module[-1].bias.zero_()
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()}
 bias=module[-1].bias
 actual=torch.stack([torch.autograd.grad(h.correction(o[i:i+1])[0,2],bias)[0][2]/.1 for i in range(len(o))])
 torch.testing.assert_close(actual,network_gate(h,'reverse_frontier_anchor',o),atol=2e-6,rtol=2e-6)
 torch.testing.assert_close(actual,state_network_gate(state,'rmuc_physics_adaptation.reverse_frontier_anchor',o),atol=2e-6,rtol=2e-6)
 assert (actual[3:]==0).all()
 actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True)
 assert torch.equal(h.correction(x),restored.correction(x))
 assert not any(p.requires_grad for p in restored.reverse_frontier_retention_anchor.parameters())
 with pytest.raises(ValueError):h.install_reverse_frontier_retention([-1.5,-1.45])
 for key in ['reverse_frontier_retention_domain','reverse_frontier_retention_anchor.0.weight']:
  bad=dict(state);del bad['rmuc_physics_adaptation.'+key]
  fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
  with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,bad)


def test_wheel_extension_retention_keeps_old_budget_and_conservative_bound():
 from rmuc_command_adapter import install_rmuc_physics_adaptation
 h=PhysicsAdaptation();h.install_reverse_network([.1]*6,[-2,-1.8,-.02,0]);h.install_reverse_reserve([0,0,.1,0,0,.1],[-.85,-.6]);h.install_reverse_reserve_frontier([0,0,.112,0,0,.112],[-.925,-.875]);h.install_reverse_reserve_fast_preservation([0,0,.12,0,0,.12],[-.95,-.925]);h.install_reverse_reserve_extension([0,0,.14,0,0,.14],[-1.05,-1.])
 with torch.no_grad():h.reverse_reserve_network[-1].bias.fill_(-1.)
 o=torch.zeros(9,25);o[:,7]=torch.tensor([-1.8,-1.5,-1.475,-1.45,-1.4,-1.05,-1.,-.9,0])*.5
 before=h.reverse_reserve_output(o).detach()
 h.install_reverse_reserve_extension_retention(h.reverse_reserve_extension_budget,[-1.5,-1.45]);assert torch.equal(before,h.reverse_reserve_output(o))
 with torch.no_grad():h.reverse_reserve_extension_budget[[2,5]]=.145
 after=h.reverse_reserve_output(o);assert torch.equal(before[3:],after[3:]);assert not torch.equal(before[:3],after[:3]);assert torch.equal(after[:,[0,1,3,4]],before[:,[0,1,3,4]])
 state={'rmuc_physics_adaptation.'+k:v for k,v in h.state_dict().items()};actor=torch.nn.Module();actor.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));actor.rmuc_position_authority=torch.nn.Identity()
 restored=install_rmuc_physics_adaptation(actor,state);restored.load_state_dict(h.state_dict(),strict=True);assert torch.equal(h.reverse_reserve_output(o),restored.reverse_reserve_output(o))
 # Lower new budget must not hide the larger preserved budget from bound checks.
 with torch.no_grad():h.reverse_reserve_extension_budget[[2,5]]=.13
 assert (h.correction(o).abs()<=h.correction_bound()+1e-6).all()
 with pytest.raises(ValueError):h.install_reverse_reserve_extension_retention([0,0,.14,0,0,.14],[-1.5,-1.45])
 for key in ['reverse_reserve_extension_retained_budget','reverse_reserve_extension_retention_domain']:
  bad=dict(state);del bad['rmuc_physics_adaptation.'+key]
  fresh=torch.nn.Module();fresh.mlp=torch.nn.Sequential(torch.nn.Linear(25,6));fresh.rmuc_position_authority=torch.nn.Identity()
  with pytest.raises(ValueError):install_rmuc_physics_adaptation(fresh,bad)
