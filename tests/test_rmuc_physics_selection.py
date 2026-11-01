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
 with pytest.raises(ValueError):validate_trainable_sequence_gates(state,[[0,0,0],[2,-1.6,0]],physics_only=True,physics_modules=['rmuc_physics_adaptation.reverse_network'])


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
