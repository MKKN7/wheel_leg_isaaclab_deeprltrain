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
