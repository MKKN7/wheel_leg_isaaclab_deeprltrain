from pathlib import Path
import importlib.util
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/leg_length_terms.py'
s=importlib.util.spec_from_file_location('leg_terms',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_hip_relative_length_excludes_lateral_track_and_offset():
 hip=torch.tensor([[.1775,.000024,-.000014],[-.1775,.000024,-.000014]])
 wheel=hip+torch.tensor([[.04,-.3,0.],[-.07,-.36,0.]])
 got=m.planar_lengths(wheel,hip);torch.testing.assert_close(got,torch.tensor([.3,.36]))
 translated=torch.tensor([1.,2.,3.]);torch.testing.assert_close(m.planar_lengths(wheel+translated,hip+translated),got)

def test_differential_bank_and_length_cost_gradient():
 lengths=torch.tensor([[.32,.36],[.30,.30],[.36,.36]],requires_grad=True);target=torch.full((3,),.34)
 cost=m.mean_length_cost(lengths,target);assert cost[0].item()==0
 cost.sum().backward();assert (lengths.grad[1]<0).all() and (lengths.grad[2]>0).all()

def test_tensor_observation_append_preserves_old_and_shape():
 x=torch.randn(5,25);target=torch.linspace(.30,.36,5);y=m.append_target(x,target)
 assert torch.equal(x,y[:,:25]);torch.testing.assert_close(y[:,25],(target-.34)/.1)
 with pytest.raises(ValueError):m.append_target(x,target[:2])

def test_reward_merges_before_clip_and_legacy_hook_is_identity():
 import ast
 base=p.with_name('env.py');tree=ast.parse(base.read_text())
 hook=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_additional_reward_before_clip')
 # Execute the real identity hook without importing Isaac.
 module=ast.Module(body=[hook],type_ignores=[]);hook.args.args[1].annotation=None;hook.returns=None
 ns={};exec(compile(ast.fix_missing_locations(module),str(base),'exec'),ns)
 reward=torch.tensor([-30.,0.,25.]);assert ns['_additional_reward_before_clip'](None,reward) is reward
 rewards=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='_get_rewards')
 hook_calls=[x.lineno for x in ast.walk(rewards) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='_additional_reward_before_clip']
 clip=next(x.lineno for x in ast.walk(rewards) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='clipped_reward' for t in x.targets))
 assert len(hook_calls)==1 and hook_calls[0]<clip
 # Cost should restore a saturated base into useful gradient, not subtract from
 # an already clipped base and incorrectly change its ranking.
 cost=m.mean_length_cost(torch.tensor([[.30,.30]]),torch.tensor([.34]))
 merged=torch.clamp(torch.tensor([25.])-5*cost,-25,20)
 wrong=torch.clamp(torch.tensor([25.]),-25,20)-5*cost
 assert merged.item()>wrong.item() and merged.item()<20
