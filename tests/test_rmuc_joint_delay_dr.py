import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/joint_delay_dr.py'
s=importlib.util.spec_from_file_location('joint_delay_dr',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_impulse_all_delays_and_cascade():
 q=m.SampledDelay(torch.zeros(6,1),.002,[0,.010],1);q.steps[:]=torch.arange(6)
 output=[]
 for i in range(7):output.append(q.step(torch.full((6,1),float(i==0)))[:,0])
 assert torch.equal(torch.stack(output)[:6],torch.eye(6))
 a=m.SampledDelay(torch.zeros(1,1),.002,[.010,.010],2);b=m.SampledDelay(torch.zeros(1,1),.002,[.010,.010],3)
 values=[float(b.step(a.step(torch.tensor([[float(i==0)]])))) for i in range(12)]
 assert values[10]==1 and sum(values)==1

def test_private_rng_independence_and_full_ranges():
 template=torch.zeros(2048,6);ranges=dict(observation=[0,.008],action=[0,.010],wheel=[.002,.010],leg=[.002,.010]);torch.manual_seed(9);before=torch.random.get_rng_state().clone();q=m.JointDelayDR(template,.002,ranges,777)
 assert torch.equal(before,torch.random.get_rng_state())
 assert q.observation.steps.unique().tolist()==list(range(5))
 assert q.action.steps.unique().tolist()==list(range(6))
 assert q.wheel.steps.unique().tolist()==list(range(1,6))
 assert q.leg.steps.unique().tolist()==list(range(1,6))
 assert ((q.observation.steps>0)&(q.action.steps>0)).any()
 assert (q.wheel.steps!=q.leg.steps).any()
 clone=m.JointDelayDR(template,.002,ranges,777)
 assert torch.equal(q.action.steps,clone.action.steps)

def test_sensor_timestamp_no_double_push_and_partial_reset():
 q=m.SampledDelay(torch.zeros(2,1),.002,[.002,.002],1,True)
 q.push(torch.tensor([[1.],[2.]]),0);q.push(torch.tensor([[5.],[6.]]),0)
 assert torch.equal(q.read(),torch.tensor([[1.],[2.]]))
 q.push(torch.tensor([[3.],[4.]]),1);q.push(torch.tensor([[5.],[6.]]),2)
 assert torch.equal(q.read(),torch.tensor([[3.],[4.]]))
 q.reset(torch.tensor([0]));q.push(torch.tensor([[9.],[99.]]),2)
 assert torch.equal(q.read(),torch.tensor([[9.],[4.]]))

def test_invalid_quantization_and_ranges():
 for bounds in [[.001,.01],[.01,.002],[-.002,.01],[0,.014]]:
  with pytest.raises(ValueError):m.SampledDelay(torch.zeros(1,1),.002,bounds,1)

def test_physics_clock_observation_boundary_and_action_phase():
 # Isaac increments sim counter BEFORE apply_action; pre-step state is counter-1.
 obs=m.SampledDelay(torch.zeros(1,1),.002,[.008,.008],1,True)
 obs.push(torch.tensor([[0.]]),0)
 for counter in range(1,9):
  obs.push(torch.tensor([[float(counter-1)]]),counter-1)
  if counter%4==0:
   obs.push(torch.tensor([[float(counter)]]),counter)
   assert float(obs.read())==counter-4
 action=m.SampledDelay(torch.zeros(1,1),.002,[.010,.010],2)
 actual=[]
 for tick in range(12):actual.append(float(action.step(torch.tensor([[float(tick//4+1)]]))))
 assert actual==[0,0,0,0,0,1,1,1,1,2,2,2]


def test_joint_budget_sampling_is_reproducible_and_partial_reset_safe():
 ranges=dict(observation=[0,.008],action=[0,.008],wheel=[.002,.008],leg=[.002,.008])
 table=[[.002,.002,.004,.006,9.],[.004,.004,.006,.008,1.]]
 before=torch.random.get_rng_state().clone()
 q=m.JointDelayDR(torch.zeros(8192,6),.002,ranges,777,table)
 assert torch.equal(before,torch.random.get_rng_state())
 clone=m.JointDelayDR(torch.zeros(8192,6),.002,ranges,777,table)
 assert torch.equal(q.combination_indices,clone.combination_indices)
 assert .08<float((q.combination_indices==1).float().mean())<.12
 assert set((q.observation.steps+q.action.steps+q.wheel.steps).tolist())=={4,7}
 assert set((q.observation.steps+q.action.steps+q.leg.steps).tolist())=={5,8}
 old=q.wheel.steps.clone();q.reset(torch.arange(30));assert torch.equal(q.wheel.steps[30:],old[30:])
 q.action.set_sampling_weights([1,1,1,1,1])
 with pytest.raises(ValueError,match='independent action weights'):q.reset(torch.arange(30))


def test_joint_budget_rejects_nonquantized_delay():
 ranges={k:[0,.008] for k in ['observation','action','wheel','leg']}
 with pytest.raises(ValueError,match='physics ticks'):
  m.JointDelayDR(torch.zeros(2,6),.002,ranges,3,[[.001,0,.004,.004,1.]])
