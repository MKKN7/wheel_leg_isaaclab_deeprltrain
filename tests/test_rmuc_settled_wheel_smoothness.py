import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/settled_wheel_smoothness.py';s=importlib.util.spec_from_file_location('settled_cost',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_only_settled_idle_and_only_wheels():
 a=torch.zeros(5,6);b=a.clone();a[:,2]=.02;a[:,5]=-.02;c=torch.zeros(5,3);c[1,0]=.1;v=torch.tensor([0,0,.2,0,0]);rate=torch.tensor([0,0,0,.2,0]);tilt=torch.tensor([0,0,0,0,.1])
 cost=m.settled_wheel_cost(a,b,c,v,rate,tilt);assert abs(cost[0].item()-(2**.5-1))<1e-6;assert torch.equal(cost[1:],torch.zeros(4))
 a.zero_();a[:,0]=1;assert not m.settled_wheel_cost(a,b,c,v,rate,tilt).any()

def test_finite_bounded_gradient():
 a=torch.ones(3,6,requires_grad=True);z=torch.zeros(3);cost=m.settled_wheel_cost(a,a.detach()*0,torch.zeros(3,3),z,z,z);assert (cost<=4).all();cost.sum().backward();assert torch.isfinite(a.grad).all()
 with pytest.raises(ValueError):m.settled_wheel_cost(a,a,torch.zeros(3,3),z,z,z,action_scale=0)
