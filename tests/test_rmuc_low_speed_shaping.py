import importlib.util
from pathlib import Path
import torch

p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/low_speed_shaping.py'
s=importlib.util.spec_from_file_location('shaping',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_wrong_way_and_undershoot_have_correct_dense_direction():
    command=torch.tensor([[1.,0.,.3]]*4)
    yaw=torch.tensor([-.3,0.,.3,.6],requires_grad=True)
    cost=m.low_speed_yaw_cost(command,yaw)
    assert cost[0]>cost[1]>cost[2] and cost[2]==0
    cost.sum().backward()
    assert yaw.grad[0]<0 and yaw.grad[1]<0 and yaw.grad[2]==0 and yaw.grad[3]>0
    torch.testing.assert_close(m.low_speed_yaw_cost(-command,-yaw.detach()),cost.detach())

def test_protected_domain_cost_is_zero():
    command=torch.tensor([[1.,0.,.8],[1.,0.,1.4],[2.,0.,.3],[1.,0.,0.]])
    assert torch.equal(m.low_speed_yaw_cost(command,torch.ones(4)),torch.zeros(4))

def test_explicit_expansion_covers_new_target_without_changing_default():
    command=torch.tensor([[1.25,0.,.6],[1.25,0.,-.6],[1.,0.,1.4],[1.6,0.,.6]])
    yaw=torch.zeros(4,requires_grad=True)
    assert torch.equal(m.low_speed_yaw_cost(command,yaw),torch.zeros(4))
    cost=m.low_speed_yaw_cost(command,yaw,max_yaw_rad_s=.6001)
    cost.sum().backward()
    assert cost[0]>0 and cost[1]>0
    assert yaw.grad[0]<0 and yaw.grad[1]>0
    assert torch.equal(cost[2:],torch.zeros(2))

def test_invalid_dense_range_rejected():
    import pytest
    for bound in (0.,-1.,float('nan'),float('inf'),2.6):
        with pytest.raises(ValueError):
            m.low_speed_yaw_cost(torch.zeros(1,3),torch.zeros(1),bound)
