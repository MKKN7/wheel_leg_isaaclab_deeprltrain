import torch
from rmuc_world_model.value import discounted_cost_to_go,StopValue

def test_discounted_tail_excludes_already_incurred_cost():
    cost=torch.tensor([1.,2.,4.])
    inclusive=discounted_cost_to_go(cost,.5)
    torch.testing.assert_close(inclusive,torch.tensor([3.,4.,4.]))
    subsequent=torch.cat([inclusive[1:],torch.zeros_like(inclusive[:1])])
    torch.testing.assert_close(subsequent,torch.tensor([4.,4.,0.]))

def test_frozen_value_propagates_state_gradient():
    torch.manual_seed(27)
    model=StopValue(torch.zeros(3),torch.ones(3),torch.tensor(.1)).requires_grad_(False)
    x=torch.tensor([[.2,-.3,.7]],requires_grad=True)
    y=model(x);y.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())
    eps=1e-3
    delta=torch.tensor([[eps,0.,0.]])
    numeric=(model(x.detach()+delta)-model(x.detach()-delta))/(2*eps)
    torch.testing.assert_close(x.grad[:,0],numeric,rtol=.02,atol=1e-5)


def test_rebound_shaping_only_acts_beyond_margin():
    from rmuc_world_model.value import stopping_stage_cost
    v=torch.tensor([.1,-.08,-.1],requires_grad=True);zero=torch.zeros_like(v)
    base=stopping_stage_cost(v,zero,zero);shaped=stopping_stage_cost(v,zero,zero,50.)
    torch.testing.assert_close(shaped[:2],base[:2])
    assert shaped[2]>base[2]
    (shaped-base).sum().backward()
    assert v.grad[2]<0 and torch.equal(v.grad[:2],torch.zeros(2))

def test_settling_cost_rewards_reducing_small_and_large_speed_errors():
    from rmuc_world_model.value import stopping_stage_cost
    v=torch.tensor([0.,.005,.01,.02,.1,.5],requires_grad=True);zero=torch.zeros_like(v)
    c=stopping_stage_cost(v,zero,zero,profile='settling_v1',yaw=zero,hip=zero)
    assert (c[1:]>c[:-1]).all()
    c.sum().backward();assert (v.grad[1:]>0).all()
    base=stopping_stage_cost(zero,zero,zero,profile='settling_v1',yaw=zero,hip=zero)
    yaw=stopping_stage_cost(zero,zero,zero,profile='settling_v1',yaw=zero+.1,hip=zero)
    hip=stopping_stage_cost(zero,zero,zero,profile='settling_v1',yaw=zero,hip=zero+.06)
    assert (yaw>base).all() and (hip>base).all()
