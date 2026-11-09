import copy

import pytest
import torch

from rmuc_command_value import CommandValueNetwork, mlp, protect_inactive_expert_momentum


def fixture():
    torch.manual_seed(11)
    base=mlp(); experts=torch.nn.ModuleList([mlp(),mlp()])
    spec=dict(arm='isolated_command_scale',grid=[[0.,0.],[.5,.3]],
        mean=torch.randn(25),scale=torch.rand(25)+.1,
        command_mean=torch.tensor([[50.],[100.]]),command_scale=torch.tensor([[3.],[7.]]),
        model_state_dict=experts.state_dict())
    return base,experts,spec,CommandValueNetwork(base.state_dict(),spec)


def observations():
    x=torch.randn(32,25);x[:,7]=0;x[:,8]=-x[:,2]/.25
    x[16:,7]=.25;x[16:,8]=.3-x[16:,2]/.25
    return x


def test_exact_centers_match_supervised_predictor_and_original_outside():
    base,experts,spec,net=fixture();x=observations();old=x.clone()
    for i,ii in enumerate([slice(0,16),slice(16,32)]):
        expected=experts[i]((x[ii]-spec['mean'])/spec['scale'])*spec['command_scale'][i]+spec['command_mean'][i]
        torch.testing.assert_close(net(x[ii]),expected)
    x[:,7]=1
    torch.testing.assert_close(net(x),base(x),rtol=0,atol=0)
    assert torch.equal(old[:,:7],x[:,:7])
    rng=torch.get_rng_state().clone();net(x);assert torch.equal(rng,torch.get_rng_state())


def test_loss_gradient_scaling_matches_explicit_normalized_mse():
    _,_,_,a=fixture();b=copy.deepcopy(a);x=observations();target=torch.randn(32,1)
    a.normalize_value_gradients=True
    (a(x)-target).square().mean().backward()
    scales=torch.cat([b.value_scale[0].expand(16,1),b.value_scale[1].expand(16,1)])
    ((b(x)-target)/scales).square().mean().backward()
    for p,q in zip(a.experts.parameters(),b.experts.parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=2e-5,atol=2e-5)
    assert all(p.grad is None for p in a.base.parameters())


def test_other_expert_and_frozen_fallback_do_not_move_with_adam_momentum():
    _,_,_,net=fixture();x=observations();optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    protect_inactive_expert_momentum(net,optimizer)
    optimizer.zero_grad();net(x).square().mean().backward();optimizer.step()
    old=copy.deepcopy(net.state_dict())
    optimizer.zero_grad(set_to_none=False)
    net(x[:16]).square().mean().backward();optimizer.step()
    for k,v in old.items():
        if k.startswith(('base.','experts.1.')):assert torch.equal(v,net.state_dict()[k])
    assert any(not torch.equal(v,net.state_dict()[k]) for k,v in old.items() if k.startswith('experts.0.'))


def test_checkpoint_and_compact_continuous_fallback():
    base,_,_,net=fixture();x=observations()[:1]
    clone=copy.deepcopy(net);clone.load_checkpoint(net.checkpoint())
    torch.testing.assert_close(net(x),clone(x),rtol=0,atol=0)
    restored=CommandValueNetwork.from_checkpoint(net.checkpoint())
    torch.testing.assert_close(net(x),restored(x),rtol=0,atol=0)
    bad=net.checkpoint();bad['contract']['actor_dim']=29
    with pytest.raises(ValueError):clone.load_checkpoint(bad)
    x[:,7]=.02*.5
    torch.testing.assert_close(net(x),base(x),rtol=0,atol=0)
    x[:,7]=(.02-1e-5)*.5
    torch.testing.assert_close(net(x),base(x),atol=.001,rtol=0)
    with pytest.raises(ValueError):
        CommandValueNetwork(base.state_dict(),fixture()[2],half_width=(1.,1.))
