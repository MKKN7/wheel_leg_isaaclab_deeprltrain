import torch
import pytest
from torch import nn
from rsl_rl.modules import GaussianDistribution
from rmuc_stop_exploration import install_stop_exploration

class Actor(nn.Module):
    def __init__(self):
        super().__init__();self.rmuc_physics_adaptation=nn.Linear(25,6)
        self.rmuc_physics_adaptation.stop_network=nn.Identity()
        self.distribution=GaussianDistribution(6,init_std=.03);self.distribution.requires_grad_(False)
    def forward(self,x,stochastic=True):
        y=self.rmuc_physics_adaptation(x)
        if stochastic:self.distribution.update(y)
        return self.distribution.sample() if stochastic else y

def inputs():
    x=torch.zeros(7,25);x[:,6]=torch.tensor([0,-.042,-.1155,-.21,.21,-.21,-.21]);x[5,7]=.25;x[6,8]=.5
    return x

def test_probability_replay_and_no_state_change():
    a=Actor();x=inputs();state={k:v.clone() for k,v in a.state_dict().items()};mean=a(x,False).detach();rng=torch.random.get_rng_state().clone();audit=install_stop_exploration(a)
    assert torch.equal(rng,torch.random.get_rng_state());y=a(x);old=tuple(v.detach().clone() for v in a.distribution.params);lp=a.distribution.log_prob(y).detach()
    assert torch.equal(mean,a(x,False))
    a(x);torch.testing.assert_close(a.distribution.log_prob(y),lp);torch.testing.assert_close(a.distribution.kl_divergence(old,a.distribution.params),torch.zeros(7))
    perm=torch.tensor([6,2,4,0,3,1,5]);a(x[perm]);torch.testing.assert_close(a.distribution.log_prob(y[perm]),lp[perm])
    normal=torch.distributions.Normal(*a.distribution.params);torch.testing.assert_close(a.distribution.entropy,normal.entropy().sum(-1));torch.testing.assert_close(a.distribution.log_prob(y[perm]),normal.log_prob(y[perm]).sum(-1))
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in state.items())
    a.distribution.log_prob(y[perm]).mean().backward();assert all(torch.isfinite(p.grad).all() for p in a.rmuc_physics_adaptation.parameters())
    assert audit.readback()['calls']==3

def test_only_moving_stop_wheels_increased():
    a=Actor();install_stop_exploration(a);a(inputs());std=a.distribution.std
    torch.testing.assert_close(std[:,[0,1,3,4]],torch.full((7,4),.03))
    torch.testing.assert_close(std[[0,1,5,6]][:,[2,5]],torch.full((4,2),.03))
    torch.testing.assert_close(std[[3,4]][:,[2,5]],torch.full((2,2),.05))
    torch.testing.assert_close(std[2,[2,5]],torch.full((2,),.04))
    with pytest.raises(RuntimeError):a.distribution.update(torch.zeros(7,6))

def test_invalid_or_trainable_std_rejected():
    a=Actor();a.distribution.requires_grad_(True)
    with pytest.raises(ValueError):install_stop_exploration(a)
    a.distribution.requires_grad_(False)
    with pytest.raises(ValueError):install_stop_exploration(a,.2)

def test_rollout_inference_then_ppo_grad_mode_audit():
    a=Actor();audit=install_stop_exploration(a);x=inputs()
    with torch.inference_mode():
        actions=a(x);old_log=a.distribution.log_prob(actions)
    actions=actions.clone();old_log=old_log.clone()
    a(x)
    ratio=(a.distribution.log_prob(actions)-old_log).exp()
    torch.testing.assert_close(ratio,torch.ones(7))
    (-ratio.mean()).backward()
    with torch.inference_mode():a(x)
    a(x);assert audit.readback()['calls']==4
    assert audit.readback()['samples'][1]==28
