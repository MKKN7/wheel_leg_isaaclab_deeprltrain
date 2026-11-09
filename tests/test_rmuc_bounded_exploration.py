from types import SimpleNamespace

import pytest
import torch

from rmuc_bounded_exploration import BoundedWheelStd, configure_bounded_wheel_exploration


class Actor(torch.nn.Module):
    def __init__(self, log=False):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(6))
        self.distribution = torch.nn.Module()
        name = 'log_std_param' if log else 'std_param'
        self.distribution.register_parameter(name, torch.nn.Parameter(
            torch.full((6,), .03).log() if log else torch.full((6,), .03), requires_grad=False))


@pytest.mark.parametrize('log', [False, True])
def test_entropy_changes_only_wheel_std_and_is_bounded_each_step(log):
    actor = Actor(log); optimizer = torch.optim.Adam(actor.parameters(), lr=.1)
    control = BoundedWheelStd(actor, optimizer, (.02, .06), .001)
    assert sum(p is control.parameter for g in optimizer.param_groups for p in g['params']) == 1
    initial_mean = actor.mean.detach().clone()
    for _ in range(100):
        optimizer.zero_grad()
        std = control.parameter.exp() if log else control.parameter
        loss = -torch.distributions.Normal(actor.mean, std).entropy().sum()
        loss.backward(); optimizer.step()
        assert control.readback()['leg_std_fixed']
        assert bool((control.std()[[2, 5]] <= .060001).all())
        assert bool((control.std()[[2, 5]] >= .019999).all())
    assert bool((control.std()[[2, 5]] > .03).all())
    assert torch.equal(actor.mean, initial_mean)
    # Large optimizer step cannot create invalid std for the next probability call.
    with torch.no_grad(): control.parameter[[2, 5]] = -100.
    control.after_step(optimizer, (), {})
    assert bool((control.std() > 0).all())


def test_warmup_freeze_release_keeps_mean_and_probability_model_unchanged():
    actor = Actor(); optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    control = BoundedWheelStd(actor, optimizer, (.02, .06), .0001)
    p = control.parameter; initial = p.detach().clone()
    p.requires_grad_(False); optimizer.zero_grad(); actor.mean.square().sum().backward(); optimizer.step()
    assert torch.equal(p, initial)
    p.requires_grad_(True)
    std = p.expand(8, 6); mean = actor.mean.expand(8, 6)
    distribution = torch.distributions.Normal(mean, std)
    sample = distribution.sample()
    torch.testing.assert_close(distribution.log_prob(sample), torch.distributions.Normal(mean, p).log_prob(sample))


def test_missing_entropy_and_invalid_bounds_fail_before_training(tmp_path):
    actor = Actor(); optimizer = torch.optim.Adam(actor.parameters())
    runner = SimpleNamespace(alg=SimpleNamespace(entropy_coef=0., optimizer=optimizer, get_policy=lambda: actor),
                             logger=SimpleNamespace(log_dir=str(tmp_path)))
    with pytest.raises(ValueError, match='positive entropy'):
        configure_bounded_wheel_exploration(runner, (.02, .06), .0001)
    with pytest.raises(ValueError): BoundedWheelStd(actor, optimizer, (.02, .1), .0001)
