import copy

import pytest
import torch

from rmuc_reward_history_critic import CausalRewardHistory, RewardHistoryCritic, make_mlp


def test_features_exclude_current_and_future_rewards():
    commands = torch.zeros(2, 3)
    a, b = CausalRewardHistory(2), CausalRewardHistory(2)
    for step in range(5):
        torch.testing.assert_close(a.before_step(commands, step), b.before_step(commands, step))
        reward = torch.tensor([1., 3.])
        a.after_step(reward, torch.zeros(2, dtype=torch.bool), step)
        b.after_step(reward if step < 4 else reward + 100, torch.zeros(2, dtype=torch.bool), step)
    assert not torch.equal(a.before_step(commands, 5), b.before_step(commands, 5))


def test_reset_command_switch_and_duplicate_update():
    memory = CausalRewardHistory(2)
    command = torch.zeros(2, 3)
    before = memory.before_step(command, 0)
    assert torch.equal(before, memory.before_step(command, 0))
    memory.after_step(torch.ones(2), torch.tensor([True, False]), 0)
    with pytest.raises(ValueError):
        memory.after_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), 0)
    command[1, 0] = .5
    assert torch.count_nonzero(memory.before_step(command, 1)) == 0
    with pytest.raises(ValueError):
        memory.before_step(command, 2)


def test_history_checkpoint_resume_and_snapshot_ownership():
    memory = CausalRewardHistory(2)
    command = torch.zeros(2, 3)
    memory.before_step(command, 0)
    memory.after_step(torch.ones(2), torch.zeros(2, dtype=torch.bool), 0)
    snapshot = memory.state_dict()
    restored = CausalRewardHistory(2)
    restored.load_state_dict(snapshot)
    features = memory.before_step(command, 1)
    torch.testing.assert_close(features, restored.before_step(command, 1))
    features.zero_()
    assert memory.ema.sum() > 0
    memory.after_step(torch.full((2,), 2.), torch.zeros(2, dtype=torch.bool), 1)
    assert torch.equal(snapshot['ema'], restored.ema)
    with pytest.raises(ValueError):
        CausalRewardHistory(2, dt=.002).load_state_dict(snapshot)


def test_startup_is_exact_original_and_forward_has_no_state_mutation():
    source = make_mlp(25)
    model = RewardHistoryCritic(source, torch.zeros(29), torch.ones(29), 100.)
    x = torch.randn(32, 29)
    x[:, 28] = 0
    old_x, old_state = x.clone(), copy.deepcopy(model.state_dict())
    torch.testing.assert_close(model(x), source(x[:, :25]), rtol=0, atol=0)
    assert torch.equal(x, old_x)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old_state.items())
    x[:, 28] = 4
    torch.testing.assert_close(model(x), x[:, 26:27] / .002)


def test_only_residual_receives_parameter_gradients_and_roundtrip_is_exact():
    source = make_mlp(25)
    original = copy.deepcopy(source.state_dict())
    model = RewardHistoryCritic(source, torch.zeros(29), torch.ones(29), 100.)
    x = torch.randn(32, 29)
    x[:, 28] = 4
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad()
    model(x).square().mean().backward()
    assert all(p.grad is None for p in model.frozen_base.parameters())
    assert model.residual[-1].weight.grad.abs().sum() > 0
    opt.step()
    assert all(torch.equal(v, source.state_dict()[k]) for k, v in original.items())
    other = RewardHistoryCritic(source, torch.ones(29), torch.ones(29), 1.)
    checkpoint = model.checkpoint()
    other.load_checkpoint(checkpoint)
    torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)
    with pytest.raises(ValueError):
        RewardHistoryCritic(source, torch.zeros(29), torch.ones(29), 1., gamma=.99).load_checkpoint(checkpoint)
    checkpoint['context_version'] = 'queues_v1'
    with pytest.raises(ValueError):
        other.load_checkpoint(checkpoint)
