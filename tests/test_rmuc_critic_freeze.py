from types import SimpleNamespace
import copy
import json

import pytest
import torch

from rmuc_critic_freeze import configure_critic_freeze


def test_freeze_preserves_moments_and_actor_updates_then_releases(tmp_path):
    actor, critic = torch.nn.Linear(2, 1), torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=.01)
    x = torch.ones(8, 2)

    def update():
        for _ in range(3):
            optimizer.zero_grad()
            (actor(x).square().mean() + critic(x).square().mean()).backward()
            optimizer.step()
        return {'loss': 1.}

    update()  # Nonzero Adam momentum must not move the frozen critic.
    before = copy.deepcopy(critic.state_dict())
    moments = {p: copy.deepcopy(optimizer.state[p]) for p in critic.parameters()}
    old_actor = copy.deepcopy(actor.state_dict())
    alg = SimpleNamespace(actor=actor, critic=critic, optimizer=optimizer, update=update)
    runner = SimpleNamespace(alg=alg, logger=SimpleNamespace(log_dir=str(tmp_path)))
    configure_critic_freeze(runner, 2)
    alg.update(); alg.update()
    assert all(torch.equal(v, critic.state_dict()[k]) for k, v in before.items())
    assert any(not torch.equal(v, actor.state_dict()[k]) for k, v in old_actor.items())
    for p, saved in moments.items():
        assert all(torch.equal(value, optimizer.state[p][key]) for key, value in saved.items())
    alg.update()
    assert any(not torch.equal(v, critic.state_dict()[k]) for k, v in before.items())
    rows = [json.loads(line) for line in (tmp_path/'critic_freeze.jsonl').read_text().splitlines()]
    assert [r['frozen'] for r in rows] == [True, True, False]
    assert [r['suppressed_optimizer_steps'] for r in rows] == [3, 3, 0]


def test_invalid_freeze_budget_rejected():
    for budget in (0, -1, 1.5):
        with pytest.raises(ValueError):
            configure_critic_freeze(None, budget)
