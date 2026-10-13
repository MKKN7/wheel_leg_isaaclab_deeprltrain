"""Behavioral isolation and shared total-budget bounds for local correction."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/rsl_rl'))
from rmuc_local_authority import LocalAuthorityCorrection

def test_append_preserves_old_experts_and_zero_outputs_then_isolates_new_centers():
    model = nonzero_module()
    x = observations([.8, -.8, 1., -1., 1.2, -1.2, 1.4, -1.4])
    budget = torch.tensor([.68, 0., .06, .68, 0., .06])
    source = torch.zeros(8, 6)
    before = model(x, source, budget).detach()
    old = {k: v.clone() for k, v in model.state_dict().items()}
    assert model.append_centers([1.4, -1.4]) == [3, 4]
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.items() if k != 'centers')
    assert torch.equal(model.centers[:3], old['centers'])
    assert torch.equal(model(x, source, budget), before)
    with torch.no_grad():
        for branch in model.experts[3:]:
            for expert in branch.experts: expert[-1].bias.fill_(.5)
    after = model(x, source, budget)
    assert torch.equal(after[:6], before[:6])
    assert torch.all((after[6:] - before[6:]).abs().sum(-1) > 0)
    assert torch.all(after.abs() <= budget)


def test_invalid_append_is_atomic_and_extended_state_restores_strictly():
    import pytest
    model = LocalAuthorityCorrection()
    for values in ([], [1.2], [1.4, 1.4], [float('nan')]):
        with pytest.raises(ValueError): model.append_centers(values)
        assert len(model.experts) == 3 and model.centers.numel() == 3
    model.append_centers([1.4, -1.4])
    restored = LocalAuthorityCorrection(model.centers)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.state_dict().keys() == model.state_dict().keys()


def observations(commands, speed=1.):
    x = torch.randn(len(commands), 25)
    x[:, 7] = speed * .5
    # Reconstruct commands despite nonzero measured yaw.
    x[:, 8] = torch.tensor(commands) - x[:, 2] / .25
    return x


def nonzero_module():
    model = LocalAuthorityCorrection()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(.2)
    return model


def test_zero_initialization_preserves_arbitrary_source():
    model = LocalAuthorityCorrection()
    x = observations([-.8, -1., 1.2, -.9, .8, 1., -1.2])
    budget = torch.tensor([.68, 0., .06, .68, 0., .06])
    source = torch.randn(7, 6).tanh() * budget
    assert torch.equal(model(x, source, budget), source)


def test_nonzero_patch_cannot_change_protected_commands_or_other_speeds():
    model = nonzero_module()
    for x in [observations([.8, 1., -1.2]), observations([-.8, -1., 1.2], speed=1.5)]:
        source = torch.randn(3, 6) * .01
        assert torch.equal(model(x, source, torch.ones(6)), source)
        assert not model.gates(x).any()


def test_failed_commands_change_without_expanding_total_budget():
    model = nonzero_module()
    x = observations([-.8, -1., 1.2, -.9])
    budget = torch.tensor([.68, 0., .06, .68, 0., .06])
    source = budget.expand(4, -1) * .99
    result = model(x, source, budget)
    assert torch.any(result != source)
    assert torch.all(result.abs() <= budget)
    assert torch.equal(result[:, [1, 4]], torch.zeros(4, 2))


def test_gradients_only_reach_the_selected_expert():
    model = LocalAuthorityCorrection()
    x = observations([-.8])
    model(x, torch.zeros(1, 6), torch.ones(6)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts[0].parameters())
    assert all(p.grad is None or not p.grad.any() for expert in model.experts[1:] for p in expert.parameters())
