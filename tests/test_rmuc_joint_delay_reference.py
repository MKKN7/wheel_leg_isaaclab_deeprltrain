import copy
import pytest
import torch
from rmuc_physics_adaptation import PhysicsAdaptation


def test_reference_preserves_forward_and_reverse_endpoints_and_budget():
    source = PhysicsAdaptation()
    source.install_joint_delay_network([.08, .05, .1, .08, .05, .1])
    with torch.no_grad():
        source.joint_delay_network[-1].bias.fill_(.3)
    candidate = copy.deepcopy(source)
    candidate.install_joint_delay_scope([-1.5, -1.4, -.1, 0., .01, .05])
    candidate.install_joint_delay_reference()
    assert not any(p.requires_grad for p in candidate.joint_delay_reference_network.parameters())
    with torch.no_grad():
        candidate.joint_delay_network[-1].bias.fill_(-.4)
    raw_candidate = copy.deepcopy(source)
    raw_candidate.joint_delay_network.load_state_dict(candidate.joint_delay_network.state_dict())
    obs = torch.zeros(8, 25)
    obs[:, 7] = torch.tensor([0., .5, 2., -.25, -.8, -1., -.05, -.4]) * .5
    obs[-1, 8] = .2
    before, after = source.correction(obs), candidate.correction(obs)
    torch.testing.assert_close(after[[0, 1, 2, 7]], before[[0, 1, 2, 7]], atol=0, rtol=0)
    torch.testing.assert_close(after[3:6], raw_candidate.correction(obs)[3:6], atol=0, rtol=0)
    assert (after.abs() <= candidate.correction_bound() + 1e-7).all()
    candidate.zero_grad()
    after[[0, 1, 2, 7]].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in candidate.joint_delay_network.parameters())
    traced = torch.jit.trace(candidate, (obs, torch.zeros(8, 6)))
    torch.testing.assert_close(traced(obs, torch.zeros(8, 6)), candidate(obs, torch.zeros(8, 6)))


def test_reference_requires_explicit_scope_and_cannot_be_replaced():
    head = PhysicsAdaptation()
    with pytest.raises(ValueError):
        head.install_joint_delay_reference()

    head.install_joint_delay_network([.1] * 6)
    with pytest.raises(ValueError):
        head.install_joint_delay_reference()
    head.install_joint_delay_scope([-1.5, -1.4, -.1, 0., .01, .05])
    head.install_joint_delay_reference()
    with pytest.raises(ValueError):
        head.install_joint_delay_reference()


def test_frozen_zero_sequence_requires_reference_and_trainable_reverse_phase():
    from rmuc_training_contract import validate_trainable_sequence_gates
    head = PhysicsAdaptation()
    head.install_joint_delay_network([.1] * 6)
    head.install_joint_delay_scope([-1.5, -1.4, -.1, 0., .01, .05])
    head.install_joint_delay_reference()
    state = {'rmuc_physics_adaptation.' + k: v for k, v in head.state_dict().items()}
    selection = dict(physics_only=True, physics_modules=['rmuc_physics_adaptation.joint_delay_network'])
    rows = validate_trainable_sequence_gates(state, [[0, 0, 0], [2, -.25, 0], [4, 0, 0]], **selection)
    assert rows[0]['gates'] == [0.] and rows[1]['gates'] == [1.]
    for sequence in [[[0, 0, 0], [2, 0, 0]], [[0, 0, 0], [2, .25, 0]], [[0, 0, 0], [2, -.25, .1]]]:
        with pytest.raises(ValueError):
            validate_trainable_sequence_gates(state, sequence, **selection)


def test_reference_training_selection_preserves_reverse_core():
    from rmuc_physics_selection import network_gate, validate_paths
    head = PhysicsAdaptation()
    head.install_joint_delay_network([.1] * 6)
    head.install_joint_delay_scope([-1.5, -1.4, -.1, 0., .01, .05])
    head.install_joint_delay_reference()
    state = {'rmuc_physics_adaptation.' + k: v for k, v in head.state_dict().items()}
    validate_paths(['rmuc_physics_adaptation.joint_delay_reference_network'], state)
    obs = torch.zeros(4, 25)
    obs[:, 7] = torch.tensor([-.8, -.25, .5, 1.5]) * .5
    gate = network_gate(head, 'joint_delay_reference_network', obs)
    torch.testing.assert_close(gate, torch.tensor([0., 0., 1., 1.]))
    before = head.correction(obs).detach().clone()
    with torch.no_grad():
        head.joint_delay_reference_network[-1].bias.add_(.7)
    after = head.correction(obs)
    torch.testing.assert_close(before[:2], after[:2], atol=0, rtol=0)
    assert (after[2:] - before[2:]).abs().max() > 0


def test_extended_reverse_scope_keeps_old_commands_and_reference_fade():
    """Extending learned reverse response must not switch old/turn commands."""
    old = PhysicsAdaptation()
    old.install_joint_delay_network([.08, .05, .1, .08, .05, .1])
    old.install_joint_delay_scope([-1.5, -1.4, -.1, 0., .01, .05])
    old.install_joint_delay_reference()
    with torch.no_grad():
        old.joint_delay_network[-1].bias.fill_(-.5)
        old.joint_delay_reference_network[-1].bias.fill_(.2)
    new = PhysicsAdaptation()
    new.install_joint_delay_network([.08, .05, .1, .08, .05, .1])
    new.install_joint_delay_scope([-2., -1.8, -.1, 0., .01, .05])
    new.install_joint_delay_reference()
    state = copy.deepcopy(old.state_dict())
    state['joint_delay_domain'] = new.joint_delay_domain.clone()
    new.load_state_dict(state, strict=True)
    obs = torch.zeros(10, 25)
    obs[:, 7] = torch.tensor([-2., -1.9, -1.8, -1.5, -1.4, -.4, 0., 1., -1.8, -1.8]) * .5
    obs[8, 8] = .1
    obs[9, 8] = 2.  # clipped yaw-error must remain on the reference
    torch.testing.assert_close(new.joint_delay_gate(obs), torch.tensor([0., .5, 1., 1., 1., 1., 0., 0., 0., 0.]), atol=1e-6, rtol=0)
    before, after = old.correction(obs), new.correction(obs)
    assert torch.equal(before[4:], after[4:])
    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[2:4], after[2:4])
    assert (after.abs() <= new.correction_bound() + 1e-7).all()
    with pytest.raises(ValueError):
        invalid = PhysicsAdaptation()
        invalid.install_joint_delay_network([.1] * 6)
        invalid.install_joint_delay_scope([-2.01, -1.8, -.1, 0., .01, .05])
