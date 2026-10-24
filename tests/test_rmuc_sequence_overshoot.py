import torch
from rmuc_command_sequence import sequence_overshoot_cost, validate_sequence


def test_startup_braking_and_reverse_target_crossing_are_symmetric():
    table = validate_sequence([[0, 0, 0], [2, .5, 0], [4, 0, 0], [6, -.5, 0]], .008, 8)
    age = torch.tensor([250, 250, 250, 500, 500, 750, 750])
    command = torch.zeros(7, 3)
    command[:, 0] = torch.tensor([.5, .5, .5, 0, 0, -.5, -.5])
    velocity = torch.tensor([.4, .55, .6, .3, -.1, -.6, -.4], requires_grad=True)
    result = sequence_overshoot_cost(table, age, command, velocity, torch.ones(7, dtype=torch.bool))
    torch.testing.assert_close(result, torch.tensor([0., 0., .1, 0., .1, .1, 0.]), atol=1.e-6, rtol=1.e-6)
    result.sum().backward()
    assert velocity.grad[2] > 0 and velocity.grad[4] < 0 and velocity.grad[5] < 0


def test_static_initial_phase_command_mismatch_and_yaw_are_excluded():
    table = validate_sequence([[0, 0, 0], [2, .5, 0], [4, 0, .3]], .008, 8)
    age = torch.tensor([0, 250, 250, 500])
    command = torch.tensor([[0., 0, 0], [.5, 0, 0], [.4, 0, 0], [0, 0, .3]])
    result = sequence_overshoot_cost(table, age, command, torch.ones(4)*10, torch.tensor([True, False, True, True]))
    assert not result.any()


def test_cost_is_finite_bounded_and_does_not_punish_approach_before_target():
    table = validate_sequence([[0, .5, 0], [2, 0, 0]], .008, 4)
    cmd = torch.zeros(5, 3)
    result = sequence_overshoot_cost(table, torch.full((5,), 250), cmd,
                                   torch.tensor([1., .5, 0., -.5, -100.]), torch.ones(5, dtype=torch.bool))
    assert torch.equal(result[:3], torch.zeros(3))
    assert result[3] > 0 and result[4] == 4 and torch.isfinite(result).all()
