"""Check the source firmware gas-spring compensation law without Isaac Sim."""
import math
import runpy
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / "source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/gas_spring.py"
compensation = runpy.run_path(str(SOURCE))["firmware_spring_motor_compensation"]


def test_firmware_spring_length_clamps_and_angle_pair():
    length = torch.tensor([0.05, 0.15, 0.35])
    front = torch.full_like(length, math.pi / 2)
    back = torch.zeros_like(length)
    pair = compensation(length, front, back)
    expected = torch.tensor([27.0, 22.21908, 16.5])
    torch.testing.assert_close(pair[:, 0], -expected)
    torch.testing.assert_close(pair[:, 1], expected)
    torch.testing.assert_close(pair.sum(-1), torch.zeros_like(length))
    torch.testing.assert_close(compensation(length, front, front), torch.zeros_like(pair))


def test_firmware_spring_rejects_nonfinite_and_shape_mismatch():
    with pytest.raises(ValueError):
        compensation(torch.tensor([0.2]), torch.tensor([math.nan]), torch.tensor([0.0]))
    with pytest.raises(ValueError):
        compensation(torch.tensor([0.2]), torch.tensor([0.0, 0.0]), torch.tensor([0.0]))


def test_usd_joint_mapping_does_not_apply_encoder_offsets():
    positions = torch.zeros(1, 4)
    torques, lengths = runpy.run_path(str(SOURCE))["firmware_spring_torque_from_joint_positions"](positions)
    assert torch.isfinite(torques).all() and torch.isfinite(lengths).all()
    assert torch.allclose(torques, torch.zeros_like(torques))
