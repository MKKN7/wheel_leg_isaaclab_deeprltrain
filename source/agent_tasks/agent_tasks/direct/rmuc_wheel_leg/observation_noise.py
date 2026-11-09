"""Sensor randomization that preserves the commanded expert-routing inputs."""
from __future__ import annotations

import torch


def observation_randn_like(value, generator=None):
    if generator is None:
        return torch.randn_like(value)
    return torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator)


def add_command_consistent_noise(
    observation: torch.Tensor,
    std: torch.Tensor,
    target_yaw: torch.Tensor,
    *,
    command_error_clip: float = 2.0,
    generator=None,
) -> torch.Tensor:
    """Perturb measured state, keeping target speed and executed action exact.

    The actor reconstructs target yaw from measured yaw rate plus yaw error.
    Independent noise on both slots would create a different command and
    activate the wrong expert, especially on straight commands at full DR.
    Recompute error from the same noisy yaw sensor instead.
    """
    noisy = observation + observation_randn_like(observation, generator) * std.unsqueeze(-1)
    noisy[:, 7] = observation[:, 7]
    noisy[:, 8] = (target_yaw - noisy[:, 2] / 0.25).clamp(
        -command_error_clip, command_error_clip)
    noisy[:, 19:25] = observation[:, 19:25]
    return noisy


def sensor_noise(observation, strength, gyro_bias, accel_bias, target_yaw, cfg, *, yaw_error_clip=2.0, generator=None):
    """Physical-unit channel noise; acceleration represented as gravity estimate error."""
    out = observation.clone()
    randn = lambda value: observation_randn_like(value, generator)
    scale = strength[:, None]
    out[:, :3] += .25 * (gyro_bias + randn(out[:, :3]) * cfg['gyro_noise_std_rad_s'] * scale)
    gravity = out[:, 3:6] + (accel_bias + randn(out[:, 3:6]) * cfg['accel_noise_std_mps2'] * scale) / 9.81
    out[:, 3:6] = torch.nn.functional.normalize(gravity, dim=-1)
    out[:, 9:13] += randn(out[:, 9:13]) * cfg['encoder_angle_std_rad'] * scale
    out[:, 13:19] += .05 * randn(out[:, 13:19]) * cfg['encoder_velocity_std_rad_s'] * scale
    out[:, 6] = (out[:, 6] + 2.1 * randn(strength) * cfg['forward_velocity_std_mps'] * strength).clamp(-2,2)
    out[:, 8] = (target_yaw - out[:, 2] / .25).clamp(-yaw_error_clip,yaw_error_clip)
    return out


def delayed_sensor_observation(history, delay_steps, current, target_yaw, *, yaw_error_clip=2.0):
    """Read integer-cycle sensor history, retaining present command/action slots.

    Rebase delayed speed error onto present command. A clipped historical speed
    error remains a censored measurement, as in the existing25-D contract.
    """
    ids = torch.arange(history.shape[0], device=history.device)
    out = history[ids, delay_steps.clamp(0, history.shape[1]-1)].clone()
    out[:,6] = (out[:,6] + (current[:,7]-out[:,7]) * (2.1/.5)).clamp(-2,2)
    out[:,7] = current[:,7]
    out[:,8] = (target_yaw-out[:,2]/.25).clamp(-yaw_error_clip,yaw_error_clip)
    out[:,19:25] = current[:,19:25]
    return out
