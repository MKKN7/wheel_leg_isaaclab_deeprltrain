"""Opt-in training cost for the observed low-frequency fast-reverse pitch mode."""
import math
import torch


def reverse_pitch_oscillation_cost(command, pitch_rate, episode_steps, step_dt):
    """Dimensionless, bounded angular-rate energy after the acquisition phase.

    Uses physical body pitch rate, not a filtered actor observation. This is a
    reward only: it never modifies an observation, action or actuator command.
    It complements, but does not certify, the raw per-episode 3 degree p2p gate.
    """
    if not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError('Positive finite policy step required')
    active = ((command[..., 0] <= -.95) & (command[..., 0] >= -1.8)
              & (command[..., 2].abs() <= 1e-4)
              & (episode_steps * step_dt >= 3.0))
    energy = ((pitch_rate.abs() - .04).clamp_min(0) / .20).clamp_max(1).square()
    return energy * active.to(energy.dtype)
