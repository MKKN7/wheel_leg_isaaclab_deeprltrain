"""Opt-in zero-to-target command entry, shared by training and evaluation.

This is input shaping, not feedback or a motor-action filter. The caller must
use the returned command consistently for observation, control and reward.
"""
import math
import torch


def reverse_entry_command(target, episode_age, dt, duration_s, eligible=None):
    if not math.isfinite(duration_s) or not 0 <= duration_s <= 4:
        raise ValueError('Reverse entry duration must be finite within0..4 seconds')
    if duration_s == 0:
        return target
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite policy timestep required')
    if target.ndim != 2 or target.shape[1] != 3 or episode_age.shape != target.shape[:1]:
        raise ValueError('Nx3 commands and matching per-environment ages required')
    if eligible is not None and (eligible.shape != episode_age.shape or eligible.dtype != torch.bool):
        raise ValueError('Boolean per-environment entry mask required')
    active = (target[:, 0] < 0) & (target[:, 1:] == 0).all(-1)
    if eligible is not None:
        active = active & eligible
    fraction = (episode_age.to(target.dtype).clamp_min(0) * dt / duration_s).clamp(0, 1)
    result = target.clone()
    result[:, 0] = torch.where(active, target[:, 0] * fraction, target[:, 0])
    return result
