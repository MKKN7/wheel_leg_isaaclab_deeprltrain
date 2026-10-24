"""Opt-in reward cost for wheel action chatter in truly settled idle states."""
import math
import torch

def settled_wheel_cost(action, previous, command, measured_vx, pitch_rate, tilt_rad,
                       *, action_scale=.02):
    if not math.isfinite(action_scale) or action_scale<=0:
        raise ValueError('Positive finite normalization required')
    # True simulation state gates the reward only, never observations or outputs.
    zero=command.abs().amax(-1)<=1e-6
    settled=zero&(measured_vx.abs()<.03)&(pitch_rate.abs()<.1)&(tilt_rad.abs()<math.radians(3))
    delta=(action[..., [2,5]]-previous[..., [2,5]])/action_scale
    cost=(torch.sqrt(1+delta.square())-1).mean(-1).clamp_max(4.)
    return torch.where(settled,cost,torch.zeros_like(cost))
