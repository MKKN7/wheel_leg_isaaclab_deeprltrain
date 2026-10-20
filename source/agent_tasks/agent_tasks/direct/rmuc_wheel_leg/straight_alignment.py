"""Measured bilateral linkage alignment, only for straight/idle commands."""
import math
import torch

def alignment_cost(q,command_yaw,actual_yaw,tolerance_deg=2.,scale_deg=6.,*,
                   command_vx=None,reverse_scale_deg=0.):
    if not math.isfinite(tolerance_deg) or tolerance_deg<0 or not math.isfinite(scale_deg) or scale_deg<=0:
        raise ValueError('Finite nonnegative tolerance and positive scale required')
    if not math.isfinite(reverse_scale_deg) or reverse_scale_deg < 0:
        raise ValueError('Reverse alignment scale must be finite and nonnegative')
    if reverse_scale_deg and command_vx is None:
        raise ValueError('Reverse alignment scale requires signed vx command')
    # q in policy order: model right physical-left, hip axes mirrored.
    difference=torch.stack((q[...,0]-q[...,3],q[...,1]+q[...,4]),dim=-1)
    scale = math.radians(scale_deg)
    if reverse_scale_deg:
        # Opt-in reward shaping only. Keep forward/idle and all actuator maps
        # unchanged; yaw gating below still excludes every turn command.
        scale = torch.where(command_vx < 0,
            torch.full_like(command_vx, math.radians(reverse_scale_deg)),
            torch.full_like(command_vx, scale))[..., None]
    excess=(difference.abs()-math.radians(tolerance_deg)).clamp_min(0)/scale
    # Smooth deadband, bounded [0,1), retains a slope beyond the deadband.
    cost=(excess.square()/(1+excess.square())).mean(-1)
    # No penalty for any nonzero turn command, including slow turn/spin.
    # Fade during residual yaw motion after exiting a turn.
    t=((actual_yaw.abs()-.05)/.15).clamp(0,1)
    gate=(command_yaw.abs()<=1e-4).to(q.dtype)*(1-t*t*(3-2*t))
    return cost*gate,difference,gate
