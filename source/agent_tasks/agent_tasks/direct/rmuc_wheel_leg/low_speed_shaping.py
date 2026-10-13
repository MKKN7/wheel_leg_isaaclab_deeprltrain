"""Optional dense small-turn acquisition cost without a flat wrong-way tail."""
import torch

def low_speed_yaw_cost(command, measured_yaw, max_yaw_rad_s=.4501):
    if not 0.0 < max_yaw_rad_s <= 2.5:
        raise ValueError("dense yaw shaping range must be in (0, 2.5] rad/s")
    mask=(command[...,0].abs()<=1.5)&(command[...,2].abs()>1e-4)&(command[...,2].abs()<=max_yaw_rad_s)
    error=(measured_yaw-command[...,2])/.1
    # Pseudo-Huber: finite derivative at zero, linear tail for large misses.
    cost=torch.sqrt(1.+error.square())-1.
    return torch.where(mask,cost,torch.zeros_like(cost))
