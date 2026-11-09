"""Opt-in static straight-speed cohorts; no reset RNG or within-episode switches."""
import math
import torch

def validate_speeds(values):
    if not values:return ()
    if len(set(values))!=len(values) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not 1.6<=x<=2.0 for x in values):
        raise ValueError('Unique finite highstraight speeds in[1.6,2] required')
    return tuple(float(x) for x in values)

def apply_cohort(vx,yaw,rehearsal,continuous,speeds):
    if not speeds:return vx,yaw
    values=torch.as_tensor(speeds,device=vx.device,dtype=vx.dtype)
    assigned=values[torch.arange(vx.numel(),device=vx.device)%len(speeds)]
    mask=~rehearsal & ~continuous
    return torch.where(mask,assigned,vx),torch.where(mask,torch.zeros_like(yaw),yaw)
