"""Slow bootstrap parameters; no gradients through target creation or update."""
import torch

@torch.no_grad()
def update_target(target,online,tau):
    if not 0<tau<=1:raise ValueError('EMA tau must be in (0,1]')
    a=dict(target.named_parameters());b=dict(online.named_parameters())
    if a.keys()!=b.keys():raise ValueError('Target architecture mismatch')
    for name,p in a.items():p.lerp_(b[name],tau)
