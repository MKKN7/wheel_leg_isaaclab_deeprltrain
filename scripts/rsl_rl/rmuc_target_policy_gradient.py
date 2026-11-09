"""Opt-in actor-only domain mask; critic targets and rehearsal KL unchanged."""
import torch

def target_gate_from_commands(observations, targets, tolerance=1e-3):
    command=torch.stack((observations[...,7]/.5,observations[...,2]/.25+observations[...,8]),-1)
    table=torch.as_tensor(targets,device=command.device,dtype=command.dtype)
    if table.ndim!=2 or table.shape[1]!=2 or not len(table):raise ValueError('Nonempty vx/yaw targets required')
    if tolerance not in (1e-3,1e-5):raise ValueError('Explicit standard or exact-command tolerance required')
    return ((command[...,None,:]-table).abs().amax(-1)<tolerance).any(-1).to(command.dtype)

def mask_target_advantages(advantages,gate):
    if gate.shape!=advantages.shape[:-1]:raise ValueError('Target gate must match rollout layout')
    if not torch.isfinite(advantages).all() or not torch.isfinite(gate).all():raise ValueError('Nonfinite target policy gradient input')
    mask=gate>0
    if not mask.any():raise ValueError('No target-domain samples in rollout')
    return advantages*mask[...,None],{'target_samples':int(mask.sum()),'total_samples':mask.numel(),'amplification':1.,'scope':'Only PPO actor surrogate masked; no target advantage recentering, no critic target change, rehearsal KL unchanged'}
