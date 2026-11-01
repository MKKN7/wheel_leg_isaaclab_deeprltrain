"""Bounded simulator-only excitation, never part of deterministic acceptance."""
import math
import torch

def validate_exploration(std, replay_enabled, distilling=False, stochastic_audit=False):
    if not math.isfinite(std) or not 0<=std<=.1:
        raise ValueError('Exploration std must be finite and in [0,.1] action units')
    if std and (not replay_enabled or distilling or stochastic_audit):
        raise ValueError('Exploration requires dedicated WM capture, no distillation or critic audit')

def excite(action,std,generator=None):
    if std==0:return action
    noise=torch.randn(action.shape,device=action.device,dtype=action.dtype,generator=generator)
    return action+noise.clamp(-3,3)*std
