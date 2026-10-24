"""Tensor helpers for the explicit leg-length command, independent of Isaac."""
import torch

def planar_lengths(wheel_in_root, hip_in_root):
    # Authored source axes: canonical forward=-Z, up=Y; ignore lateral X.
    delta=wheel_in_root-hip_in_root
    return torch.linalg.vector_norm(delta[...,1:3],dim=-1)

def mean_length_cost(lengths,target,scale=.02):
    e=(lengths.mean(dim=-1)-target)/scale
    return torch.sqrt(1+e.square())-1

def append_target(obs25,target):
    if obs25.shape[-1]!=25 or target.shape!=obs25.shape[:-1]:
        raise ValueError('25D observations and per-env targets required')
    return torch.cat((obs25,((target-.34)/.10).unsqueeze(-1)),dim=-1)
