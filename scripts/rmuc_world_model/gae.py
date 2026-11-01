"""Detached on-policy lambda-return targets; no old-policy Monte Carlo critic."""
import torch
from torch import nn

@torch.no_grad()
def lambda_returns(rewards, values, continuation, gamma=.99, lam=.95):
    """Time-major [H,B]; values[H] bootstraps truncation, continuation=0 ends episodes.

    A time limit is a truncation (continuation=1), not a failure termination.
    Soft survival probabilities are supported. Never supply reset-state values
    for the final transition of an episode.
    """
    if values.shape != (rewards.shape[0]+1, *rewards.shape[1:]) or continuation.shape != rewards.shape:
        raise ValueError('Return tensor shape mismatch')
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError('Invalid discount/lambda')
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError('Nonfinite return inputs')
    if not torch.isfinite(continuation).all() or ((continuation < 0) | (continuation > 1)).any():
        raise ValueError('Invalid continuation probabilities')
    result = torch.empty_like(rewards)
    tail = values[-1]
    for t in range(len(rewards)-1, -1, -1):
        tail = rewards[t] + gamma*continuation[t]*((1-lam)*values[t+1] + lam*tail)
        result[t] = tail
    return result

class ReturnCritic(nn.Module):
    def __init__(self, mean, scale, input_mask=None):
        super().__init__()
        self.register_buffer('mean', mean.clone())
        if input_mask is not None:
            if input_mask.shape!=mean.shape or not ((input_mask==0)|(input_mask==1)).all():raise ValueError('Invalid critic input mask')
            self.register_buffer('input_mask',input_mask.clone())
        self.register_buffer('scale', scale.clone().clamp_min(1e-6))
        self.net = nn.Sequential(nn.Linear(len(mean),256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        normalized=(x-self.mean)/self.scale
        if hasattr(self,'input_mask'):normalized=normalized*self.input_mask
        return self.net(normalized).squeeze(-1)
