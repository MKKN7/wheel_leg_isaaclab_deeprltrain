"""Opt-in deterministic26D teacher component; legacy25D runtime is unchanged.

source25 must be a complete deterministic actor, including its existing input
normalization and consumer-independent composition. PPO integration and export
are separate: this component does not implement a stochastic actor API.
"""
import copy
import torch
from torch import nn

class LegLengthConditionedActor(nn.Module):
    observation_dim = 26
    action_dim = 6

    def __init__(self, source25: nn.Module, action_budget):
        super().__init__()
        budget=torch.as_tensor(action_budget,dtype=torch.float32)
        if budget.shape!=(6,) or not torch.isfinite(budget).all() or (budget<0).any() or (budget>1).any():
            raise ValueError('Six finite nonnegative action-unit budgets <=1 required')
        self.source25=copy.deepcopy(source25).requires_grad_(False).eval()
        self.register_buffer('action_budget',budget.clone())
        self.length_network=nn.Sequential(nn.Linear(26,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,6))
        nn.init.zeros_(self.length_network[-1].weight)
        nn.init.zeros_(self.length_network[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        # Running statistics/dropout of the inherited actor must never drift.
        self.source25.eval()
        return self

    def forward(self, observation):
        if observation.shape[-1]!=26:
            raise ValueError('26D observation required; legacy25D cannot imply a length command')
        with torch.no_grad():
            baseline=self.source25(observation[...,:25])
        correction=torch.tanh(self.length_network(observation))*self.action_budget
        # Preserve the existing six action meanings; actuator clipping remains
        # in the shared consumer, not silently duplicated in this wrapper.
        return baseline+correction
