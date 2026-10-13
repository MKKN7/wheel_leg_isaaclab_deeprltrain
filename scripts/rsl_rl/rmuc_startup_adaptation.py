"""Opt-in bounded state-dependent startup residual; no timer or command outer loop."""
import torch
from torch import nn

class StartupAdaptation(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(25,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,6))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('budget',torch.tensor([.4,.15,.6,.4,.15,.6]))
        self.register_buffer('input_mean',torch.zeros(25))
        self.register_buffer('input_scale',torch.ones(25))

    @staticmethod
    def gate(obs):
        def fade(x,low,high):
            t=((x-low)/(high-low)).clamp(0.,1.)
            return 1.-t*t*(3.-2.*t)
        speed=obs[...,7]/.5
        yaw=obs[...,2]/.25+obs[...,8]
        estimated_vx=speed-obs[...,6]/2.1
        return fade((speed-1.).abs(),.02,.1)*fade((yaw-1.4).abs(),.02,.1)*fade(estimated_vx.abs(),.35,.75)

    def forward(self,obs,source_action):
        inputs=((obs-self.input_mean)/self.input_scale).clamp(-10.,10.)
        delta=self.gate(obs)[...,None]*self.budget*torch.tanh(self.network(inputs))
        return source_action+delta
