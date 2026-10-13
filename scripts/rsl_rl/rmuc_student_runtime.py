"""Fixed student inference; all task/control semantics remain with the consumer."""
import torch
from torch import nn
class StudentActor(nn.Module):
    def __init__(self,payload):
        super().__init__()
        assert payload['architecture']==[25,128,128,6]
        self.net=nn.Sequential(nn.Linear(25,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,6))
        self.net.load_state_dict(payload['student_state_dict'],strict=True)
        self.register_buffer('mean',payload['mean'].clone())
        self.register_buffer('std',payload['std'].clone())
        assert torch.isfinite(self.mean).all() and torch.isfinite(self.std).all() and (self.std>0).all()
    def forward(self,obs,stochastic_output=False):
        if stochastic_output:raise ValueError('Student is deterministic')
        return self.net((obs['policy']-self.mean)/self.std)
