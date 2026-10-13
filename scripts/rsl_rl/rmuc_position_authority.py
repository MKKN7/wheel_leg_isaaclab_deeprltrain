"""Opt-in post-clip authority head with optional command-local correction.

Keep legacy executed actions, then learn extra joint-side authority. Output is
still six actions in legacy units (0.22 rad/unit for the active leg channels at
1 m/s). A compatible consumer MUST allow the declared active-leg limits and
must not clip the entire vector to +/-1 beforehand.
"""
import torch
from torch import nn
from rmuc_state_router import StateResidualExperts, IsolatedTurnStateRouter

class PositionAuthorityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.positive = StateResidualExperts()
        self.negative = StateResidualExperts()
        self.local_correction = None
        # Only active links receive extra position range, wheel budget small.
        self.register_buffer('budget', torch.tensor([.15/.22, 0., .06, .15/.22, 0., .06]))
        self.register_buffer('limit', torch.tensor([1.+.15/.22, 1., 1., 1.+.15/.22, 1., 1.]))

    @staticmethod
    def gates(observation, speed_plateau_end=1.0):
        speed=observation[...,7]/.5
        yaw=observation[...,2]/.25+observation[...,8]
        smooth=IsolatedTurnStateRouter.smooth
        support=smooth(speed,.8,1.)*(1.-smooth(speed,speed_plateau_end,speed_plateau_end+.2))
        return support*smooth(yaw,.4501,.6),support*smooth(-yaw,.4501,.6)

    def correction(self, observation):
        end=getattr(self, 'speed_plateau_end', 1.0)
        pos,neg=self.gates(observation, end)
        correction=self.budget*(pos[...,None]*self.positive(observation)
                                +neg[...,None]*self.negative(observation))
        if self.local_correction is not None:
            correction=self.local_correction(observation, correction, self.budget, end)
        return correction

    def forward(self, observation, legacy_action):
        pos,neg=self.gates(observation, getattr(self, 'speed_plateau_end', 1.0))
        correction=self.correction(observation)
        # Preserve raw legacy output exactly outside the adaptation domain.
        # Inside, preserve its OLD executed value before adding trainable delta.
        adapted=legacy_action.clamp(-1.,1.)+correction
        adapted=torch.maximum(torch.minimum(adapted,self.limit),-self.limit)
        return torch.where((pos+neg)[...,None]>0,adapted,legacy_action)

def execute_authority_action(observation, action, speed_plateau_end=1.0):
    """Prototype consumer: expanded limits only in the explicit local domain."""
    pos,neg=PositionAuthorityHead.gates(observation, speed_plateau_end)
    expanded=action.new_tensor([1.+.15/.22,1.,1.,1.+.15/.22,1.,1.])
    limit=torch.where((pos+neg)[...,None]>0,expanded,torch.ones_like(expanded))
    return torch.maximum(torch.minimum(action,limit),-limit)


def checkpoint_authority_speed_plateau_end(state):
    """Explicit opt-in, stored in the actor state; legacy checkpoints stay exact."""
    end=float(state.get('rmuc_position_authority.speed_plateau_end', 1.0))
    if not 1.0 <= end <= 1.5:
        raise ValueError('Authority speed plateau must be finite and in [1.0, 1.5] m/s')
    return end
