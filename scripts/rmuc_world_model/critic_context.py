"""Reward-specific privileged critic context; never changes policy observations."""
import torch

def features(state, context, event, event_conditioned):
    return torch.cat([state,context,event] if event_conditioned else [state,context],-1)

def validate_reward_schema(checkpoint, event_conditioned, overshoot_weight):
    if checkpoint.get('event_conditioned',False)!=event_conditioned:
        raise ValueError('Critic event schema differs; initialize a fresh critic for changed reward')
    if checkpoint.get('overshoot_weight',0.)!=overshoot_weight:
        raise ValueError('Critic reward changed; initialize a fresh critic')
