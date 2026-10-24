"""RSL-RL26D actor for frozen teacher plus bounded leg-length conditioning.

Explicit new runner/config required; never reconstruct a legacy25D checkpoint
as this model implicitly. The source checkpoint is required at construction.
"""
from pathlib import Path
from torch import nn
from rsl_rl.models import MLPModel
from rmuc_leg_length_actor import LegLengthConditionedActor

class DeterministicTeacher25(nn.Module):
    def __init__(self,actor):
        super().__init__();self.actor=actor
    def forward(self,x):
        return self.actor({'policy':x},stochastic_output=False)

class LegLengthPPOActor(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,*,source_checkpoint,
                 action_budget=(.2,.2,.03,.2,.2,.03),distribution_cfg=None,
                 obs_normalization=False,**kwargs):
        if obs_normalization:
            raise ValueError('Preserve source25 normalization; do not renormalize appended target')
        if output_dim!=6:
            raise ValueError('Six actions required')
        super().__init__(obs,obs_groups,obs_set,output_dim,
                         obs_normalization=False,distribution_cfg=distribution_cfg,**kwargs)
        if self.obs_dim!=26 or (self.distribution is not None and self.distribution.input_dim!=6):
            raise ValueError('26D input and mean-only six-action distribution required')
        from verify_rmuc_export import restore_actor
        teacher=DeterministicTeacher25(restore_actor(Path(source_checkpoint)))
        self.mlp=LegLengthConditionedActor(teacher,action_budget)
        # Match the inherited std if present, but freeze exploration during
        # initial physical adaptation. No optimizer can update frozen teacher.
        if self.distribution is not None:
            self.distribution.requires_grad_(False)

    def as_jit(self):
        raise RuntimeError('26D combined export needs explicit IO/consumer audit; use dedicated export workflow')

    def as_onnx(self,verbose=False):
        raise RuntimeError('26D combined export not yet certified; do not export legacy25D wrapper')

class LegLengthPPOCritic(MLPModel):
    """Expand the original first layer by one zero-weight command column."""
    def __init__(self,obs,obs_groups,obs_set,output_dim,*,source_checkpoint,
                 hidden_dims=(256,128,64),obs_normalization=False,**kwargs):
        import torch
        if obs_normalization or output_dim!=1:
            raise ValueError('Scalar unnormalized critic migration required')
        super().__init__(obs,obs_groups,obs_set,output_dim,hidden_dims=hidden_dims,
                         obs_normalization=False,**kwargs)
        if self.obs_dim!=26:raise ValueError('26D critic input required')
        state=torch.load(source_checkpoint,map_location='cpu',weights_only=False)['critic_state_dict']
        expanded={k:v.clone() for k,v in state.items()}
        first=state['mlp.0.weight']
        if first.shape[1]!=25:raise ValueError('Expected source25D critic')
        expanded['mlp.0.weight']=torch.cat((first,torch.zeros_like(first[:,:1])),dim=1)
        self.load_state_dict(expanded,strict=True)
