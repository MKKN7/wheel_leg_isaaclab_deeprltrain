"""Explicit26D experimental task implementation, not registered for training yet.

Uses current25D environment for all mechanics; transport delay is unchanged.
No automatic curriculum or inferred leg target is enabled here. A future runner
must install a26D actor/critic and set the command explicitly.
"""
import math
import torch
import gymnasium.spaces as spaces
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse
from .env import RmucWheelLegEnv
from .env_cfg import RmucWheelLegCommandEnvCfg
from .leg_length_terms import planar_lengths,mean_length_cost,append_target

@configclass
class RmucLegLengthCommandEnvCfg(RmucWheelLegCommandEnvCfg):
    observation_space=spaces.Box(low=-1.e6,high=1.e6,shape=(26,),dtype=float)
    state_space=spaces.Box(low=-1.e6,high=1.e6,shape=(26,),dtype=float)
    leg_length_command_range_m=(.30,.36)
    leg_length_initial_command_m=.34
    leg_length_tracking_weight=0.0  # No unreviewed reward scale by default.
    leg_length_tracking_scale_m=.02

class RmucLegLengthCommandEnv(RmucWheelLegEnv):
    def __init__(self,cfg,render_mode=None,**kwargs):
        lo,hi=cfg.leg_length_command_range_m
        if not .16<=lo<hi<=.36 or not lo<=cfg.leg_length_initial_command_m<=hi:
            raise ValueError('Explicit bounded leg-length curriculum required')
        if not math.isfinite(cfg.leg_length_tracking_weight) or cfg.leg_length_tracking_weight<0:
            raise ValueError('Nonnegative leg-length reward weight required')
        if not math.isfinite(cfg.leg_length_tracking_scale_m) or cfg.leg_length_tracking_scale_m<=0:
            raise ValueError('Positive leg-length reward scale required')
        super().__init__(cfg,render_mode,**kwargs)
        # Current contract disables whole-observation delay. Reject untested
        # delay modes rather than mixing delayed25D with an undelayed target.
        if self._obs_delay_steps>0 or torch.any(self._domain_observation_delay_steps>0) or torch.any(self._sensor_delay_steps>0):
            raise ValueError('26D observation-delay alignment not yet supported')
        self._length_command=torch.full((self.num_envs,),cfg.leg_length_initial_command_m,device=self.device)
        # Read actual root-side hip anchor transforms from the USD, not an
        # assumed .34 root height or a copied linkage length.
        from pxr import Usd,UsdPhysics
        stage=Usd.Stage.Open(cfg.robot_cfg.spawn.usd_path);anchors=[]
        for name in ['left_hip_pitch','right_hip_pitch']:
            found=[prim for prim in stage.Traverse() if prim.GetName()==name and prim.IsA(UsdPhysics.Joint)]
            if len(found)!=1:raise ValueError('Unique hip joint required')
            joint=UsdPhysics.Joint(found[0]);parent=joint.GetBody0Rel().GetTargets()
            if len(parent)!=1 or parent[0].name!='robot':raise ValueError('Hip root frame mismatch')
            anchors.append(list(joint.GetLocalPos0Attr().Get()))
        self._length_hip_anchors=torch.tensor(anchors,device=self.device)

    def set_leg_length_command(self,target):
        value=torch.as_tensor(target,device=self.device,dtype=torch.float32).expand(self.num_envs)
        lo,hi=self.cfg.leg_length_command_range_m
        if not torch.isfinite(value).all() or (value<lo).any() or (value>hi).any():
            raise ValueError('Leg command outside explicit task envelope')
        self._length_command.copy_(value)

    def _get_observations(self):
        result=super()._get_observations()
        if (self._obs_delay_steps>0 or torch.any(self._domain_observation_delay_steps>0)
                or torch.any(self._sensor_delay_steps>0)):
            raise ValueError('Active observation delay needs explicit26D target alignment')
        target=getattr(self,'_length_command',None)
        if target is None:target=torch.full((self.num_envs,),self.cfg.leg_length_initial_command_m,device=self.device)
        return {'policy':append_target(result['policy'],target)}

    def measured_leg_lengths(self):
        pos=self.robot.data.body_pos_w.torch
        delta=pos[:,self._wheel_body_ids]-pos[:,self._root_body_id].unsqueeze(1)
        quat=self.robot.data.body_quat_w.torch[:,self._root_body_id].unsqueeze(1).expand(-1,2,-1)
        return planar_lengths(quat_apply_inverse(quat,delta),self._length_hip_anchors)

    def _additional_reward_before_clip(self, reward):
        if not hasattr(self,'_length_hip_anchors'):return reward
        lengths=self.measured_leg_lengths()
        cost=mean_length_cost(lengths,self._length_command,self.cfg.leg_length_tracking_scale_m)
        weight=self.cfg.leg_length_tracking_weight
        adjusted=reward-weight*cost
        clipped=torch.clamp(adjusted,self.cfg.reward_clip_min,self.cfg.reward_clip_max)
        self.extras.setdefault('log',{}).update({
            'leg_command/mean_length_m':lengths.mean(),
            'leg_command/mean_abs_error_m':(lengths.mean(-1)-self._length_command).abs().mean(),
            'leg_command/reward_cost':(weight*cost).mean(),
            'leg_command/final_clip_fraction':(clipped!=adjusted).float().mean()})
        return adjusted
